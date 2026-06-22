"""Typer CLI: `eih ingest <repo_path>`, `eih ask "..."`, `eih eval`."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from .answer import ask as ask_fn
from .ingest import walk_repo
from .store import add_chunks, get_collection, rebuild_bm25_from_collection, reset_collection

app = typer.Typer(add_completion=False, help="Engineering Intelligence Hub — RAG over Apache Airflow with measured retrieval methods.")
console = Console()


@app.command()
def ingest(
    repo_path: Path = typer.Argument(..., exists=True, file_okay=False, help="Local path to the repo (e.g. a clone of apache/airflow)."),
    reset: bool = typer.Option(True, help="Drop and recreate the collection before ingesting."),
):
    """Walk REPO_PATH, extract docs + docstrings, embed, store in Chroma."""
    if reset:
        console.print("[yellow]Resetting collection...[/yellow]")
        reset_collection()
    console.print(f"[cyan]Walking {repo_path}...[/cyan]")
    chunks = list(tqdm(walk_repo(repo_path), desc="extracting", unit="chunk"))
    console.print(f"Extracted [bold]{len(chunks)}[/bold] chunks. Embedding + storing...")
    n = add_chunks(chunks)
    console.print(f"[green]Done. Stored {n} chunks.[/green]")


@app.command()
def ask(
    question: str = typer.Argument(...),
    k: int = typer.Option(6, help="Number of context snippets to retrieve."),
    method: str = typer.Option(
        "routed",
        help=(
            "Retrieval method. RECOMMENDED: routed (default) or hybrid. "
            "EXPERIMENTAL (worse on the eval set): hybrid_rerank, hybrid_hyde, "
            "hybrid_rerank_hyde, agentic. RAW: vector, bm25."
        ),
    ),
):
    """Ask a question against the ingested corpus."""
    effective = method
    if method == "routed":
        from .router import route
        effective = route(question)
        console.print(f"[dim]router → {effective}[/dim]")
    result = ask_fn(question, k=k, method=effective)
    console.print(Panel(result["answer"], title=f"Answer ({method})", border_style="green"))
    console.print("\n[bold]Sources:[/bold]")
    for i, h in enumerate(result["hits"], 1):
        m = h["meta"]
        sym = f" · {m['symbol']}" if m.get("symbol") else ""
        src = h.get("source", "")
        # GH issues/PRs use synthetic paths — render as a clickable URL instead.
        if m["source_path"].startswith("_gh/"):
            parts = m["source_path"].split("/", 4)  # ['_gh', owner, repo, kind, num]
            if len(parts) == 4:  # _gh/repo/kind/num (we store as _gh/{repo}/{kind}/{num})
                _, repo, kind_seg, num = parts
                url = f"https://github.com/apache/{repo}/{'pull' if kind_seg=='pr' else 'issues'}/{num}"
                console.print(f"  [dim][#{i}][/dim] [magenta]{src}[/magenta] [link={url}]{url}[/link] ({m['kind']}{sym})")
                continue
        console.print(f"  [dim][#{i}][/dim] [magenta]{src}[/magenta] {m['source_path']}:{m['start_line']}-{m['end_line']} ({m['kind']}{sym})")


@app.command("ingest-github")
def ingest_github(
    repo: str = typer.Argument(..., help="owner/repo, e.g. apache/airflow"),
    max_items: int = typer.Option(5000, help="Max closed issues+PRs to fetch (most-recently-updated first)."),
):
    """Fetch closed issues + PRs from GitHub, add to the existing collection, rebuild BM25."""
    from .github import fetch_items, item_to_chunk
    if "/" not in repo:
        raise typer.BadParameter("repo must be owner/repo, e.g. apache/airflow")
    owner, name = repo.split("/", 1)
    console.print(f"[cyan]Fetching up to {max_items} closed items from {repo}...[/cyan]")
    chunks = []
    seen: set[str] = set()
    for item in tqdm(fetch_items(owner, name, max_items=max_items), total=max_items, desc="github"):
        c = item_to_chunk(item, name)
        if c is None:
            continue
        cid = c.id()
        if cid in seen:
            continue
        seen.add(cid)
        chunks.append(c)
    console.print(f"Got {len(chunks)} chunks. Embedding + adding to collection...")
    n = add_chunks(chunks, build_bm25=False)
    console.print(f"[green]Added {n} chunks. Rebuilding BM25 from full collection...[/green]")
    rebuilt = rebuild_bm25_from_collection()
    console.print(f"[green]BM25 rebuilt over {rebuilt} chunks. (Collection total: {get_collection().count()})[/green]")


@app.command("gen-eval")
def gen_eval(
    n: int = typer.Option(1000, help="Target number of questions to generate."),
    output: Path = typer.Option(Path("data/eval/generated_1000.yaml"), help="Output YAML path."),
    seed: int = typer.Option(42, help="Random seed for chunk sampling."),
):
    """Sample chunks from Chroma and ask Claude to write a question per chunk."""
    from .eval.generate import generate
    console.print(f"[cyan]Generating up to {n} questions → {output}[/cyan]")
    written = generate(n=n, output=output, seed=seed)
    console.print(f"[green]Wrote {written} questions to {output}[/green]")


@app.command()
def eval(
    questions: Path = typer.Option(Path("data/eval/questions.yaml"), help="Path to eval questions YAML."),
    methods: str = typer.Option("hybrid,vector,bm25", help="Comma-separated retrieval methods."),
    k: int = typer.Option(10, help="Top-k to retrieve per question."),
    skip_judge: bool = typer.Option(False, help="Skip LLM-as-judge (retrieval metrics only)."),
    runs: int = typer.Option(1, help="Re-run the full eval N times to surface run-to-run variance."),
    sample: int = typer.Option(0, help="If >0, sample this many questions from the eval set (random)."),
    output_dir: Path = typer.Option(Path("data/eval/results"), help="Where to save the run JSON."),
):
    """Run the eval set across one or more retrieval methods, print + save a scorecard."""
    import random
    from .eval.runner import (
        aggregate, aggregate_by_category, load_eval_set, run_eval, save_results,
    )

    qs = load_eval_set(questions)
    if sample > 0 and sample < len(qs):
        qs = random.Random(42).sample(qs, sample)
    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    console.print(
        f"[cyan]Running {len(qs)} questions × {len(method_list)} methods × {runs} runs "
        f"(k={k}, judge={'off' if skip_judge else 'on'})[/cyan]"
    )

    results = run_eval(qs, method_list, k=k, skip_judge=skip_judge, runs=runs)
    summary = aggregate(results)
    by_cat = aggregate_by_category(results)

    def fmt(pair, decimals=2):
        if pair is None:
            return "—"
        m, s = pair
        if runs > 1:
            return f"{m:.{decimals}f}±{s:.{decimals}f}"
        return f"{m:.{decimals}f}"

    # Two tables: path-only (user experience) first, strict second
    p_table = Table(
        title=f"Path-only: 'did the right FILE surface?' — {runs} run(s)",
        show_header=True, header_style="bold green",
    )
    p_table.add_column("method")
    p_table.add_column("Q", justify="right")
    p_table.add_column("hit@1", justify="right")
    p_table.add_column("hit@3", justify="right")
    p_table.add_column("hit@10", justify="right")
    p_table.add_column("MRR", justify="right")
    p_table.add_column("path_cov", justify="right")
    p_table.add_column("judge", justify="right")
    for method, s in summary.items():
        p_table.add_row(
            method, str(s["n_questions"]),
            fmt(s.get("p_hit@1")), fmt(s.get("p_hit@3")), fmt(s.get("p_hit@10")),
            fmt(s.get("p_mrr")), fmt(s["path_cov"]), fmt(s["judge_avg"]),
        )
    console.print(p_table)

    s_table = Table(
        title=f"Strict: 'did the EXACT expected chunk surface?' — {runs} run(s)",
        show_header=True, header_style="bold magenta",
    )
    s_table.add_column("method")
    s_table.add_column("Q", justify="right")
    s_table.add_column("hit@1", justify="right")
    s_table.add_column("hit@3", justify="right")
    s_table.add_column("hit@10", justify="right")
    s_table.add_column("MRR", justify="right")
    s_table.add_column("symbol_cov", justify="right")
    for method, s in summary.items():
        s_table.add_row(
            method, str(s["n_questions"]),
            fmt(s["hit@1"]), fmt(s["hit@3"]), fmt(s["hit@10"]),
            fmt(s["mrr"]), fmt(s["symbol_cov"]),
        )
    console.print(s_table)

    for method, cats in by_cat.items():
        ct = Table(title=f"By category — {method}", show_header=True, header_style="bold cyan")
        ct.add_column("category")
        ct.add_column("n", justify="right")
        ct.add_column("hit@3", justify="right")
        ct.add_column("MRR", justify="right")
        ct.add_column("path_cov", justify="right")
        for cat, s in sorted(cats.items()):
            ct.add_row(cat, str(int(s["n"])), f"{s['hit@3']:.2f}", f"{s['mrr']:.2f}", f"{s['path_cov']:.2f}")
        console.print(ct)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"run_{stamp}.json"
    save_results(results, summary, out_path)
    console.print(f"[green]Saved {out_path}[/green]")


if __name__ == "__main__":
    app()
