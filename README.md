# Engineering Intelligence Hub

A code-RAG system over the Apache Airflow codebase (~65,000 chunks: code, docs, GitHub issues, PRs). Answers questions with file-level citations.

Built incrementally over six weeks to measure which retrieval techniques actually move the needle. **Several techniques I added didn't help.** This README documents what worked, what didn't, and what the real numbers look like — including the ones I'd rather not show.

## The real numbers

Evaluated on 300 questions × 3 runs each = 900 evaluations, with LLM-as-judge scoring 0–3.

```
                  p_hit@1   p_hit@3   p_hit@10   p_MRR    judge (0-3)
hybrid             0.56      0.85      0.88      0.70     1.50
routed             0.56      0.85      0.88      0.70     1.65   ← default
hybrid_rerank      0.56      0.62      0.82      0.63     1.85   ← worse on hit@3
hybrid_hyde        0.55      0.73      0.79      0.65     1.67
```

`p_hit@k` = path-only: did the right *file* appear at rank k. Strict `hit@k` (exact chunk) is much lower (~14% hit@1) — many auto-generated questions expect one specific chunk when several chunks from the same file would equally answer the question.

**What this tells me:**
- The system puts the right file at #1 about **56% of the time** and finds it in top-10 about **88% of the time** — solid for a single-corpus, single-language portfolio project.
- **The cross-encoder reranker hurts hit@3 by 24 points.** I built it, measured it, and it loses. Kept in the code as `--method hybrid_rerank` for inspection but `routed` doesn't use it.
- **HyDE underperforms plain hybrid.** Same story: built, measured, doesn't help on this eval set.
- The "router" that picks per-question method **barely fires** (3% of questions hit a non-default path). Essentially equivalent to plain hybrid on this corpus.
- LLM-judge scores cluster at **1.5–1.8 out of 3** — answers are *mostly correct but partial*. Not embarrassing, not impressive.

## What works (and why I'd keep it)

1. **Tree-sitter code chunking** — chunks Python by function/class boundary, not by character count. Methods are retrievable with their bodies, not just docstrings. This was the largest single quality jump.
2. **Hybrid retrieval (BM25 + vector with reciprocal rank fusion)** — neither alone matches the combination. BM25 catches exact identifiers; vector catches conceptual matches.
3. **Symbol-aware injection** — when a question names an identifier like `GCSHook.upload`, that chunk is pinned to the top. Targeted fix for a real failure mode discovered through eval.
4. **GitHub issues + PRs in the same corpus** — cross-references like "has anyone hit this error?" work because issues are searchable alongside code.

## What didn't work

1. **Cross-encoder reranker (`bge-reranker-base`)** — 4GB dep, 300ms per query, makes hit@3 worse. Probably needs a code-specific reranker; web-trained models don't transfer.
2. **HyDE query rewriting** — extra Claude call per query, slightly worse on this eval. Possibly useful for purely conceptual questions but the gain doesn't justify the cost.
3. **Agentic retrieval (Claude as tool-using agent)** — qualitatively impressive on multi-hop questions ("when X fails, what callbacks fire and in what order?") but **expensive** (5× the cost of single-shot) and my single-shot eval metrics couldn't measure its actual value. The metric was the problem, not the technique.

## Architecture

```
                                    ┌─────────────────┐
Question ──┬─► router (regex) ──┬──►│ vector + BM25   │
           │                    │   │ hybrid (RRF)    │
           │                    │   │ + symbol-pin    │
           │                    │   └────────┬────────┘
           │                    │            │
           │                    └──► top-k ──┘
           │                              │
           │                              ▼
           │                  ┌─────────────────────┐
           └─────────────────►│ Claude Haiku 4.5    │
                              │ + numbered context  │
                              │ + citation rules    │
                              └──────────┬──────────┘
                                         ▼
                                  answer + [#N] citations
```

### Components

| File | Role |
|---|---|
| [src/eih/chunker.py](src/eih/chunker.py) | Tree-sitter Python parser — extracts functions, classes, methods with full bodies |
| [src/eih/ingest.py](src/eih/ingest.py) | Walks the repo, dispatches to chunker for `.py` or markdown extractor for `.md`/`.rst` |
| [src/eih/github.py](src/eih/github.py) | Fetches closed GitHub issues + PRs as chunks |
| [src/eih/store.py](src/eih/store.py) | Chroma + OpenAI embeddings + hybrid retrieval (BM25 + vector + RRF + symbol pin) |
| [src/eih/bm25.py](src/eih/bm25.py) | BM25 index with identifier-aware tokenization (handles `snake_case` and `CamelCase`) |
| [src/eih/rerank.py](src/eih/rerank.py) | Cross-encoder reranker (experimental — doesn't help on this eval) |
| [src/eih/hyde.py](src/eih/hyde.py) | HyDE query rewriting (experimental — doesn't help) |
| [src/eih/agentic.py](src/eih/agentic.py) | Tool-using agent loop (qualitatively impressive, hard to measure) |
| [src/eih/router.py](src/eih/router.py) | Heuristic regex classifier that picks retrieval method per question |
| [src/eih/answer.py](src/eih/answer.py) | Composes the final answer with citation discipline |
| [src/eih/eval/](src/eih/eval/) | Eval harness — questions, metrics, judge, runner, scorecard |
| [src/eih/cli.py](src/eih/cli.py) | Typer CLI: `ingest`, `ingest-github`, `ask`, `eval`, `gen-eval` |

## Setup

```bash
# 1. Install
uv venv && source .venv/bin/activate
uv pip install -e .

# 2. API keys — needs Anthropic (Claude) + OpenAI (embeddings)
cp .env.example .env
# edit with your keys

# 3. Clone the target repo
mkdir -p data/repos
git clone --depth 1 https://github.com/apache/airflow data/repos/airflow

# 4. Ingest code + docs (~30 min, ~$0.50 in OpenAI embeddings)
eih ingest data/repos/airflow

# 5. Ingest GitHub issues + PRs (~15 min, $0.01)
eih ingest-github apache/airflow --max-items 1000

# 6. Ask
eih ask "How does GCSHook.upload handle chunked uploads?"
eih ask "What's the difference between LocalExecutor and CeleryExecutor?"
```

## Reproducing the eval

```bash
# Quick: original 17 hand-curated questions
eih eval --questions data/eval/questions.yaml

# Real: 100-question sample from auto-generated set, 3 runs, mean ± stddev
eih eval --questions data/eval/generated_1000.yaml --sample 100 --runs 3 --methods routed,hybrid

# Compare all methods
eih eval --questions data/eval/combined.yaml --sample 200 --runs 1 \
    --methods routed,hybrid,hybrid_rerank,hybrid_hyde
```

Baselines saved per-iteration in `data/eval/baseline_*.json` so you can diff between weeks.

## Honest assessment

This is **above average for a personal RAG project** because it has an eval at all and the numbers are honestly reported. It is **not** production-quality:

- No retry/circuit-breaker logic on the Anthropic API
- No model fallback (Haiku-only)
- No real-user testing — nobody besides me has typed a question into it
- Eval is 379 questions; production RAG systems eval on thousands
- Auto-generated questions have a known answer chunk by construction (slight leakage)
- Single language (Python), single corpus (Airflow)

The journey is the artifact: I shipped six weeks of measured iteration, found several "best practice" techniques don't transfer to code RAG, and learned my own eval was undersized and under-reporting quality by 4x.

## Roadmap (what shipped, what didn't)

| Week | Done | Notes |
|------|------|-------|
| 1 | ✅ | Naive RAG: markdown + Python docstrings, embed, retrieve, answer with citations |
| 2 | ✅ | Tree-sitter code chunking + hybrid retrieval (BM25 + vector + RRF) |
| 2.5 | ✅ | Symbol-aware injection — pin chunks by exact identifier match |
| 3 | ✅ | Eval harness: 17 hand-curated questions, hit@k, MRR, LLM-judge |
| 4 | ⚠️ | Cross-encoder reranker + HyDE — both built, **neither improved the numbers** |
| 4.5 | ✅ | Question-shape router — useful but rarely fires on this corpus |
| 5 | ✅ | GitHub issues + PRs ingested (capped at 1000 due to pagination) |
| 6 | ⚠️ | Agentic retrieval — qualitatively transformative, measurement broke |
| 6.5 | ✅ | Eval expansion to 1000 auto-generated + 18 hand-written multi-hop questions |
| 7 | ✅ | Streamlit web UI |
| 8 | ✅ | This README, honest numbers, model fallback |

## License

MIT.
