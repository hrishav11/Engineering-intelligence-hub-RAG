"""Generate eval questions by sampling chunks from Chroma and asking Claude
to write a realistic user question each chunk would answer.

Trade-off: every question has a known source chunk by construction, so
retrieval is somewhat easier than for arbitrary user queries. But this scales
to 1000+ questions in ~30 min and ~$1, vs ~100 hours of hand-curation."""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from anthropic import Anthropic
from tqdm import tqdm

from ..config import cfg
from ..store import get_collection


GEN_SYSTEM = """You generate realistic user questions for evaluating a code-RAG system over the Apache Airflow codebase.

You'll receive ONE chunk from the corpus (code, docs, or a GitHub issue/PR). Write ONE question that:
1. A real Airflow user or contributor would actually ask
2. This chunk would be useful for answering
3. Is phrased in the USER's voice — do NOT paraphrase the chunk
4. Is not so specific that it names the chunk's exact location ("what's in line 327 of xcom.py")
5. Is not so vague that any chunk would answer it ("what is Airflow")

Also classify the question:
- "factual"     — looking up a specific behavior, default, or value
- "where_is"    — looking for implementation, definition, or "how does X work"
- "conceptual"  — understanding a concept, comparison, or design rationale
- "multi_hop"   — answer requires walking multiple files (e.g., "what fires and in what order")

If the chunk is too small, too generic, or otherwise unsuitable for generating a meaningful question, reply with: {"skip": true, "reason": "..."}

Otherwise reply ONLY with JSON (no fences):
{"question": "...", "category": "factual" | "where_is" | "conceptual" | "multi_hop"}
"""


@dataclass
class GeneratedQuestion:
    id: str
    question: str
    category: str
    expected_paths: list[str]
    expected_symbols: list[str]
    source_chunk_id: str
    notes: str


def _client() -> Anthropic:
    return Anthropic(api_key=cfg.anthropic_api_key)


def _sample_chunk_metadatas(n_target: int, seed: int = 42) -> list[tuple[str, str, dict]]:
    """Random-sample `n_target` chunks suitable for question generation.

    Strategy: load all chunk IDs, sample 3x our target (to allow filter rejects),
    fetch their docs+metas, filter for quality."""
    coll = get_collection()
    all_ids = coll.get(include=[])["ids"]
    rng = random.Random(seed)
    # ~17% of random chunks survive the quality filter (most rejects are test files
    # and dev tooling). 8x oversample lets us hit n_target reliably.
    oversample_n = min(n_target * 8, len(all_ids))
    sampled = rng.sample(all_ids, oversample_n)

    items: list[tuple[str, str, dict]] = []
    # Chroma get() in batches of ~1000 to keep things sane
    BATCH = 500
    for i in range(0, len(sampled), BATCH):
        batch_ids = sampled[i:i + BATCH]
        res = coll.get(ids=batch_ids, include=["documents", "metadatas"])
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            # Quality filter: substantial content, real symbol, useful kinds
            if len(doc) < 200:
                continue
            sym = meta.get("symbol") or ""
            if not sym or sym == "<module>":
                continue
            if meta.get("kind") == "class_header":
                continue
            # Skip test files — users don't ask "how does this test work?"
            path = meta.get("source_path", "")
            if "/tests/" in path or path.startswith("tests/") or "/test_" in path:
                continue
            # Skip dev/breeze tooling — internal to Airflow's build, not user-facing
            if path.startswith("dev/") or path.startswith("devel-common/"):
                continue
            items.append((cid, doc, meta))
            if len(items) >= n_target:
                return items
    return items


def _parse_response(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(text)
        if data.get("skip"):
            return None
        if "question" not in data or "category" not in data:
            return None
        if data["category"] not in {"factual", "where_is", "conceptual", "multi_hop"}:
            return None
        return data
    except json.JSONDecodeError:
        return None


def generate_one(chunk_doc: str, chunk_meta: dict, client: Anthropic) -> dict | None:
    """Call Claude to write a question for this chunk. Returns None if unusable."""
    context = (
        f"path: {chunk_meta['source_path']}\n"
        f"kind: {chunk_meta['kind']}\n"
        f"symbol: {chunk_meta.get('symbol', '')}\n\n"
        f"{chunk_doc[:1500]}"
    )
    try:
        msg = client.messages.create(
            model=cfg.anthropic_model,
            max_tokens=200,
            system=GEN_SYSTEM,
            messages=[{"role": "user", "content": context}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return _parse_response(text)
    except Exception:
        return None


def generate(n: int, output: Path, seed: int = 42) -> int:
    """Generate `n` questions and write to `output` as YAML. Returns count written."""
    print(f"Sampling {n} chunks...")
    chunks = _sample_chunk_metadatas(n, seed=seed)
    print(f"Got {len(chunks)} candidate chunks after quality filter.")

    client = _client()
    generated: list[dict] = []

    for i, (cid, doc, meta) in enumerate(tqdm(chunks, desc="generating", unit="q")):
        result = generate_one(doc, meta, client)
        if result is None:
            continue
        # Use the chunk's directory or full file path as expected_paths.
        # GH chunks use synthetic _gh paths — those are fine to match exactly.
        path = meta["source_path"]
        sym = meta.get("symbol", "") or ""
        generated.append({
            "id": f"gen_{i:04d}",
            "question": result["question"],
            "category": result["category"],
            "expected_paths": [path],
            "expected_symbols": [sym] if sym and sym != "<module>" else [],
            "source_chunk_id": cid,
            "notes": f"auto-generated from chunk: {cid}",
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        yaml.safe_dump(generated, f, sort_keys=False, width=120)
    return len(generated)
