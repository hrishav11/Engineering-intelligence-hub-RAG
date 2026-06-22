"""Voyage AI cross-encoder reranker — trained on code, unlike bge-reranker-base.

The hypothesis: a code-trained reranker won't make the same "I can't tell the
implementation from a test that calls it" mistakes bge made on our corpus.

Usage mirrors rerank.py: same signature, same `pin_count` semantics, so
store.query can swap between rerankers via method name."""
from __future__ import annotations

from functools import lru_cache

from .config import cfg

MODEL_NAME = cfg.voyage_rerank_model  # rerank-2-lite by default — cheap + good


@lru_cache(maxsize=1)
def _client():
    import voyageai
    if not cfg.voyage_api_key:
        raise RuntimeError(
            "VOYAGE_API_KEY not set. Add it to .env to use hybrid_voyage_rerank."
        )
    return voyageai.Client(api_key=cfg.voyage_api_key)


def rerank(query: str, hits: list[dict], top_k: int, pin_count: int = 0) -> list[dict]:
    """Rerank `hits` against `query`, keeping the first `pin_count` chunks pinned."""
    if not hits:
        return hits
    pinned = hits[:pin_count]
    rest = hits[pin_count:]
    if not rest:
        return pinned

    docs = [h["text"][:4000] for h in rest]  # Voyage caps at 32K tokens per chunk
    resp = _client().rerank(
        query=query,
        documents=docs,
        model=MODEL_NAME,
        top_k=len(rest),
    )
    # resp.results is sorted by relevance_score desc, each has .index, .relevance_score
    rest_sorted = []
    for r in resp.results:
        h = rest[r.index]
        h["rerank_score"] = float(r.relevance_score)
        if "source" in h and "voyage" not in h["source"]:
            h["source"] = h["source"] + "+voyage"
        rest_sorted.append(h)

    return (pinned + rest_sorted)[:top_k]
