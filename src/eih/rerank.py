"""Cross-encoder reranker. Reads (query, chunk) pairs and assigns a true
relevance score — slower than BM25/vector but qualitatively different signal."""
from __future__ import annotations

from functools import lru_cache

# Module: bge-reranker-base is ~440MB, lazily downloaded on first use into
# ~/.cache/huggingface/. After that it's cached locally.
MODEL_NAME = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _model():
    import torch
    from sentence_transformers import CrossEncoder
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return CrossEncoder(MODEL_NAME, device=device, trust_remote_code=False)


def rerank(query: str, hits: list[dict], top_k: int, pin_count: int = 0) -> list[dict]:
    """Rerank `hits` against `query`, keeping the first `pin_count` chunks pinned
    in place (used for symbol-injected matches that shouldn't be re-ordered)."""
    if not hits:
        return hits
    pinned = hits[:pin_count]
    rest = hits[pin_count:]
    if not rest:
        return pinned

    model = _model()
    pairs = [(query, h["text"]) for h in rest]
    scores = model.predict(pairs)
    for h, s in zip(rest, scores):
        h["rerank_score"] = float(s)
    rest_sorted = sorted(rest, key=lambda h: -h["rerank_score"])

    # Tag source field so the CLI prints `+rerank` for reranked chunks
    for h in rest_sorted:
        if "source" in h and "rerank" not in h["source"]:
            h["source"] = h["source"] + "+rerank"

    return (pinned + rest_sorted)[:top_k]
