"""Retrieval metrics for the eval harness.

Two kinds of measurement:

1. **Rank-position metrics** (hit@k, MRR): "is the right chunk at the top?"
   Right for single-shot retrieval. WRONG for agentic, which accumulates 40+
   chunks across multiple search calls — the canonical answer often surfaces
   in search #3, not in the first k of search #1.

2. **Coverage metrics** (path_coverage, symbol_coverage): "did the system find
   chunks from ALL the expected files / matching ALL the expected symbols,
   anywhere in the cumulative result?" This is what multi-hop and agentic
   need, and it correctly rewards a system that walks the codebase even when
   no single search ranks the right chunk first."""
from __future__ import annotations

from dataclasses import dataclass


def _path_hit(meta_path: str, expected_paths: list[str]) -> bool:
    return any(p in meta_path for p in expected_paths)


def _symbol_hit(meta_symbol: str, expected_symbols: list[str]) -> bool:
    if not expected_symbols:
        return True
    if not meta_symbol:
        return False
    return any(s in meta_symbol for s in expected_symbols)


def hit_ranks(
    hits: list[dict],
    expected_paths: list[str],
    expected_symbols: list[str] | None = None,
) -> list[int]:
    """Return the 0-indexed ranks (within hits) of chunks matching path AND symbol."""
    ranks: list[int] = []
    for rank, h in enumerate(hits):
        meta = h["meta"]
        if _path_hit(meta.get("source_path", ""), expected_paths) and \
           _symbol_hit(meta.get("symbol", ""), expected_symbols or []):
            ranks.append(rank)
    return ranks


def path_hit_ranks(
    hits: list[dict],
    expected_paths: list[str],
) -> list[int]:
    """Path-only ranks. Looser than `hit_ranks` — counts a chunk as a hit when
    its file matches, even if the specific symbol doesn't. This is what users
    actually experience: 'did the system point me to the right file?'"""
    return [rank for rank, h in enumerate(hits)
            if _path_hit(h["meta"].get("source_path", ""), expected_paths)]


def _coverage(hits: list[dict], expected: list[str], key: str) -> float:
    """Fraction of expected substrings that match at least one chunk's metadata field."""
    if not expected:
        return 1.0
    matched = 0
    for needle in expected:
        for h in hits:
            value = h["meta"].get(key, "") or ""
            if needle in value:
                matched += 1
                break
    return matched / len(expected)


@dataclass
class RetrievalScore:
    # Strict rank-position metrics (path AND symbol must match)
    hit_at_1: int             # 0 or 1
    hit_at_3: int
    hit_at_10: int
    mrr: float                # 0.0 if no hit, else 1/(rank+1) of first hit
    num_hits_in_topk: int     # how many returned chunks matched

    # Path-only rank-position metrics — what users actually experience
    p_hit_at_1: int
    p_hit_at_3: int
    p_hit_at_10: int
    p_mrr: float

    # Coverage metrics (anywhere in cumulative hits, agentic-friendly)
    path_coverage: float      # fraction of expected_paths matched anywhere in hits
    symbol_coverage: float    # fraction of expected_symbols matched anywhere in hits


def score(
    hits: list[dict],
    expected_paths: list[str],
    expected_symbols: list[str] | None = None,
) -> RetrievalScore:
    ranks = hit_ranks(hits, expected_paths, expected_symbols)
    first = ranks[0] if ranks else None
    p_ranks = path_hit_ranks(hits, expected_paths)
    p_first = p_ranks[0] if p_ranks else None
    return RetrievalScore(
        hit_at_1=int(first is not None and first < 1),
        hit_at_3=int(first is not None and first < 3),
        hit_at_10=int(first is not None and first < 10),
        mrr=(1.0 / (first + 1)) if first is not None else 0.0,
        num_hits_in_topk=len(ranks),
        p_hit_at_1=int(p_first is not None and p_first < 1),
        p_hit_at_3=int(p_first is not None and p_first < 3),
        p_hit_at_10=int(p_first is not None and p_first < 10),
        p_mrr=(1.0 / (p_first + 1)) if p_first is not None else 0.0,
        path_coverage=_coverage(hits, expected_paths, "source_path"),
        symbol_coverage=_coverage(hits, expected_symbols or [], "symbol"),
    )
