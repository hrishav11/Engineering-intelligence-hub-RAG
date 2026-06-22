"""BM25 keyword index with identifier-aware tokenization. Persists to a pickle.

We index two views of each chunk:
1. Free-text tokens (BM25Okapi) — enriched with file path + symbol repeated
2. A symbol map (symbol → chunk_ids) for exact-identifier injection at query time

At query time, if the user mentions an identifier like `GCSHook.upload`, we inject
the matching chunks at the top regardless of body keyword density."""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from .config import cfg

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# Identifier-like patterns in a natural-language query: Class.method, CamelCase, snake_case.
_QUERY_IDENT_RE = re.compile(
    r"\b("
    r"[A-Z][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"   # Foo.bar or Foo.bar.baz
    r"|[A-Z][a-z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*)+"      # CamelCase (≥ 2 caps)
    r"|[a-z_][a-z0-9_]*_[a-z0-9_]+"                       # snake_case (≥ 1 underscore)
    r")\b"
)

_STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "for", "with", "by", "from", "as", "at", "this", "that",
    "it", "its", "if", "not", "no", "do", "does", "did", "has", "have", "had",
    "self", "cls", "none", "true", "false", "return", "def", "class",
})


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, and also split snake_case / CamelCase parts."""
    out: list[str] = []
    for word in _WORD_RE.findall(text):
        lower = word.lower()
        if lower not in _STOP and len(lower) > 1:
            out.append(lower)
        parts = _CAMEL_RE.findall(word)
        if len(parts) > 1:
            for p in parts:
                pl = p.lower()
                if pl != lower and pl not in _STOP and len(pl) > 1:
                    out.append(pl)
    return out


def extract_identifiers(query: str) -> list[str]:
    """Find identifier-like substrings in a natural-language query."""
    return _QUERY_IDENT_RE.findall(query)


@dataclass
class BM25Index:
    chunk_ids: list[str]
    bm25: BM25Okapi
    # Maps lowercase symbol → indices into chunk_ids. Includes both the full
    # symbol ("GCSHook.upload") and just the method name ("upload") so a query
    # mentioning either can hit.
    symbol_map: dict[str, list[int]] = field(default_factory=dict)

    def _injection_indices(self, identifiers: list[str]) -> list[int]:
        """Return chunk indices whose symbol matches any of the given identifiers."""
        seen: set[int] = set()
        out: list[int] = []
        for ident in identifiers:
            key = ident.lower()
            for i in self.symbol_map.get(key, []):
                if i not in seen:
                    seen.add(i)
                    out.append(i)
        return out

    def query(self, text: str, k: int) -> list[tuple[str, float]]:
        # Boost identifiers by repeating them in the query token stream
        identifiers = extract_identifiers(text)
        boosted_text = text + " " + " ".join(identifiers * 5)
        tokens = tokenize(boosted_text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])

        # Symbol injection: chunks whose symbol exactly matches a query identifier
        # get pinned to the front. Order: dotted matches first (more specific),
        # then bare-name matches.
        max_score = float(scores[ranked[0]]) if ranked and scores[ranked[0]] > 0 else 1.0
        inject = self._injection_indices(identifiers)
        injected_set = set(inject)

        out: list[tuple[str, float]] = []
        # Pinned identifier matches at the top
        for i in inject:
            out.append((self.chunk_ids[i], max_score + 100.0))
            if len(out) >= k:
                return out
        # Fill remainder from BM25 ranking, skipping already-injected
        for i in ranked:
            if scores[i] <= 0:
                break
            if i in injected_set:
                continue
            out.append((self.chunk_ids[i], float(scores[i])))
            if len(out) >= k:
                break
        return out


def _path() -> Path:
    p = Path(cfg.chroma_dir) / f"{cfg.collection_name}_bm25.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _enrich(text: str, source_path: str | None, symbol: str | None) -> str:
    """Prepend path + symbol so BM25 can match them. Symbol is repeated to boost weight."""
    parts: list[str] = []
    if source_path:
        parts.append(source_path)
    if symbol:
        parts.append(f"{symbol} {symbol} {symbol}")
    if parts:
        return " ".join(parts) + "\n\n" + text
    return text


def _build_symbol_map(chunk_ids: list[str], metas: list[dict] | None) -> dict[str, list[int]]:
    """Map each symbol (full and bare-name form) to the chunk indices it appears in."""
    out: dict[str, list[int]] = {}
    if metas is None:
        return out
    for i, m in enumerate(metas):
        sym = (m.get("symbol") or "").strip()
        if not sym or sym == "<module>":
            continue
        keys = {sym.lower()}
        if "." in sym:
            # "Foo.bar" → also map "bar" so a query mentioning just the method name hits
            keys.add(sym.rsplit(".", 1)[-1].lower())
        for key in keys:
            out.setdefault(key, []).append(i)
    return out


def build(chunk_ids: list[str], texts: list[str], metas: list[dict] | None = None) -> BM25Index:
    if metas is None:
        enriched = texts
    else:
        enriched = [_enrich(t, m.get("source_path"), m.get("symbol")) for t, m in zip(texts, metas)]
    tokenized = [tokenize(t) for t in enriched]
    bm25 = BM25Okapi(tokenized)
    symbol_map = _build_symbol_map(chunk_ids, metas)
    idx = BM25Index(chunk_ids=chunk_ids, bm25=bm25, symbol_map=symbol_map)
    with _path().open("wb") as f:
        pickle.dump(idx, f)
    return idx


def load() -> BM25Index | None:
    p = _path()
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)
