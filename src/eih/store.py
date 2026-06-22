"""Chroma vector store with OpenAI embeddings."""
from __future__ import annotations

import time
from typing import Iterable

import chromadb
import tiktoken
from chromadb.config import Settings
from openai import OpenAI

from . import bm25 as bm25mod
from .config import cfg
from .ingest import Chunk

RRF_K = 60

ENC = tiktoken.get_encoding("cl100k_base")
BATCH_TOKEN_BUDGET = 20_000  # 20K tokens per request
MIN_INTERVAL_SEC = 35.0      # 20K / 35s ≈ 34K TPM, under the 40K limit


def _client() -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(path=cfg.chroma_dir, settings=Settings(anonymized_telemetry=False))


def get_collection():
    return _client().get_or_create_collection(name=cfg.collection_name, metadata={"hnsw:space": "cosine"})


def reset_collection() -> None:
    client = _client()
    try:
        client.delete_collection(cfg.collection_name)
    except Exception:
        pass
    client.create_collection(name=cfg.collection_name, metadata={"hnsw:space": "cosine"})


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = OpenAI(api_key=cfg.openai_api_key, max_retries=10)
    out: list[list[float]] = []
    batch: list[str] = []
    batch_tokens = 0
    last_call = 0.0
    for text in texts:
        n = len(ENC.encode(text))
        if batch and batch_tokens + n > BATCH_TOKEN_BUDGET:
            wait = MIN_INTERVAL_SEC - (time.monotonic() - last_call)
            if wait > 0:
                time.sleep(wait)
            resp = client.embeddings.create(model=cfg.openai_embed_model, input=batch)
            out.extend(d.embedding for d in resp.data)
            last_call = time.monotonic()
            batch, batch_tokens = [], 0
        batch.append(text)
        batch_tokens += n
    if batch:
        wait = MIN_INTERVAL_SEC - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
        resp = client.embeddings.create(model=cfg.openai_embed_model, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


def add_chunks(chunks: Iterable[Chunk], batch_size: int = 256, build_bm25: bool = True) -> int:
    coll = get_collection()
    seen: set[str] = set()
    batch: list[Chunk] = []
    total = 0
    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metas: list[dict] = []
    for chunk in chunks:
        cid = chunk.id()
        if cid in seen:
            continue
        seen.add(cid)
        batch.append(chunk)
        all_ids.append(cid)
        all_texts.append(chunk.text)
        all_metas.append({"source_path": chunk.source_path, "symbol": chunk.symbol or ""})
        if len(batch) >= batch_size:
            total += _flush(coll, batch)
            batch = []
    if batch:
        total += _flush(coll, batch)
    if all_ids and build_bm25:
        bm25mod.build(all_ids, all_texts, all_metas)
    return total


def rebuild_bm25_from_collection() -> int:
    """Rebuild the BM25 index from chunks already stored in Chroma. No re-embedding."""
    coll = get_collection()
    total = coll.count()
    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []
    page = 5000
    offset = 0
    while offset < total:
        res = coll.get(limit=page, offset=offset)
        ids.extend(res["ids"])
        texts.extend(res["documents"])
        metas.extend(res["metadatas"])
        offset += page
    bm25mod.build(ids, texts, metas)
    return len(ids)


def _flush(coll, batch: list[Chunk]) -> int:
    embeddings = embed_texts([c.text for c in batch])
    coll.add(
        ids=[c.id() for c in batch],
        documents=[c.text for c in batch],
        embeddings=embeddings,
        metadatas=[
            {
                "source_path": c.source_path,
                "kind": c.kind,
                "symbol": c.symbol or "",
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in batch
        ],
    )
    return len(batch)


def _vector_query(text: str, k: int, embed_text: str | None = None) -> list[tuple[str, dict, str, float]]:
    """Returns [(id, meta, doc, distance), ...] from Chroma.
    If `embed_text` is given, embed THAT (e.g. a HyDE hypothetical answer) instead
    of `text`. BM25 still uses the original `text` — HyDE only affects vectors."""
    coll = get_collection()
    emb = embed_texts([embed_text if embed_text is not None else text])[0]
    res = coll.query(query_embeddings=[emb], n_results=k)
    return list(zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]))


def query(text: str, k: int = 6, method: str = "hybrid", rerank: bool = False, hyde: bool = False) -> list[dict]:
    """method: 'vector' | 'bm25' | 'hybrid' | 'hybrid_rerank' | 'hybrid_hyde' |
    'hybrid_rerank_hyde' | 'routed' (router picks hybrid vs hybrid_hyde per question).
    rerank / hyde flags are additive — method='hybrid_rerank_hyde' sets both."""
    if method == "hybrid_rerank":
        rerank = True
        method = "hybrid"
    elif method == "hybrid_hyde":
        hyde = True
        method = "hybrid"
    elif method == "hybrid_rerank_hyde":
        rerank = True
        hyde = True
        method = "hybrid"

    embed_text: str | None = None
    if hyde and method in {"hybrid", "vector"}:
        from . import hyde as hyde_mod
        embed_text = hyde_mod.hyde_query_text(text)

    if method == "vector":
        vec_hits = _vector_query(text, k, embed_text=embed_text)
        return [{"text": doc, "meta": meta, "score": -dist,
                 "source": "vector+hyde" if hyde else "vector"}
                for _id, meta, doc, dist in vec_hits]

    if method == "bm25":
        idx = bm25mod.load()
        if idx is None:
            return _empty_if_no_index()
        top = idx.query(text, k)
        coll = get_collection()
        res = coll.get(ids=[i for i, _ in top])
        by_id = {i: (m, d) for i, m, d in zip(res["ids"], res["metadatas"], res["documents"])}
        out = []
        for chunk_id, score in top:
            if chunk_id not in by_id:
                continue
            meta, doc = by_id[chunk_id]
            out.append({"text": doc, "meta": meta, "score": score, "source": "bm25"})
        return out

    # hybrid: pull a wide net from each side, then fuse with reciprocal rank fusion.
    # When rerank=True, pull an even wider net and let the cross-encoder choose top-k.
    inner_k = max(k * 5, 30) if rerank else k
    fetch_k = max(inner_k * 10, 60)
    vec_hits = _vector_query(text, fetch_k, embed_text=embed_text)
    idx = bm25mod.load()
    bm25_hits = idx.query(text, fetch_k) if idx is not None else []

    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    vec_tag = "vector+hyde" if hyde else "vector"
    for rank, (chunk_id, _meta, _doc, _dist) in enumerate(vec_hits):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        sources.setdefault(chunk_id, set()).add(vec_tag)
    for rank, (chunk_id, _score) in enumerate(bm25_hits):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        sources.setdefault(chunk_id, set()).add("bm25")

    if not scores:
        return []

    # Symbol injection: if the query names an identifier (e.g. `GCSHook.upload`),
    # pin chunks whose symbol matches it to the top of the hybrid result regardless
    # of RRF score. This rescues exact-name lookups that BM25's keyword density and
    # vector's semantic similarity both fail to surface.
    # Symbol-pinning is a narrow rescue, not a flood. Cap at 1/3 of the budget
    # (max 3) — otherwise generic identifiers like `TaskInstance` that match many
    # class_header chunks will eat the entire result and crowd out everything else.
    pinned: list[str] = []
    pin_budget = max(1, min(3, k // 3))
    identifiers = bm25mod.extract_identifiers(text)
    if identifiers and idx is not None:
        injected = idx._injection_indices(identifiers)
        # If the identifier matches too many chunks (>5), it's generic — don't pin.
        if len(injected) <= 5:
            for i in injected:
                cid = idx.chunk_ids[i]
                pinned.append(cid)
                sources.setdefault(cid, set()).add("symbol")
                if len(pinned) >= pin_budget:
                    break

    rrf_top = sorted(scores, key=lambda i: -scores[i])
    seen = set(pinned)
    top_ids = list(pinned)
    for cid in rrf_top:
        if len(top_ids) >= inner_k:
            break
        if cid in seen:
            continue
        seen.add(cid)
        top_ids.append(cid)
    coll = get_collection()
    res = coll.get(ids=top_ids)
    by_id = {i: (m, d) for i, m, d in zip(res["ids"], res["metadatas"], res["documents"])}
    out = []
    for chunk_id in top_ids:
        if chunk_id not in by_id:
            continue
        meta, doc = by_id[chunk_id]
        out.append({
            "text": doc,
            "meta": meta,
            "score": scores[chunk_id],
            "source": "+".join(sorted(sources[chunk_id])),
        })

    if rerank:
        # Keep symbol-pinned chunks at the top (they're our high-confidence
        # exact-name matches), let the reranker re-order the rest. bge-reranker
        # is good at "is this generally relevant?" but tends to prefer
        # example/test code over canonical definitions for code-heavy queries.
        from . import rerank as rerank_mod
        out = rerank_mod.rerank(text, out, top_k=k, pin_count=len(pinned))

    return out[:k]


def _empty_if_no_index() -> list[dict]:
    return []
