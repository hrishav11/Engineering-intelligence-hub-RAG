"""Week 2 ingestion: markdown/rst via token chunking + Python via tree-sitter (function/class bodies)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tiktoken

from .chunker import chunk_python_file

ENC = tiktoken.get_encoding("cl100k_base")

CHUNK_TOKENS = 500
CHUNK_OVERLAP = 75

DOC_EXTS = {".md", ".rst", ".txt"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"}


@dataclass
class Chunk:
    text: str
    source_path: str
    kind: str       # "doc" | "function" | "method" | "class_header" | "module_header"
    symbol: str | None
    start_line: int
    end_line: int

    def id(self) -> str:
        sym = self.symbol or "_"
        return f"{self.source_path}:{self.start_line}-{self.end_line}:{self.kind}:{sym}"


def _chunk_text(text: str, source_path: str, kind: str, symbol: str | None, base_line: int) -> Iterator[Chunk]:
    tokens = ENC.encode(text)
    if not tokens:
        return
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    for start in range(0, len(tokens), step):
        window = tokens[start:start + CHUNK_TOKENS]
        if not window:
            break
        chunk_text = ENC.decode(window)
        # Approximate line span from char ratio — good enough for W1
        line_offset_start = text[:len(ENC.decode(tokens[:start]))].count("\n")
        line_offset_end = line_offset_start + chunk_text.count("\n")
        yield Chunk(
            text=chunk_text,
            source_path=source_path,
            kind=kind,
            symbol=symbol,
            start_line=base_line + line_offset_start,
            end_line=base_line + line_offset_end,
        )
        if len(window) < CHUNK_TOKENS:
            break


def _extract_python(py_path: Path, rel_path: str) -> Iterator[Chunk]:
    for cc in chunk_python_file(py_path, rel_path):
        yield Chunk(
            text=cc.text,
            source_path=cc.source_path,
            kind=cc.kind,
            symbol=cc.symbol,
            start_line=cc.start_line,
            end_line=cc.end_line,
        )


def _extract_doc_file(path: Path, rel_path: str) -> Iterator[Chunk]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    if len(text.strip()) < 50:
        return
    yield from _chunk_text(text, rel_path, "doc", None, 1)


def walk_repo(repo_root: Path) -> Iterator[Chunk]:
    repo_root = repo_root.resolve()
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = str(path.relative_to(repo_root))
        if path.suffix.lower() in DOC_EXTS:
            yield from _extract_doc_file(path, rel)
        elif path.suffix == ".py":
            yield from _extract_python(path, rel)
