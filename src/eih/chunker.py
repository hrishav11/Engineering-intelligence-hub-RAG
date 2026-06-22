"""Tree-sitter Python chunker. Emits one chunk per top-level function, class header,
and class method — each carrying its full body, not just the docstring."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tiktoken
import tree_sitter_python
from tree_sitter import Language, Node, Parser

ENC = tiktoken.get_encoding("cl100k_base")
PY_LANG = Language(tree_sitter_python.language())
PARSER = Parser(PY_LANG)

MAX_CHUNK_TOKENS = 1500
MIN_CHUNK_TOKENS = 20


@dataclass
class CodeChunk:
    text: str
    source_path: str
    kind: str         # "function" | "method" | "class_header" | "module_header"
    symbol: str       # "MyClass.my_method" / "my_function" / "<module>"
    start_line: int
    end_line: int


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_name(node: Node) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    return name_node.text.decode("utf-8", errors="replace")


def _split_oversized(text: str) -> list[str]:
    tokens = ENC.encode(text)
    if len(tokens) <= MAX_CHUNK_TOKENS:
        return [text]
    out = []
    step = MAX_CHUNK_TOKENS - 100
    for i in range(0, len(tokens), step):
        window = tokens[i:i + MAX_CHUNK_TOKENS]
        out.append(ENC.decode(window))
    return out


def _module_header(root: Node, source: bytes) -> str | None:
    """Module-level imports + module docstring."""
    parts: list[str] = []
    for child in root.children:
        if child.type in {"import_statement", "import_from_statement", "future_import_statement"}:
            parts.append(_node_text(child, source))
        elif child.type == "expression_statement":
            inner = child.children[0] if child.children else None
            if inner is not None and inner.type == "string":
                parts.append(_node_text(inner, source))
                break
        elif child.type in {"function_definition", "class_definition", "decorated_definition"}:
            break
    return "\n".join(parts).strip() or None


def _class_header(class_node: Node, source: bytes) -> str:
    """Class signature + docstring + field assignments (no method bodies)."""
    name = _get_name(class_node) or "<class>"
    bases = class_node.child_by_field_name("superclasses")
    sig = f"class {name}{_node_text(bases, source) if bases else ''}:"
    parts = [sig]
    body = class_node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            if child.type == "expression_statement":
                inner = child.children[0] if child.children else None
                if inner is not None and inner.type == "string":
                    parts.append("    " + _node_text(inner, source).replace("\n", "\n    "))
            elif child.type == "assignment":
                parts.append("    " + _node_text(child, source))
    return "\n".join(parts)


def _unwrap_decorated(node: Node) -> Node:
    """A decorated_definition wraps the real function/class. Return the inner node."""
    if node.type != "decorated_definition":
        return node
    for child in node.children:
        if child.type in {"function_definition", "class_definition"}:
            return child
    return node


def _walk_methods(class_node: Node, source: bytes, rel_path: str, class_name: str) -> Iterator[CodeChunk]:
    body = class_node.child_by_field_name("body")
    if body is None:
        return
    for child in body.children:
        target = _unwrap_decorated(child) if child.type == "decorated_definition" else child
        if target.type != "function_definition":
            continue
        method_name = _get_name(target) or "<method>"
        text = _node_text(child, source)
        for piece in _split_oversized(text):
            if len(ENC.encode(piece)) < MIN_CHUNK_TOKENS:
                continue
            yield CodeChunk(
                text=piece,
                source_path=rel_path,
                kind="method",
                symbol=f"{class_name}.{method_name}",
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
            )


def chunk_python_file(path: Path, rel_path: str) -> Iterator[CodeChunk]:
    try:
        source = path.read_bytes()
    except OSError:
        return
    if not source.strip():
        return
    try:
        tree = PARSER.parse(source)
    except Exception:
        return
    root = tree.root_node

    header = _module_header(root, source)
    if header and len(ENC.encode(header)) >= MIN_CHUNK_TOKENS:
        yield CodeChunk(
            text=header,
            source_path=rel_path,
            kind="module_header",
            symbol="<module>",
            start_line=1,
            end_line=header.count("\n") + 1,
        )

    for child in root.children:
        target = _unwrap_decorated(child)
        if target.type == "function_definition":
            name = _get_name(target) or "<fn>"
            text = _node_text(child, source)
            for piece in _split_oversized(text):
                if len(ENC.encode(piece)) < MIN_CHUNK_TOKENS:
                    continue
                yield CodeChunk(
                    text=piece,
                    source_path=rel_path,
                    kind="function",
                    symbol=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                )
        elif target.type == "class_definition":
            name = _get_name(target) or "<class>"
            header_text = _class_header(target, source)
            if len(ENC.encode(header_text)) >= MIN_CHUNK_TOKENS:
                yield CodeChunk(
                    text=header_text,
                    source_path=rel_path,
                    kind="class_header",
                    symbol=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                )
            yield from _walk_methods(target, source, rel_path, name)
