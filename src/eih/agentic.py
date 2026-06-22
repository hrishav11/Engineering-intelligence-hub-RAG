"""Agentic retrieval: give Claude a `search` tool it can call multiple times.

For each question, Claude decides what to search for, sees the results, then
decides whether to search again (with a refined or follow-up query). Stops when
it writes a final answer or hits the turn budget.

Why this matters: multi-hop questions ("when X fails, which callbacks fire and
in what order?") need information from multiple files. Single-shot retrieval
ranks by overall relevance to the question; agentic retrieval lets the model
walk the graph one hop at a time."""
from __future__ import annotations

from anthropic import Anthropic

from .config import cfg
from .store import query as store_query

SYSTEM = """You are an engineering assistant answering questions about the Apache Airflow codebase.

You have a `search` tool that returns relevant chunks from the codebase (code, docs, GitHub issues/PRs). Use it strategically:

- For simple lookups, one search is enough.
- For multi-step questions ("X happens, then what fires? in what order?"), do MULTIPLE searches — one per step. After seeing the first result, search for the NEXT thing it points to.
- Search by specific symbol names when you know them (e.g. `handle_failure`, `BaseXCom.serialize_value`) — these are exact-match friendly.
- Search with natural language for concepts.
- Don't repeat the same query.

When you have enough information, write the final answer with [#N] citations matching the chunk numbers from your search results. Be specific — name actual files, classes, and methods. If you genuinely can't find something after searching, say so plainly.

Budget: up to 5 searches per question. Use them wisely."""

TOOLS = [
    {
        "name": "search",
        "description": "Search the Airflow corpus (code + docs + GitHub issues/PRs) and return relevant chunks. Returns a numbered list with file path, line range, kind, symbol, and a text preview. Call multiple times with different queries for multi-step questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — natural language or specific identifier (e.g. 'GCSHook.upload', 'how scheduler picks DAGs').",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of chunks to return (1-10). Default 5.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    }
]


def _format_chunks(hits: list[dict], offset: int) -> str:
    """Compact chunk listing for the model — path + preview, no full body."""
    if not hits:
        return "No results."
    lines: list[str] = []
    for i, h in enumerate(hits, start=offset + 1):
        m = h["meta"]
        sym = m.get("symbol", "")
        preview = h["text"][:400].replace("\n", " ")
        lines.append(
            f"[{i}] {m['source_path']}:{m['start_line']}-{m['end_line']}"
            f" ({m['kind']}{(' · ' + sym) if sym else ''})\n    {preview}"
        )
    return "\n\n".join(lines)


def ask(question: str, max_turns: int = 5, search_k: int = 5) -> dict:
    """Agentic loop: Claude searches → reads → searches again → answers."""
    client = Anthropic(api_key=cfg.anthropic_api_key)
    messages: list[dict] = [{"role": "user", "content": question}]
    all_hits: list[dict] = []  # cumulative list across all tool calls
    trace: list[str] = []      # human-readable trace of search queries

    for turn in range(max_turns):
        msg = client.messages.create(
            model=cfg.anthropic_model,
            max_tokens=2048,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        if msg.stop_reason == "end_turn":
            answer_text = "".join(b.text for b in msg.content if b.type == "text")
            return {
                "answer": answer_text,
                "hits": all_hits,
                "turns": turn + 1,
                "trace": trace,
            }

        if msg.stop_reason != "tool_use":
            # Unexpected — bail with whatever text we have
            answer_text = "".join(b.text for b in msg.content if b.type == "text")
            return {"answer": answer_text or "(no answer)", "hits": all_hits, "turns": turn + 1, "trace": trace}

        # Echo assistant turn (required for tool result follow-up)
        messages.append({"role": "assistant", "content": msg.content})

        tool_results = []
        for block in msg.content:
            if block.type != "tool_use":
                continue
            q = block.input.get("query", "").strip()
            k = int(block.input.get("k", search_k))
            trace.append(f"turn {turn + 1}: search({q!r}, k={k})")
            hits = store_query(q, k=k, method="routed")
            offset = len(all_hits)
            all_hits.extend(hits)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _format_chunks(hits, offset),
            })
        messages.append({"role": "user", "content": tool_results})

    # Out of turn budget — force a final synthesis
    final = client.messages.create(
        model=cfg.anthropic_model,
        max_tokens=2048,
        system=SYSTEM + "\n\nYou are out of search budget. Write the best final answer you can with the chunks you've already seen.",
        messages=messages,
    )
    answer_text = "".join(b.text for b in final.content if b.type == "text")
    return {"answer": answer_text, "hits": all_hits, "turns": max_turns, "trace": trace}
