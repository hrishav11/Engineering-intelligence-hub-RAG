"""Compose retrieved context into a Claude prompt and return a cited answer.

Try-then-fallback pattern: when method='routed' and the question routes to
hybrid, we run hybrid first. If Claude emits the INSUFFICIENT_CONTEXT sentinel
(meaning: "the retrieved chunks don't actually answer this"), we transparently
escalate to agentic retrieval. Smarter than trying to guess upfront from
question shape — detects the actual failure."""
from __future__ import annotations

from .llm import call_with_fallback
from .store import query

INSUFFICIENT_SENTINEL = "<INSUFFICIENT_CONTEXT>"

SYSTEM = f"""You are an expert engineer answering questions about the Apache Airflow codebase.

Rules:
- Answer ONLY using the provided context snippets.
- Every factual claim must cite a source as [#N] where N is the snippet number.
- Be concise. Prefer concrete file paths and symbol names over generalities.
- If snippets conflict, note the conflict.

CRITICAL FALLBACK RULE:
- If the snippets do NOT contain enough specific information to actually answer the
  question (e.g., you see only a class header but the user asked how a method works,
  or all snippets are tests/usages and the implementation is missing), reply with
  EXACTLY this single token and nothing else:
  {INSUFFICIENT_SENTINEL}
- A downstream system will re-run the query with deeper multi-step retrieval when
  it sees this sentinel. So it's better to emit the sentinel than to write a vague
  "I don't have enough information" prose answer.
"""


def _format_context(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        header = f"[#{i}] {m['source_path']} (lines {m['start_line']}-{m['end_line']}, {m['kind']}"
        if m.get("symbol"):
            header += f", {m['symbol']}"
        header += ")"
        lines.append(f"{header}\n{h['text']}")
    return "\n\n---\n\n".join(lines)


def _is_insufficient(answer_text: str) -> bool:
    """Return True if Claude emitted the sentinel (allowing trivial whitespace/quoting)."""
    stripped = answer_text.strip().strip("`'\"")
    return stripped == INSUFFICIENT_SENTINEL or stripped.startswith(INSUFFICIENT_SENTINEL)


def _single_shot(question: str, k: int, method: str) -> dict:
    """Run one retrieve+answer cycle. Returns {answer, hits, insufficient: bool}."""
    hits = query(question, k=k, method=method)
    context = _format_context(hits)
    msg = call_with_fallback(
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}],
    )
    answer_text = "".join(block.text for block in msg.content if block.type == "text")
    return {
        "answer": answer_text,
        "hits": hits,
        "insufficient": _is_insufficient(answer_text),
    }


def ask(question: str, k: int = 6, method: str = "hybrid") -> dict:
    """Retrieve + answer. For `method='routed'`, transparently falls back from
    hybrid → agentic if the first attempt emits the INSUFFICIENT_CONTEXT sentinel."""
    used_fallback = False
    routed_first_method: str | None = None

    if method == "routed":
        from .router import route
        method = route(question)
        routed_first_method = method

    if method == "agentic":
        from . import agentic
        result = agentic.ask(question)
        result["used_fallback"] = used_fallback
        result["routed_to"] = "agentic" if routed_first_method else None
        return result

    # Single-shot retrieval attempt
    result = _single_shot(question, k=k, method=method)

    # If router chose a non-agentic method and the model says insufficient,
    # fall back to agentic. We only do this in the routed flow (don't override
    # the user's explicit method choice).
    if routed_first_method is not None and result["insufficient"]:
        from . import agentic
        fallback = agentic.ask(question)
        fallback["used_fallback"] = True
        fallback["routed_to"] = f"{routed_first_method} → agentic"
        # Preserve the original hits for transparency
        fallback["fallback_from_hits"] = result["hits"]
        return fallback

    result["used_fallback"] = used_fallback
    result["routed_to"] = routed_first_method
    return result
