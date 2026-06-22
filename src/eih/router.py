"""Question router. Picks the right retrieval method based on question shape.

Derived from the Week 4 eval data: HyDE dominates conceptual questions (hit@3
went from 0.67 → 1.00) but hurts code lookups. Plain hybrid wins everything
that names a specific identifier or asks "where/how is X implemented." So the
rule is simple: conceptual → hyde, otherwise → hybrid.

We use cheap regex heuristics (no API call). If a future LLM classifier proves
worth the latency, swap it in by replacing `classify_question`."""
from __future__ import annotations

import re
from typing import Literal

from . import bm25 as bm25mod

QuestionType = Literal["conceptual", "code", "multi_hop"]

# Phrases that strongly indicate a conceptual / definitional question.
# Order matters — more specific patterns first.
_CONCEPTUAL_PATTERNS = [
    re.compile(r"\bdifference between\b", re.I),
    re.compile(r"\bcompare\b.*\band\b", re.I),
    re.compile(r"\bvs\.?\b", re.I),
    re.compile(r"^\s*what\s+(?:is|are)\s+(?:a\s+|an\s+|the\s+)?", re.I),
    re.compile(r"^\s*what'?s\s+(?:a\s+|an\s+|the\s+)?", re.I),
    re.compile(r"\bwhen\s+should\s+(?:i|you|we)\b", re.I),
    re.compile(r"\bwhy\s+(?:does|do|is|are)\b", re.I),
    re.compile(r"\bexplain\s+(?:the|how)\b", re.I),
]

# Multi-step questions where agentic retrieval pays off (the model walks the
# graph, refining queries based on what each step reveals).
_MULTIHOP_PATTERNS = [
    re.compile(r"\bin\s+what\s+order\b", re.I),
    re.compile(r"\bfull\s+path\s+from\b", re.I),
    re.compile(r"\bend[-\s]?to[-\s]?end\b", re.I),
    re.compile(r"\binteract(?:ion|s)?\s+(?:between|with)\b", re.I),
    re.compile(r"\brespect(?:s)?\s+both\b", re.I),
    re.compile(r"\blifecycle\s+of\b", re.I),
    re.compile(r"\bstep[\s-]by[\s-]step\b", re.I),
    re.compile(r"\bwhich\s+\w+\s+(?:fire|fires|run|runs)\s+and\b", re.I),
    re.compile(r"\bflow\s+(?:from|of)\b", re.I),
    re.compile(r"\bchain\s+of\b", re.I),
]

# Phrases that strongly indicate a code/implementation question.
_CODE_PATTERNS = [
    re.compile(r"\bwhere\s+is\b.*\b(?:defined|implemented|located)\b", re.I),
    re.compile(r"\bhow\s+(?:does|do)\s+\w+\.\w+", re.I),       # "How does Foo.bar..."
    re.compile(r"\bimplementation\s+of\b", re.I),
    re.compile(r"\bsource\s+code\b", re.I),
]


def classify_question(question: str) -> QuestionType:
    """Heuristic classifier. Order matters:
    1. Multi-hop signals win first — rare but signal expensive agentic retrieval.
    2. Explicit `Class.method` or code pattern → code.
    3. Conceptual phrasing → conceptual.
    4. Default → code (safest single-shot baseline).

    Vague questions that hybrid will fail on are NOT escalated here — instead
    `answer.ask()` runs hybrid first, detects insufficient-context via a
    sentinel, and falls back to agentic. Smarter than guessing upfront."""
    if any(p.search(question) for p in _MULTIHOP_PATTERNS):
        return "multi_hop"

    if any("." in i for i in bm25mod.extract_identifiers(question)):
        return "code"

    if any(p.search(question) for p in _CODE_PATTERNS):
        return "code"

    if any(p.search(question) for p in _CONCEPTUAL_PATTERNS):
        return "conceptual"

    return "code"


_ROUTING_TABLE = {
    "multi_hop": "agentic",
    "conceptual": "hybrid_hyde",
    "code": "hybrid",
}


def route(question: str) -> str:
    """Return the retrieval method to use for this question."""
    return _ROUTING_TABLE[classify_question(question)]
