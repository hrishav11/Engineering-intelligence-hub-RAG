"""LLM-as-judge: score a generated answer 0-3 against the question.

We don't pass a ground-truth answer; the judge reads the question and the
generated answer (with citations), then scores based on a fixed rubric. The
rubric explicitly rewards honest 'I don't know' over confident hallucinations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from anthropic import Anthropic

from ..config import cfg

JUDGE_SYSTEM = """You are an evaluator scoring answers from a code-RAG system over the Apache Airflow codebase.

You are given the user's question and the system's answer (with inline citations).
Score the answer 0-3 using THIS rubric exactly:

3 — Fully correct, specific, well-cited. Answers the question with concrete details
    that match what an Airflow expert would say. Names real symbols/files.

2 — Mostly correct but partial. Either misses an important aspect, or is correct
    but stays high-level when specifics were warranted.

1 — Vague, conceptually adjacent, or mixes correct claims with incorrect ones.

0 — Wrong, hallucinated, or refuses to answer when the question was reasonable.

IMPORTANT: An honest "I cannot find this in the context" answer should score 1, not 0 —
it's worse than answering but better than confidently hallucinating. Only score 0 if the
answer is actually misleading or fabricated.

Reply ONLY with JSON of the form:
{"score": <int 0-3>, "reason": "<one sentence>"}
"""


@dataclass
class Judgment:
    score: int    # 0-3
    reason: str


def _client() -> Anthropic:
    return Anthropic(api_key=cfg.anthropic_api_key)


def judge(question: str, answer: str) -> Judgment:
    client = _client()
    msg = client.messages.create(
        model=cfg.anthropic_model,
        max_tokens=256,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {question}\n\nAnswer:\n{answer}"}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    return _parse(text)


def _parse(text: str) -> Judgment:
    # Strip code fences if the model returned them.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(text)
        score = int(data["score"])
        reason = str(data.get("reason", ""))[:300]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # Fallback: try to find a single digit and salvage.
        m = re.search(r'"score"\s*:\s*([0-3])', text)
        score = int(m.group(1)) if m else 0
        reason = text[:300]
    return Judgment(score=max(0, min(3, score)), reason=reason)
