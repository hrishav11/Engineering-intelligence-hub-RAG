"""HyDE — Hypothetical Document Embeddings.

Ask Claude to write a plausible answer to the question, then embed THAT for
vector retrieval. The hypothetical answer often shares more wording with actual
answer chunks than the question does — especially for conceptual questions where
the user phrases things differently than the docs/code.

The hypothetical doesn't need to be correct. It just needs to look like the kind
of text that would answer the question. Wrong factual claims still help retrieval
because the surrounding language anchors the embedding."""
from __future__ import annotations

from anthropic import Anthropic

from .config import cfg

SYSTEM = """You write plausible hypothetical answers to questions about Apache Airflow.

Your goal is NOT to be correct — it's to look like the kind of paragraph that would
answer the question. Include likely Airflow identifiers (class names, methods,
config keys), file paths, and jargon. 3-5 sentences. No preamble, no caveats,
no "I believe" or "It may be" — just write the answer as if you knew."""


def hypothetical_answer(question: str) -> str:
    client = Anthropic(api_key=cfg.anthropic_api_key)
    msg = client.messages.create(
        model=cfg.anthropic_model,
        max_tokens=300,
        system=SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def hyde_query_text(question: str) -> str:
    """Return text suitable for embedding — hypothetical answer concatenated with
    the original question. Concatenation gives the embedding both the imagined
    answer's vocabulary and the question's intent."""
    hyp = hypothetical_answer(question)
    return f"{hyp}\n\n{question}"
