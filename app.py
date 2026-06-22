"""Streamlit web UI for Engineering Intelligence Hub.

Run: `streamlit run app.py`
"""
from __future__ import annotations

import streamlit as st

from eih.answer import ask as ask_fn
from eih.router import classify_question, route
from eih.store import get_collection

st.set_page_config(
    page_title="Engineering Intelligence Hub",
    page_icon="📚",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — settings + corpus stats
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    method = st.selectbox(
        "Retrieval method",
        options=["routed", "hybrid", "hybrid_rerank", "hybrid_hyde", "agentic", "vector", "bm25"],
        index=0,
        help=(
            "**routed** (default): auto-picks per question. "
            "**hybrid**: BM25 + vector + RRF + symbol pin. "
            "**hybrid_rerank/hybrid_hyde**: experimental, worse on eval. "
            "**agentic**: Claude calls search multiple times — expensive but powerful for multi-hop."
        ),
    )

    k = st.slider("Top-k chunks", min_value=3, max_value=15, value=6)

    st.divider()
    st.caption("**Corpus stats**")
    try:
        n = get_collection().count()
        st.metric("Chunks indexed", f"{n:,}")
    except Exception as e:
        st.warning(f"Couldn't reach Chroma: {e}")

    st.divider()
    st.caption("**About**")
    st.markdown(
        "Code-RAG over [Apache Airflow](https://github.com/apache/airflow). "
        "Honest portfolio project — see [README](https://github.com/hrishav11/Engineering-intelligence-hub-RAG) for measured numbers."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pane — question + answer + citations
# ─────────────────────────────────────────────────────────────────────────────
st.title("📚 Engineering Intelligence Hub")
st.caption("Ask questions about the Apache Airflow codebase. Answers cite real files, methods, and GitHub issues.")

# Initialize session state for the question
if "question_text" not in st.session_state:
    st.session_state.question_text = ""

# Example questions — clicking one populates the text area on the next rerun
with st.expander("💡 Try these examples"):
    examples = [
        "How does GCSHook.upload handle chunked uploads for large files?",
        "What is the difference between LocalExecutor and CeleryExecutor?",
        "How does BaseXCom.serialize_value work?",
        "When a task fails, which callbacks fire and in what order?",
        "Have there been recent issues with the task SDK?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}"):
            st.session_state.question_text = ex
            st.rerun()

# Bind the text area to session state via `key`. No `value=` — Streamlit
# manages the widget value from session_state[key] across reruns.
st.text_area(
    "Question",
    key="question_text",
    placeholder="e.g. How does the scheduler decide which DAGs to run next?",
    height=80,
)

# Read the current question from session_state, not from the local return
question = st.session_state.question_text or ""

if st.button("Ask", type="primary", disabled=not question.strip()):
    question = question.strip()
    # Show what the router chose (transparency)
    if method == "routed":
        chosen = route(question)
        category = classify_question(question)
        st.info(f"🧭 Router classified this as **{category}** → starting with **{chosen}**")

    with st.spinner("Retrieving + answering..."):
        try:
            result = ask_fn(question, k=k, method=method)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # If the system fell back from hybrid → agentic, surface that
    if result.get("used_fallback"):
        st.warning(
            f"⚡ Hybrid retrieval came back with INSUFFICIENT_CONTEXT — "
            f"automatically escalated to agentic retrieval ({result.get('turns', '?')} search turns)."
        )

    # ── Answer
    st.markdown("### Answer")
    st.markdown(result["answer"])

    # ── Sources
    st.markdown("### Sources")
    for i, hit in enumerate(result["hits"], 1):
        m = hit["meta"]
        src = hit.get("source", "")
        sym = m.get("symbol", "")
        kind = m.get("kind", "")

        # GitHub issues/PRs render as clickable URLs
        if m["source_path"].startswith("_gh/"):
            parts = m["source_path"].split("/")
            if len(parts) == 4:
                _, repo, kind_seg, num = parts
                url = f"https://github.com/apache/{repo}/{'pull' if kind_seg=='pr' else 'issues'}/{num}"
                title = f"**#{i}** [{kind.upper()} #{num}]({url}) · *{sym}*"
            else:
                title = f"**#{i}** {m['source_path']}"
        else:
            # Code/doc chunks
            sym_part = f" · `{sym}`" if sym else ""
            title = f"**#{i}** `{m['source_path']}:{m['start_line']}-{m['end_line']}` ({kind}{sym_part})"

        with st.expander(f"{title}  —  *via {src}*"):
            st.code(hit["text"][:1500], language="python" if m["source_path"].endswith(".py") else "text")

    # ── Agentic trace (if available)
    if "trace" in result:
        with st.expander(f"🔍 Agent trace ({result.get('turns', 0)} turns)"):
            for step in result["trace"]:
                st.text(step)
