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
    page_icon="🟧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Theme toggle — must run BEFORE CSS injection so variables resolve correctly
# ─────────────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

with st.sidebar:
    st.session_state.theme = st.radio(
        "Theme",
        ["Light", "Dark"],
        horizontal=True,
        index=0 if st.session_state.theme == "Light" else 1,
        key="theme_radio",
    )

is_dark = st.session_state.theme == "Dark"


# ─────────────────────────────────────────────────────────────────────────────
# CSS — single block, variables flip between light & dark
# ─────────────────────────────────────────────────────────────────────────────
if is_dark:
    palette = """
      --bg:          #0B0B0F;
      --surface:     #16161D;
      --surface-alt: #1E1E27;
      --text:        #F5F5F5;
      --text-muted:  #94A3B8;
      --border:      #2A2A33;
      --accent:      #FF7A33;
      --accent-hover:#E5651E;
      --accent-soft: #2A1810;
      --accent-text: #FFB088;
      --shadow:      0 1px 3px rgba(0,0,0,0.4);
      --shadow-hover:0 4px 12px rgba(255,122,51,0.18);
    """
else:
    palette = """
      --bg:          #FFFFFF;
      --surface:     #FFFFFF;
      --surface-alt: #FFF7ED;
      --text:        #0F172A;
      --text-muted:  #64748B;
      --border:      #E2E8F0;
      --accent:      #EA580C;
      --accent-hover:#C2410C;
      --accent-soft: #FFF7ED;
      --accent-text: #9A3412;
      --shadow:      0 1px 3px rgba(15,23,42,0.04);
      --shadow-hover:0 2px 8px rgba(234,88,12,0.12);
    """

st.markdown(
    f"""
    <style>
      :root {{
        {palette}
      }}

      /* ─── Global ─── */
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: var(--bg) !important;
      }}
      .stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3,
      .stApp h4, .stApp h5, .stApp h6, .stApp div {{
        color: var(--text);
      }}
      .main .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 980px;
      }}

      /* ─── Sidebar ─── */
      [data-testid="stSidebar"] {{
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border);
      }}
      [data-testid="stSidebar"] * {{
        color: var(--text) !important;
      }}
      [data-testid="stSidebar"] [data-testid="stMetricValue"] {{
        color: var(--accent) !important;
      }}

      /* ─── Header ─── */
      .eih-header {{
        border-bottom: 2px solid var(--accent);
        padding-bottom: 1rem;
        margin-bottom: 2rem;
      }}
      .eih-title {{
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text);
        margin: 0;
        letter-spacing: -0.02em;
      }}
      .eih-title .accent {{ color: var(--accent); }}
      .eih-subtitle {{
        font-size: 0.95rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
      }}

      /* ─── Section labels ─── */
      .eih-label {{
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: var(--accent);
        margin-top: 1.75rem;
        margin-bottom: 0.5rem;
      }}

      /* ─── Status badges ─── */
      .eih-status {{
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 1rem;
        background: var(--accent-soft);
        color: var(--accent-text);
        border-left: 3px solid var(--accent);
        padding-left: 0.6rem;
      }}
      .eih-status.fallback {{
        background: {('#3B2C0A' if is_dark else '#FEF3C7')};
        color:      {('#FCD34D' if is_dark else '#92400E')};
        border-left: 3px solid #D97706;
      }}

      /* ─── Answer card ─── */
      .eih-answer-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
        margin: 0.5rem 0 1.5rem 0;
        box-shadow: var(--shadow);
        color: var(--text);
      }}
      .eih-answer-card code {{
        background: var(--surface-alt);
        color: var(--accent-text);
        padding: 0.1rem 0.35rem;
        border-radius: 3px;
        font-size: 0.88em;
      }}

      /* ─── Source row ─── */
      .eih-source {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
      }}
      .eih-source:hover {{
        border-color: var(--accent);
        box-shadow: var(--shadow-hover);
      }}
      .eih-source-num {{
        display: inline-block;
        background: var(--accent-soft);
        color: var(--accent);
        font-weight: 700;
        font-size: 0.78rem;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.6rem;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
      }}
      .eih-source-path {{
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.86rem;
        color: var(--text);
      }}
      .eih-source-meta {{
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
      }}
      .eih-source-tag {{
        display: inline-block;
        background: var(--surface-alt);
        color: var(--text-muted);
        font-size: 0.7rem;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        padding: 0.1rem 0.4rem;
        border-radius: 3px;
        margin-left: 0.4rem;
      }}
      .eih-source-tag.symbol {{ background: var(--accent-soft); color: var(--accent); }}

      /* ─── Buttons ─── */
      .stButton > button[kind="primary"] {{
        background: var(--accent);
        border: none;
        font-weight: 600;
        padding: 0.55rem 1.5rem;
        color: #FFFFFF !important;
        transition: background 0.15s ease;
      }}
      .stButton > button[kind="primary"]:hover {{
        background: var(--accent-hover);
      }}
      .stButton > button:not([kind="primary"]) {{
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text);
        font-weight: 400;
        text-align: left;
        white-space: normal;
        transition: all 0.15s ease;
      }}
      .stButton > button:not([kind="primary"]):hover {{
        border-color: var(--accent);
        color: var(--accent);
        background: var(--accent-soft);
      }}

      /* ─── Text area ─── */
      .stTextArea textarea {{
        background-color: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
      }}
      .stTextArea textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent);
      }}

      /* ─── Selectbox (BaseWeb) ─── */
      [data-baseweb="select"] > div {{
        background-color: var(--surface) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
      }}
      [data-baseweb="select"] > div:hover {{
        border-color: var(--accent) !important;
      }}
      [data-baseweb="select"] [data-baseweb="select-arrow"] svg,
      [data-baseweb="select"] svg {{
        fill: var(--text-muted) !important;
        color: var(--text-muted) !important;
      }}
      /* Selected value text */
      [data-baseweb="select"] [class*="ValueContainer"],
      [data-baseweb="select"] [class*="SingleValue"],
      [data-baseweb="select"] input {{
        color: var(--text) !important;
        background: transparent !important;
      }}
      /* Dropdown popover */
      [data-baseweb="popover"] [role="listbox"],
      [data-baseweb="menu"] {{
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
      }}
      [data-baseweb="menu"] li,
      [role="option"] {{
        background-color: var(--surface) !important;
        color: var(--text) !important;
      }}
      [data-baseweb="menu"] li:hover,
      [role="option"]:hover,
      [role="option"][aria-selected="true"] {{
        background-color: var(--accent-soft) !important;
        color: var(--accent) !important;
      }}

      /* ─── Radio (theme toggle) ─── */
      [data-testid="stRadio"] label,
      [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {{
        color: var(--text) !important;
      }}
      [data-testid="stRadio"] [role="radiogroup"] label > div:first-child {{
        background-color: var(--surface) !important;
        border-color: var(--border) !important;
      }}


      /* ─── Expander ─── */
      [data-testid="stExpander"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
      }}

      /* ─── Code blocks ─── */
      .stCode {{
        border-radius: 6px;
        font-size: 0.82rem;
      }}

      /* ─── Hide Streamlit chrome ─── */
      #MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}

      /* ─── Link styling inside answer ─── */
      .eih-answer-card a {{ color: var(--accent); }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar (continued — theme already up top)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("### ⚙️ Settings")

    method = st.selectbox(
        "Retrieval method",
        options=["routed", "hybrid", "hybrid_rerank", "hybrid_hyde", "agentic", "vector", "bm25"],
        index=0,
        help=(
            "**routed** (default): auto-picks per question, falls back to agentic if hybrid is insufficient. "
            "**hybrid**: BM25 + vector + RRF + symbol pin. "
            "**hybrid_rerank / hybrid_hyde**: experimental — worse on the eval set. "
            "**agentic**: Claude calls search multiple times — powerful for multi-hop."
        ),
    )

    # Top-k is fixed at the sane default; advanced users can tune via CLI.
    k = 6

    st.divider()
    st.caption("**CORPUS**")
    try:
        n = get_collection().count()
        st.metric("Chunks indexed", f"{n:,}")
    except Exception as e:
        st.warning(f"Couldn't reach Chroma: {e}")

    st.divider()
    st.caption("**ABOUT**")
    st.markdown(
        "Code-RAG over [Apache Airflow](https://github.com/apache/airflow). "
        "Honest portfolio project — see the "
        "[GitHub repo](https://github.com/hrishav11/Engineering-intelligence-hub-RAG) "
        "for measured numbers and what didn't work."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="eih-header">
      <div class="eih-title">Engineering Intelligence <span class="accent">Hub</span></div>
      <div class="eih-subtitle">Ask questions about the Apache Airflow codebase — answers cite real files, methods, and GitHub issues.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Question input + examples
# ─────────────────────────────────────────────────────────────────────────────
if "question_text" not in st.session_state:
    st.session_state.question_text = ""

st.markdown('<div class="eih-label">Try an example</div>', unsafe_allow_html=True)
example_cols = st.columns(2)
examples = [
    ("🔍", "How does GCSHook.upload handle chunked uploads?"),
    ("⚖️", "What's the difference between LocalExecutor and CeleryExecutor?"),
    ("🧬", "How does BaseXCom.serialize_value work?"),
    ("🔁", "When a task fails, which callbacks fire and in what order?"),
    ("📥", "Have there been recent issues with the task SDK?"),
    ("⚙️", "How does PythonOperator execute its callable?"),
]
for i, (emoji, ex) in enumerate(examples):
    with example_cols[i % 2]:
        if st.button(f"{emoji}  {ex}", key=f"ex_{i}", use_container_width=True):
            st.session_state.question_text = ex
            st.rerun()

st.markdown('<div class="eih-label">Your question</div>', unsafe_allow_html=True)
st.text_area(
    "Your question",
    key="question_text",
    placeholder="e.g. How does the scheduler decide which DAGs to run next?",
    height=90,
    label_visibility="collapsed",
)
question = st.session_state.question_text or ""

ask_clicked = st.button("Ask →", type="primary", disabled=not question.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Answer + sources
# ─────────────────────────────────────────────────────────────────────────────
if ask_clicked:
    question = question.strip()

    if method == "routed":
        chosen = route(question)
        category = classify_question(question)
        st.markdown(
            f'<div class="eih-status">🧭 Router classified this as '
            f'<strong>{category}</strong> → starting with <strong>{chosen}</strong></div>',
            unsafe_allow_html=True,
        )

    with st.spinner("Retrieving + answering..."):
        try:
            result = ask_fn(question, k=k, method=method)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if result.get("used_fallback"):
        st.markdown(
            f'<div class="eih-status fallback">⚡ Hybrid returned INSUFFICIENT_CONTEXT — '
            f'auto-escalated to agentic ({result.get("turns", "?")} search turns)</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="eih-label">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="eih-answer-card">{result["answer"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="eih-label">Sources</div>', unsafe_allow_html=True)
    for i, hit in enumerate(result["hits"], 1):
        m = hit["meta"]
        src = hit.get("source", "")
        sym = m.get("symbol", "")
        kind = m.get("kind", "")

        if m["source_path"].startswith("_gh/"):
            parts = m["source_path"].split("/")
            if len(parts) == 4:
                _, repo, kind_seg, num = parts
                url = f"https://github.com/apache/{repo}/{'pull' if kind_seg=='pr' else 'issues'}/{num}"
                main_label = f'<a href="{url}" target="_blank" style="color:var(--accent); text-decoration:none;">{kind.upper()} #{num}</a>'
                meta_label = sym or "(no title)"
            else:
                main_label = m["source_path"]
                meta_label = ""
        else:
            main_label = f'{m["source_path"]}:{m["start_line"]}-{m["end_line"]}'
            meta_label = f"{kind}" + (f" · {sym}" if sym else "")

        tag_html = ""
        for tag in src.split("+"):
            t = tag.strip()
            cls = "symbol" if t == "symbol" else ""
            tag_html += f'<span class="eih-source-tag {cls}">{t}</span>'

        st.markdown(
            f"""
            <div class="eih-source">
              <span class="eih-source-num">#{i}</span>
              <span class="eih-source-path">{main_label}</span>
              {tag_html}
              <div class="eih-source-meta">{meta_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("preview"):
            lang = "python" if m["source_path"].endswith(".py") else "text"
            st.code(hit["text"][:1500], language=lang)

    if "trace" in result:
        st.markdown('<div class="eih-label">Agent trace</div>', unsafe_allow_html=True)
        with st.expander(f"🔍 {result.get('turns', 0)} search turns"):
            for step in result["trace"]:
                st.text(step)
