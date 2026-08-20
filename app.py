"""
Streamlit UI - Professional interface for the Gemini RAG agent.
Hybrid search (BM25 + semantic) with cross-encoder reranking.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

from src.pipeline import Pipeline
from tests.eval_dataset import EVAL_QUESTIONS

load_dotenv()

st.set_page_config(
    page_title="Website RAG Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Main background ── */
    .stApp {
        background: #f8f9fb;
        color: #1a1a1a;
    }

    /* ── Sidebar: dark slate-blue ── */
    [data-testid="stSidebar"] {
        background: #4a5a72 !important;
        border-right: none;
    }
    [data-testid="stSidebar"] > div {
        background: #4a5a72 !important;
    }
    [data-testid="stSidebar"] * {
        color: #dde3ed !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] h2:first-of-type {
        font-size: 2rem !important;
        letter-spacing: -0.02em;
        position: sticky;
        top: 0;
        z-index: 100;
        background: #4a5a72;
        padding: 0.75rem 0 0.5rem;
        margin-top: 0;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stTextInput"] label {
        color: #c8d0dd !important;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background: #3b4d63 !important;
        color: #e8edf5 !important;
        border: 1px solid #5c6e88 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] p {
        color: #a8b4c4 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #5a6a80;
        margin: 0.8rem 0;
    }
    /* Slider accent blue */
    [data-testid="stSidebar"] [data-baseweb="slider"] [data-baseweb="thumb"] {
        background: #4a9eff !important;
        border-color: #4a9eff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="slider"] [data-baseweb="track-fill"] {
        background: #4a9eff !important;
    }
    /* Sidebar button */
    [data-testid="stSidebar"] .stButton > button {
        background: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #2563eb !important;
    }
    /* Metric in sidebar */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #3b4d63;
        border: 1px solid #5c6e88;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] label {
        color: #a8b4c4 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* ── Tabs: pill style on top of theme textColor ── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: transparent !important;
        border-bottom: 1px solid #e2e6ec !important;
        padding-bottom: 8px !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: #eef0f4 !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        border: none !important;
        font-weight: 500 !important;
        color: #333333 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] p {
        color: #333333 !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: #3b82f6 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] p {
        color: #ffffff !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Chat input: fixed bottom, white, blue send button ── */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 20px;
        left: 280px;
        right: 20px;
        background: #ffffff !important;
        border: 1.5px solid #d8dde6 !important;
        border-radius: 14px !important;
        padding: 6px 12px !important;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08) !important;
        z-index: 999;
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [data-baseweb="input-container"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #1a1a1a !important;
        caret-color: #3b82f6 !important;
        font-size: 1rem;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #9ca3af !important;
    }
    /* Blue circular send button */
    [data-testid="stChatInput"] button {
        background: #3b82f6 !important;
        color: #ffffff !important;
        border-radius: 50% !important;
        border: none !important;
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
    }
    [data-testid="stChatInput"] button:hover {
        background: #2563eb !important;
    }
    section.main .block-container {
        padding-bottom: 100px !important;
    }

    /* ── Chat messages ── */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #eef2ff;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
        margin-left: 12%;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: transparent;
        border-radius: 12px;
        padding: 12px 8px;
        margin-bottom: 8px;
    }
    div[data-testid="stChatMessage"] {
        background: transparent;
        border-radius: 12px;
        padding: 12px 8px;
        margin-bottom: 8px;
    }

    /* ── Typography ── */
    h1 {
        color: #0f172a;
        font-weight: 700;
        font-size: 2rem;
        letter-spacing: -0.03em;
    }
    h2, h3 {
        color: #1e293b;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .stCaption {
        color: #64748b;
        font-size: 0.9rem;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
    }
    [data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700;
    }

    /* ── Main area button ── */
    .stButton > button[kind="primary"] {
        background: #3b82f6;
        border: none;
        color: #ffffff !important;
        font-weight: 600;
        border-radius: 8px;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2563eb;
    }

    /* ── Expanders & misc ── */
    [data-testid="stAlert"] { border-radius: 8px; }
    [data-baseweb="expandable"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: #ffffff;
    }
    [data-testid="stExpander"] summary {
        color: #1e293b;
        font-weight: 500;
    }
    .stDivider { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## RAG Agent")

    st.divider()

    website_url = st.text_input(
        "Website URL",
        value="https://python.langchain.com",
        help="Publicly accessible website to crawl",
    )
    max_pages = st.slider("Max pages to crawl", 10, 500, 100)
    top_k = st.slider("Retrieved chunks (top_k)", 2, 10, 5)

    st.divider()

    if st.button("🔄 Crawl & Index", type="primary", use_container_width=True):
        with st.spinner("Crawling website..."):
            pipeline = Pipeline(
                website_url=website_url,
                max_pages=max_pages,
                top_k=top_k,
            )
            result = pipeline.ingest(force=True)
            st.session_state["pipeline"] = pipeline
            st.session_state["ingest_result"] = result
            st.session_state.pop("eval_questions", None)  # clear stale questions
            if result["status"] == "success":
                st.success(
                    f"Indexed {result.get('chunks_created', 0)} chunks "
                    f"from {result.get('pages_crawled', 0)} pages"
                )
                with st.spinner("Generating evaluation questions for this site..."):
                    from tests.eval_generator import generate_eval_questions
                    generated = generate_eval_questions(pipeline, website_url)
                    if generated:
                        st.session_state["eval_questions"] = generated
                        st.session_state["eval_questions_url"] = website_url
                        st.caption(f"Generated {len(generated)} eval questions.")
            else:
                st.warning(result.get("message", result["status"]))



# --- Main Area ---
st.title("Website-Grounded RAG Agent")
st.caption("Ask questions answered only from website content · Gemini + Hybrid Search + Cross-Encoder Reranking")

# Auto-load pipeline
if "pipeline" not in st.session_state:
    pipeline = Pipeline(website_url=website_url, top_k=top_k)
    if pipeline.vector_store.count() > 0:
        st.session_state["pipeline"] = pipeline
        if "eval_questions" not in st.session_state:
            with st.spinner("Generating evaluation questions..."):
                from tests.eval_generator import generate_eval_questions
                generated = generate_eval_questions(pipeline, website_url)
                if generated:
                    st.session_state["eval_questions"] = generated
                    st.session_state["eval_questions_url"] = website_url
                    st.rerun()

# Tabs
tab_chat, tab_eval, tab_costs, tab_arch = st.tabs(
    ["💬 Chat", "📊 Evaluation", "💰 Cost Analysis", "🏗️ Architecture"]
)

# --- Chat Tab ---
with tab_chat:
    if "pipeline" not in st.session_state:
        st.info("👈 Crawl & index a website first using the sidebar.")
    else:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if question := st.chat_input("Ask a question about the website..."):
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state["pipeline"].query(question)

                st.markdown(result["answer"])

                if result.get("sources"):
                    with st.container():
                        for i, src in enumerate(result["sources"], 1):
                            st.caption(f"[{i}] {src}")

            st.session_state["messages"].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })

# --- Evaluation Tab ---
with tab_eval:
    eval_questions_url = st.session_state.get("eval_questions_url", "")
    url_matches = eval_questions_url.rstrip("/") == website_url.rstrip("/")
    has_questions = "eval_questions" in st.session_state and url_matches

    col_h, col_btn = st.columns([3, 1])
    with col_h:
        st.subheader("Evaluation Dataset")
        if has_questions:
            active_questions = st.session_state["eval_questions"]
            st.caption(f"Auto-generated for **{website_url}** · {len(active_questions)} questions · 6 types")
        else:
            st.info("👈 Crawl & index this site first to generate evaluation questions.")

    if not has_questions:
        active_questions = []
    with col_btn:
        if st.session_state.get("pipeline") and st.button("🔁 Regenerate", help="Regenerate questions for current site"):
            with st.spinner("Generating questions..."):
                from tests.eval_generator import generate_eval_questions
                generated = generate_eval_questions(
                    st.session_state["pipeline"], website_url
                )
                if generated:
                    st.session_state["eval_questions"] = generated
                    st.session_state["eval_questions_url"] = website_url
                    st.rerun()

    if not has_questions:
        active_questions = []

    type_color_map = {
        "straightforward": "green",
        "paraphrased": "blue",
        "multi_page": "orange",
        "technical": "purple",
        "misleading": "red",
        "unanswerable": "gray",
    }

    for item in active_questions:
        type_color = type_color_map.get(item["type"], "gray")
        with st.expander(f"Q{item['id']} [{item['type']}] {item['question'][:70]}"):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Type:** :{type_color}[{item['type']}]")
            if item.get("expected_keywords"):
                st.markdown(f"**Expected keywords:** {', '.join(item['expected_keywords'])}")
            if item.get("note"):
                st.markdown(f"**Note:** {item['note']}")

    if active_questions:
        st.divider()

    if active_questions and st.button("▶️ Run Full Evaluation", type="primary"):
        if "pipeline" not in st.session_state:
            st.error("Index a website first.")
        else:
            with st.spinner(f"Running {len(active_questions)} evaluation questions..."):
                from tests.evaluate import run_evaluation
                eval_report = run_evaluation(
                    st.session_state["pipeline"],
                    questions=active_questions,
                )

            st.success("Evaluation complete!")

            summary = eval_report["summary"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Score", f"{summary['average_score']:.2f}")
            with col2:
                st.metric("Questions", summary["total_questions"])
            with col3:
                st.metric("Total Cost", f"${summary['cost_summary']['total_cost_usd']:.4f}")

            st.subheader("Scores by Type")
            for t, s in summary["by_type"].items():
                st.progress(s, text=f"{t}: {s:.2f}")

            st.subheader("Detailed Results")
            for r in eval_report["results"]:
                color = "green" if r["score"] >= 0.7 else "orange" if r["score"] >= 0.4 else "red"
                st.markdown(
                    f"**Q{r['id']}** [{r['type']}] Score: :{color}[{r['score']}] — {r['question'][:80]}"
                )
                if r["matched_keywords"]:
                    st.caption(f"Matched: {r['matched_keywords']}")
                if r["sources"]:
                    st.caption(f"Sources: {', '.join(r['sources'][:3])}")

# --- Cost Tab ---
with tab_costs:
    st.subheader("Cost Analysis — Gemini Models")

    if "pipeline" in st.session_state:
        tracker_summary = st.session_state["pipeline"].get_cost_summary()
        scale_costs = st.session_state["pipeline"].estimate_scale_costs()

        st.markdown("### Current Session")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Tokens",
                f"{tracker_summary['total_input_tokens'] + tracker_summary['total_output_tokens']:,}",
            )
        with col2:
            st.metric("Total Cost", f"${tracker_summary['total_cost_usd']:.6f}")
        with col3:
            st.metric("API Calls", tracker_summary["call_count"])

        st.markdown("### Cost Projections")
        proj_data = {
            "Scale": ["Per Query", "100 Queries", "1,000 Queries", "10,000 Queries"],
            "Cost (USD)": [
                f"${scale_costs['per_query_usd']:.6f}",
                f"${scale_costs['100_queries']:.4f}",
                f"${scale_costs['1000_queries']:.4f}",
                f"${scale_costs['10000_queries']:.2f}",
            ],
        }
        st.table(proj_data)

        st.markdown("### Per-Query Breakdown")
        breakdown = scale_costs["breakdown"]
        st.json({
            "Embedding (query)": f"${breakdown['embedding_per_query']:.7f}",
            "LLM Input (prompt)": f"${breakdown['llm_input_per_query']:.6f}",
            "LLM Output (answer)": f"${breakdown['llm_output_per_query']:.6f}",
        })
    else:
        st.info("Index a website first to see cost analysis.")

    st.divider()
    st.markdown("### Gemini Pricing Reference")
    st.markdown("""
    | Model | Input | Output |
    |-------|-------|--------|
    | gemini-embedding-2 | $0.025/1M tokens | — |
    | gemini-3.6-flash | $0.10/1M tokens | $0.40/1M tokens |
    """)

# --- Architecture Tab ---
with tab_arch:
    st.subheader("System Architecture")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                        Streamlit UI                            │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
    │  │  Chat Input  │  │  Eval Panel  │  │   Cost Dashboard     │  │
    │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
    └─────────┼─────────────────┼─────────────────────┼──────────────┘
              │                 │                     │
              ▼                 ▼                     ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Pipeline Orchestrator                        │
    └─────────┬─────────────────────────────────────────────┬────────┘
              │                                             │
              ▼                                             ▼
    ┌──────────────────┐                    ┌──────────────────────────┐
    │  Web Crawler     │                    │    LangGraph RAG Chain   │
    │  requests + BS4  │                    │                          │
    └────────┬─────────┘                    │  ┌────────────────────┐  │
             │                              │  │ 1. Query Rewrite   │  │
             ▼                              │  │    (Gemini Flash)  │  │
    ┌──────────────────┐                    │  └─────────┬──────────┘  │
    │ Content Processor│                    │            │              │
    │ Gemini Embeddings│                    │  ┌─────────▼──────────┐  │
    └────────┬─────────┘                    │  │ 2. Hybrid Search   │  │
             │                              │  │  • Semantic (Chroma)│  │
             ▼                              │  │  • BM25 (keyword)  │  │
    ┌──────────────────────────────────┐    │  │  • RRF Fusion      │  │
    │       Hybrid Vector Store        │    │  └─────────┬──────────┘  │
    │                                  │    │            │              │
    │  ┌────────────┐ ┌─────────────┐  │    │  ┌─────────▼──────────┐  │
    │  │  ChromaDB  │ │  BM25 Index │  │    │  │ 3. Cross-Encoder   │  │
    │  │  (dense)   │ │  (sparse)   │  │    │  │    Reranking       │  │
    │  └────────────┘ └─────────────┘  │    │  └─────────┬──────────┘  │
    │            │                      │    │            │              │
    │      ┌─────▼──────┐              │    │  ┌─────────▼──────────┐  │
    │      │ RRF Fusion │◄─────────────┼────┤  │ 4. Answer Gen      │  │
    │      └─────┬──────┘              │    │  │    (Gemini Flash)  │  │
    │            │                     │    │  │  + source citation │  │
    │      ┌─────▼──────┐              │    │  └────────────────────┘  │
    │      │ Cross-Enc. │              │    └──────────┬───────────────┘
    │      │  Reranker  │              │               │
    │      └────────────┘              │               ▼
    └──────────────────────────────────┘    ┌──────────────────┐
                                            │  Token Tracker   │
                                            │  Cost Estimator  │
                                            └──────────────────┘
    ```

    ### Retrieval Pipeline
    1. **Query Rewrite** — Gemini rewrites user query for better retrieval
    2. **Semantic Search** — Gemini `gemini-embedding-2` via ChromaDB
    3. **BM25 Search** — Keyword matching with rank_bm25
    4. **RRF Fusion** — Reciprocal Rank Fusion combines both result lists
    5. **Cross-Encoder Reranking** — ms-marco-MiniLM-L-6-v2 for precision
    6. **Answer Generation** — Gemini 3.6 Flash with grounded prompting
    """)

    st.markdown("""
    ### Key Components
    | Component | Technology | Purpose |
    |-----------|-----------|---------|
    | Crawler | requests + BeautifulSoup | BFS crawl, clean extraction |
    | Chunking | RecursiveCharacterTextSplitter | 1000 chars, 200 overlap |
    | Dense Embeddings | Gemini `gemini-embedding-2` | Semantic vector search |
    | Sparse Search | BM25Okapi (rank_bm25) | Keyword matching |
    | Fusion | Reciprocal Rank Fusion | Combine dense + sparse results |
    | Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Precision reranking |
    | Vector DB | ChromaDB (persistent) | Local vector storage |
    | LLM | Gemini 3.6 Flash | Query rewrite + answer generation |
    | Framework | LangGraph | State machine for RAG steps |
    """)
