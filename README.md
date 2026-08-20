# Website-Grounded RAG Agent

An AI agent that crawls a publicly accessible website, builds a searchable knowledge base, and answers questions grounded only in that website's content.

**Target Website:** [LangChain Documentation](https://python.langchain.com) (30+ content-rich pages)

---

## Architecture

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
│  coordinates: crawl → process → index → query                  │
└─────────┬─────────────────────────────────────────────┬────────┘
          │                                             │
          ▼                                             ▼
┌──────────────────┐                    ┌──────────────────────────┐
│  Web Crawler     │                    │    LangGraph RAG Chain   │
│  requests + BS4  │                    │                          │
│                  │                    │  ┌────────────────────┐  │
│  - BFS crawl     │                    │  │  1. Query Rewrite  │  │
│  - Clean extract │                    │  │  (Gemini 3.6 Flash)│  │
│  - Dedup         │                    │  └─────────┬──────────┘  │
└────────┬─────────┘                    │            │              │
         │                              │  ┌─────────▼──────────┐  │
         ▼                              │  │  2. Hybrid Search  │  │
┌──────────────────┐                    │  │  • Semantic (Chroma)│  │
│ Content Processor│                    │  │  • BM25 (keyword)  │  │
│ Gemini Embeddings│                    │  │  • RRF Fusion      │  │
└────────┬─────────┘                    │  └─────────┬──────────┘  │
         │                              │            │              │
         ▼                              │  ┌─────────▼──────────┐  │
┌──────────────────────────────────┐    │  │  3. Cross-Encoder │  │
│       Hybrid Vector Store        │    │  │     Reranking     │  │
│                                  │    │  └─────────┬──────────┘  │
│  ┌────────────┐ ┌─────────────┐  │    │            │              │
│  │  ChromaDB  │ │  BM25 Index │  │    │  ┌─────────▼──────────┐  │
│  │  (dense)   │ │  (sparse)   │  │    │  │  4. Answer Gen     │  │
│  └────────────┘ └─────────────┘  │    │  │  (Gemini 3.6 Flash)│  │
│            │                      │    │  │  + source citation │  │
│      ┌─────▼──────┐              │    │  └────────────────────┘  │
│      │ RRF Fusion │◄─────────────┼────┘                         │
│      └─────┬──────┘              │          ┌──────────────────┐  │
│            │                     │          │  Token Tracker   │  │
│      ┌─────▼──────┐              │          │  Cost Estimator  │  │
│      │ Cross-Enc. │              │          └──────────────────┘  │
│      │  Reranker  │              │                               │
│      └────────────┘              │                               │
└──────────────────────────────────┘                               │
```

## Data Flow

```
1. Crawl      Website → [PageContent] (title, text, headings, url)
2. Process    [PageContent] → [Chunk] (1000 chars, 200 overlap)
3. Index      [Chunk] → ChromaDB (dense) + BM25 (sparse)
4. Query      Question → Rewrite → Hybrid Search → RRF → Rerank → Answer + sources
```

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector Store | ChromaDB | Local, zero-config, persistent |
| Dense Embeddings | Gemini `gemini-embedding-2` | Free tier available, good quality |
| Sparse Search | BM25Okapi (rank_bm25) | Captures exact keyword matches |
| Fusion | Reciprocal Rank Fusion | Robustly combines dense + sparse |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Local, high precision |
| LLM | Gemini 3.6 Flash | Fast, cheap, good at grounding |
| Framework | LangGraph | Explicit state machine for RAG steps |
| Chunking | RecursiveCharacterTextSplitter | Respects document structure |
| UI | Streamlit | Fast prototyping, Python-native |

## Setup

```bash
# 1. Clone and enter directory
cd rag-pipeline

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env and add your Gemini API key (GEMINI_API_KEY=...)

# 5. Run the app
streamlit run app.py
```

## Usage

### Streamlit UI
```bash
streamlit run app.py
```
1. Enter a website URL in the sidebar (default: LangChain docs)
2. Click "Crawl & Index" to build the knowledge base
3. Ask questions in the Chat tab
4. View evaluation results in the Evaluation tab
5. See cost projections in the Cost Analysis tab

### CLI Evaluation
```bash
python tests/evaluate.py
```

## Evaluation Results

12 questions across 6 types:

| Type | Count | Description |
|------|-------|-------------|
| straightforward | 2 | Direct fact retrieval |
| paraphrased | 2 | Same intent, different wording |
| multi_page | 2 | Requires combining multiple pages |
| technical | 2 | Specific technical concepts |
| misleading | 2 | Incorrect premise in question |
| unanswerable | 2 | Info not on the website |

**Measured performance** (LangChain docs index, 1,184 chunks):

| Type | Score | Notes |
|------|-------|-------|
| straightforward | 0.67 | Direct fact retrieval, high precision |
| paraphrased | 0.67 | Paraphrase variants recognized |
| multi_page | 0.88 | Combines info across pages correctly |
| technical | 1.00 | create_agent, @tool answers exact |
| misleading | 0.80 | Refuses false premises or declares insufficient info |
| unanswerable | 1.00 | Correctly refuses off-site questions |
| **Average** | **0.83** | 12 questions across 6 types |

Note: keyword-based scoring under-credits correct answers that use synonyms (e.g. "pip" vs "package manager"), so true accuracy is higher than the raw score.

## Cost Analysis

### Pricing Used
- **Embedding (`gemini-embedding-2`):** $0.025 / 1M tokens
- **LLM Input (`gemini-3.6-flash`):** $0.10 / 1M tokens
- **LLM Output (`gemini-3.6-flash`):** $0.40 / 1M tokens

### Example Query Cost Breakdown

A typical query involves:
1. **Query embedding:** ~50 tokens → $0.000001
2. **LLM prompt (rewrite + context + answer):** ~2,000 tokens → $0.00020
3. **LLM output (answer):** ~300 tokens → $0.00012

**Total per query: ~$0.00033**

### Cost Projections

| Scale | Estimated Cost |
|-------|---------------|
| 1 query | $0.00033 |
| 100 queries | $0.033 |
| 1,000 queries | $0.33 |
| 10,000 queries | $3.30 |

### Ingestion Cost (one-time)

Crawling 30 pages → ~1,200 chunks → ~250K tokens embedded:
**Embedding cost: ~$0.007**

## Project Structure

```
rag-pipeline/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── __init__.py
│   ├── crawler.py          # Web crawler
│   ├── processor.py        # Chunking + Gemini embeddings
│   ├── vectorstore.py      # Hybrid store (ChromaDB + BM25 + RRF + rerank)
│   ├── rag_chain.py        # LangGraph RAG pipeline (Gemini 3.6 Flash)
│   ├── token_tracker.py    # Token usage + cost tracking
│   └── pipeline.py         # Orchestrator
├── tests/
│   ├── eval_dataset.py     # 12 evaluation questions
│   └── evaluate.py         # Evaluation runner
├── data/
│   └── eval_report.json    # Generated after eval
└── vectorstore/            # ChromaDB + BM25 persistence dir
```

## Known Limitations

1. **Crawler scope:** Respects `robots.txt` but doesn't handle JavaScript-rendered content
2. **Single domain:** Crawler stays within the target domain
3. **Static content:** No support for dynamically loaded content (SPAs)
4. **Embedding cost:** Re-indexing requires re-embedding all chunks
5. **Context window:** Large retrieved contexts may exceed LLM context limits
6. **Reranker model download:** Cross-encoder downloads once on first use (~90MB)
7. **BM25 tokenization:** Naive whitespace tokenization; multilingual text may need improvements

## Future Improvements

- Support JavaScript-rendered pages (Playwright)
- Multi-domain crawling with domain filtering
- Incremental indexing (only re-crawl changed pages)
- Query expansion with Gemini before retrieval
- Streaming answers in the UI
