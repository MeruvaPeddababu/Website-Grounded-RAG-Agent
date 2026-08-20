"""
Pipeline orchestrator - coordinates crawling, processing, indexing, and querying.
Uses Gemini embeddings, hybrid search, cross-encoder reranking, Gemini LLM.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable

from src.crawler import WebsiteCrawler
from src.processor import ContentProcessor
from src.vectorstore import HybridVectorStore
from src.rag_chain import RAGChain
from src.token_tracker import TokenTracker


load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Pipeline:
    """End-to-end RAG pipeline orchestrator."""

    def __init__(
        self,
        website_url: str = "https://python.langchain.com",
        max_pages: int = 40,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
    ):
        self.website_url = website_url
        self.max_pages = max_pages
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # LangSmith monitoring (each step traced when env vars are set)
        if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
            project = os.getenv("LANGSMITH_PROJECT", "webpage-rag")
            print(f"[monitor] LangSmith tracing enabled -> project '{project}'")

        self.tracker = TokenTracker()
        self.vector_store = HybridVectorStore()
        self.rag_chain = RAGChain(
            vector_store=self.vector_store,
            tracker=self.tracker,
            top_k=top_k,
        )

    @traceable(
        name="ingest",
        run_type="chain",
        process_inputs=lambda i: {"force": i.get("force", False)},
    )
    def ingest(self, force: bool = False) -> dict:
        """Crawl, process, and index the website."""
        if self.vector_store.count() > 0 and not force:
            return {
                "status": "already_indexed",
                "chunks": self.vector_store.count(),
            }

        if force:
            self.vector_store.clear()

        print(f"Starting crawl of {self.website_url}...")
        crawler = WebsiteCrawler(
            base_url=self.website_url,
            max_pages=self.max_pages,
        )
        pages = crawler.crawl()

        if not pages:
            return {"status": "error", "message": "No pages crawled"}

        print(f"\nProcessing {len(pages)} pages...")
        processor = ContentProcessor(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = processor.process_pages(pages)

        total_text = " ".join(c.text for c in chunks)
        self.tracker.log_embedding(total_text, "bulk_embedding")

        print(f"\nIndexing {len(chunks)} chunks (ChromaDB + BM25)...")
        self.vector_store.add_chunks(chunks)

        summary = self.tracker.get_summary()

        return {
            "status": "success",
            "pages_crawled": len(pages),
            "chunks_created": len(chunks),
            "total_tokens_embedded": summary["total_input_tokens"],
            "embedding_cost_usd": summary["total_cost_usd"],
        }

    @traceable(
        name="rag_query",
        run_type="chain",
        process_inputs=lambda i: {"question": i["question"]},
    )
    def query(self, question: str) -> dict:
        """Ask a question against the indexed content."""
        result = self.rag_chain.invoke(question)
        return result

    def get_cost_summary(self) -> dict:
        return self.tracker.get_summary()

    def estimate_scale_costs(self) -> dict:
        return self.tracker.estimate_cost_at_scale()
