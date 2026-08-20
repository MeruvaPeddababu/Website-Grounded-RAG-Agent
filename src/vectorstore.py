"""
Vector store with hybrid search (BM25 + semantic) and cross-encoder reranking.
"""
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import traceable
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.processor import Chunk


VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"
COLLECTION_NAME = "rag_docs"
BM25_INDEX_PATH = VECTORSTORE_DIR / "bm25_index.pkl"

_CHROMA_LOCK = threading.Lock()
_chroma_client_cache: dict[str, Any] = {}

EMBED_BATCH_SIZE = 16
EMBED_MAX_RETRIES = 5
EMBED_RETRY_DELAY = 3.0


class HybridVectorStore:
    """ChromaDB + BM25 hybrid search with cross-encoder reranking."""

    def __init__(self):
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            task_type="retrieval_query",
        )

        self.chroma_client = self._get_client()

        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

        # BM25 index
        self.bm25: BM25Okapi | None = None
        self.bm25_docs: list[dict] = []  # metadata for each BM25 doc
        self.bm25_corpus: list[str] = []  # tokenized texts

        # Cross-encoder reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self._load_bm25_index()

    @staticmethod
    def _get_client() -> Any:
        """Return a process-wide Chroma client.

        Chroma's SharedSystemClient is not safe against concurrent creation for
        the same path; Streamlit reruns the script in threads. Cache a single
        client under a lock to avoid the race (KeyError / missing bindings).
        """
        key = str(VECTORSTORE_DIR)
        with _CHROMA_LOCK:
            if key not in _chroma_client_cache:
                _chroma_client_cache[key] = chromadb.PersistentClient(
                    path=key,
                    settings=Settings(anonymized_telemetry=False),
                )
            return _chroma_client_cache[key]

    @traceable(
        name="embedding_gemini",
        run_type="embedding",
        process_inputs=lambda i: {"texts_to_embed": len(i["texts"])},
        process_outputs=lambda emb: {
            "vectors_generated": len(emb),
            "dimension": len(emb[0]) if emb else 0,
        },
    )
    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in small batches with retry on transient errors."""
        all_embeddings = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            for attempt in range(1, EMBED_MAX_RETRIES + 1):
                try:
                    emb = self.embeddings.embed_documents(batch)
                    all_embeddings.extend(emb)
                    break
                except Exception as e:
                    if attempt == EMBED_MAX_RETRIES:
                        raise
                    print(f"  [embed] batch {i} attempt {attempt} failed: "
                          f"{type(e).__name__}; retrying in {EMBED_RETRY_DELAY}s")
                    time.sleep(EMBED_RETRY_DELAY * attempt)
            time.sleep(0.2)  # rate-limit guard
        return all_embeddings

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _load_bm25_index(self):
        if BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.bm25_docs = data["docs"]
                self.bm25_corpus = data["corpus"]
            print(f"Loaded BM25 index: {len(self.bm25_docs)} documents")

    def _save_bm25_index(self):
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "docs": self.bm25_docs,
                "corpus": self.bm25_corpus,
            }, f)

    @traceable(
        name="embed_and_index",
        run_type="tool",
        process_inputs=lambda i: {
            "chunks_to_index": len(i["chunks"]),
            "total_chars": sum(len(c.text) for c in i["chunks"]),
        },
    )
    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100):
        """Add chunks to both ChromaDB and BM25 index."""
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "url": c.url,
                "title": c.title,
                "chunk_index": c.chunk_index,
                "headings": " > ".join(c.headings[:5]),
            }
            for c in chunks
        ]
        ids = [f"{c.url}#{c.chunk_index}" for c in chunks]

        # Embed in small batches with retry
        print(f"  Embedding {len(texts)} chunks (batch size {EMBED_BATCH_SIZE})...")
        embeddings = self._embed_with_retry(texts)
        print(f"  Embedding done: {len(embeddings)} vectors")

        # Batch insert into raw ChromaDB collection with precomputed embeddings
        collection = self.vectorstore._collection
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
            print(f"  ChromaDB batch {i // batch_size + 1}: {end}/{len(texts)}")

        # Build BM25 index
        self.bm25_corpus = [self._tokenize(t) for t in texts]
        self.bm25_docs = [
            {"text": texts[i], "metadata": metadatas[i], "id": ids[i]}
            for i in range(len(texts))
        ]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        self._save_bm25_index()

        print(f"Hybrid store ready: {len(chunks)} chunks (ChromaDB + BM25)")

    @traceable(
        name="semantic_search_chroma",
        run_type="retriever",
        process_inputs=lambda i: {"query": i["query"], "k": i["k"]},
        process_outputs=lambda docs: [
            {"url": d["url"], "title": d["title"], "score": d["score"]}
            for d in docs
        ],
    )
    def _semantic_search(self, query: str, k: int) -> list[dict]:
        """Dense vector search via ChromaDB."""
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return [
            {
                "text": doc.page_content,
                "url": doc.metadata.get("url", ""),
                "title": doc.metadata.get("title", ""),
                "score": score,
                "source": "semantic",
                "headings": doc.metadata.get("headings", ""),
            }
            for doc, score in results
        ]

    @traceable(
        name="bm25_search",
        run_type="retriever",
        process_inputs=lambda i: {"query": i["query"], "k": i["k"]},
        process_outputs=lambda docs: [
            {"url": d["url"], "title": d["title"], "score": d["score"]}
            for d in docs
        ],
    )
    def _bm25_search(self, query: str, k: int) -> list[dict]:
        """Sparse keyword search via BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.bm25_docs[idx]
                results.append({
                    "text": doc["text"],
                    "url": doc["metadata"]["url"],
                    "title": doc["metadata"]["title"],
                    "score": float(scores[idx]),
                    "source": "bm25",
                    "headings": doc["metadata"].get("headings", ""),
                })
        return results

    @traceable(
        name="rrf_fusion",
        run_type="retriever",
        process_inputs=lambda i: {
            "semantic_results": len(i["semantic_results"]),
            "bm25_results": len(i["bm25_results"]),
        },
        process_outputs=lambda docs: [
            {"url": d["url"], "title": d["title"], "rrf_score": d["rrf_score"]}
            for d in docs
        ],
    )
    def _rrf_fusion(
        self, semantic_results: list[dict], bm25_results: list[dict], k: int = 60
    ) -> list[dict]:
        """Reciprocal Rank Fusion to combine result lists."""
        doc_scores: dict[str, float] = {}
        doc_data: dict[str, dict] = {}

        # Score semantic results
        for rank, r in enumerate(semantic_results):
            doc_id = f"{r['url']}#{r.get('title', '')}"
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_data[doc_id] = r

        # Score BM25 results
        for rank, r in enumerate(bm25_results):
            doc_id = f"{r['url']}#{r.get('title', '')}"
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = r

        # Sort by fused score
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        results = []
        for doc_id in sorted_ids:
            entry = doc_data[doc_id].copy()
            entry["rrf_score"] = doc_scores[doc_id]
            results.append(entry)

        return results

    @traceable(
        name="rerank_cross_encoder",
        run_type="retriever",
        process_inputs=lambda i: {
            "query": i["query"],
            "candidates": len(i["results"]),
            "top_n": i["top_n"],
        },
        process_outputs=lambda docs: [
            {
                "url": d["url"],
                "title": d["title"],
                "rerank_score": d.get("rerank_score"),
                "text_preview": d["text"][:200],
            }
            for d in docs
        ],
    )
    def _rerank(self, query: str, results: list[dict], top_n: int) -> list[dict]:
        """Cross-encoder reranking for precision."""
        if not results:
            return []

        pairs = [(query, r["text"][:512]) for r in results]
        rerank_scores = self.reranker.predict(pairs)

        for i, score in enumerate(rerank_scores):
            results[i]["rerank_score"] = float(score)

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_n]

    @traceable(
        name="hybrid_search",
        run_type="retriever",
        process_inputs=lambda i: {"query": i["query"], "k": i["k"]},
        process_outputs=lambda docs: {
            "final_results": len(docs),
            "docs": [
                {
                    "url": d["url"],
                    "title": d["title"],
                    "rerank_score": d.get("rerank_score"),
                    "text_preview": d["text"][:200],
                }
                for d in docs
            ],
        },
    )
    def hybrid_search(self, query: str, k: int = 5, top_k_candidates: int = 20) -> list[dict]:
        """Full hybrid pipeline: semantic + BM25 -> RRF fusion -> rerank."""
        # Retrieve candidates from both sources
        semantic_results = self._semantic_search(query, k=top_k_candidates)
        bm25_results = self._bm25_search(query, k=top_k_candidates)

        print(f"  Semantic: {len(semantic_results)} results")
        print(f"  BM25: {len(bm25_results)} results")

        # Fuse with RRF
        fused = self._rrf_fusion(semantic_results, bm25_results)
        print(f"  After RRF fusion: {len(fused)} candidates")

        # Rerank with cross-encoder
        reranked = self._rerank(query, fused, top_n=k)
        print(f"  After reranking: {len(reranked)} final results")

        return reranked

    def get_retriever(self, k: int = 5):
        """Return a callable for the RAG chain."""
        def retrieve(query: str) -> list[dict]:
            return self.hybrid_search(query, k=k)
        return retrieve

    def count(self) -> int:
        return self.vectorstore._collection.count()

    def clear(self):
        self.chroma_client.delete_collection(COLLECTION_NAME)
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )
        self.bm25 = None
        self.bm25_docs = []
        self.bm25_corpus = []
        if BM25_INDEX_PATH.exists():
            BM25_INDEX_PATH.unlink()
