"""
RAG chain using LangGraph with Gemini LLM and hybrid retrieval.

Steps (each monitored as a LangSmith span when LANGSMITH_TRACING=true):
  1. rewrite_query  - query rewriting via Gemini
  2. retrieve       - hybrid search (semantic + BM25 + rerank)
  3. generate_answer - grounded answer generation via Gemini
"""
import os
import operator
from typing import Annotated, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.vectorstore import HybridVectorStore
from src.token_tracker import TokenTracker


RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on website documentation.

Rules:
1. ONLY use information from the provided context to answer questions.
2. If the context does not contain enough information, clearly state: "I don't have enough information from the website to answer this question."
3. Always cite the source URL(s) supporting your answer in the format [Source: URL].
4. Give thorough, detailed answers of 3-7 sentences. Explain the concept, mention key details, sub-features, or usage patterns found in the context. Do not just summarize in one line.
5. If multiple sources provide different perspectives, mention both.
6. If the question asserts something that the documentation contradicts (e.g. claims LangChain ships a built-in vector database), explicitly correct the misconception using the context BEFORE answering.
7. Do not make up or infer information not present in the context.

Context from website:
{context}
"""


class RAGState(TypedDict):
    question: str
    rewritten_question: str
    retrieved_docs: list[dict]
    context: str
    answer: str
    sources: list[str]
    enough_info: bool
    token_usage: dict


class RAGChain:
    """LangGraph-based RAG pipeline with Gemini and hybrid retrieval."""

    def __init__(
        self,
        vector_store: HybridVectorStore,
        tracker: TokenTracker,
        model_name: str = "gemini-3.6-flash",
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.tracker = tracker
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, timeout=60)
        self.model_name = model_name
        self.top_k = top_k
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(RAGState)

        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate_answer", self._generate_answer)

        graph.set_entry_point("rewrite_query")
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("retrieve", "generate_answer")
        graph.add_edge("generate_answer", END)

        return graph.compile()

    def _extract_text(self, content) -> str:
        """Extract plain text from model output (str or list of blocks)."""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif hasattr(block, "text") and block.text:
                    parts.append(block.text)
                elif isinstance(block, dict):
                    t = block.get("text", "")
                    if t:
                        parts.append(t)
            return "\n".join(parts).strip()
        return str(content).strip()

    def _rewrite_query(self, state: RAGState) -> RAGState:
        """Rewrite the user query for better retrieval."""
        question = state["question"]

        rewrite_prompt = f"""Rewrite this question to improve document retrieval.
Keep the core intent. Output ONLY the rewritten question, nothing else.

Original: {question}
Rewritten:"""

        messages = [HumanMessage(content=rewrite_prompt)]
        result = self.llm.invoke(messages)

        rewritten = self._extract_text(result.content)
        self.tracker.log_llm_call(rewrite_prompt, rewritten, "query_rewrite")

        return {**state, "rewritten_question": rewritten}

    def _retrieve(self, state: RAGState) -> RAGState:
        """Retrieve via hybrid search (semantic + BM25 + rerank)."""
        query = state["rewritten_question"]

        docs = self.vector_store.hybrid_search(query, k=self.top_k)

        self.tracker.log_embedding(query, "query_embedding")

        context_parts = []
        sources = []
        for doc in docs:
            score_info = f" [rerank:{doc.get('rerank_score', 0):.2f}]"
            context_parts.append(f"[Source: {doc['url']}]{score_info}\n{doc['text']}")
            if doc["url"] not in sources:
                sources.append(doc["url"])

        context = "\n\n---\n\n".join(context_parts)

        return {
            **state,
            "retrieved_docs": docs,
            "context": context,
            "sources": sources,
        }

    def _generate_answer(self, state: RAGState) -> RAGState:
        """Generate an answer grounded in retrieved context."""
        context = state["context"]
        question = state["question"]

        system_msg = RAG_SYSTEM_PROMPT.format(context=context)
        messages = [SystemMessage(content=system_msg), HumanMessage(content=question)]

        result = self.llm.invoke(messages)
        answer = self._extract_text(result.content)

        self.tracker.log_llm_call(
            system_msg + "\n\n" + question, answer, "answer_generation"
        )

        insufficient_phrases = [
            "don't have enough information",
            "not enough information",
            "cannot answer",
            "no information available",
            "not mentioned in the provided",
            "not found in the context",
            "doesn't contain information",
        ]
        enough_info = not any(p in answer.lower() for p in insufficient_phrases)

        return {
            **state,
            "answer": answer,
            "enough_info": enough_info,
        }
    def invoke(self, question: str) -> dict:
        """Run the full RAG pipeline with LangSmith step monitoring."""
        initial_state: RAGState = {
            "question": question,
            "rewritten_question": "",
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "sources": [],
            "enough_info": True,
            "token_usage": {},
        }

        tracing_on = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        if tracing_on:
            print(f"  [tracing] Step spans sent to LangSmith project: "
                  f"{os.getenv('LANGSMITH_PROJECT', 'default')}")

        result = self.graph.invoke(
            initial_state,
            config={
                "run_name": "WebsiteRAG",
                "metadata": {
                    "question": question,
                    "top_k": self.top_k,
                    "model": self.model_name,
                    "embedding_model": "gemini-embedding-2",
                    "retrieval": "hybrid (semantic+BM25+rerank)",
                },
            },
        )
        result["token_usage"] = self.tracker.get_summary()
        return result
