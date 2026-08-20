"""
Dynamic evaluation question generator.
Samples content from the indexed vector store and generates
12 eval questions (6 types x 2) in parallel using Gemini.
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


QUESTION_TYPES = [
    {
        "type": "straightforward",
        "count": 2,
        "instruction": (
            "Write 2 straightforward factual questions whose answers are directly "
            "stated on the page. Each answer should require only 1-2 sentences. "
            "Pick different facts — do not ask two questions about the same thing."
        ),
    },
    {
        "type": "paraphrased",
        "count": 2,
        "instruction": (
            "Write 2 questions that ask for the same facts as a straightforward question "
            "would, but use completely different wording. The intent is identical, "
            "the phrasing is not. Good for testing semantic retrieval."
        ),
    },
    {
        "type": "multi_page",
        "count": 2,
        "instruction": (
            "Write 2 questions that require combining information from multiple pages "
            "or sections to answer fully. The answer cannot come from a single paragraph."
        ),
    },
    {
        "type": "technical",
        "count": 2,
        "instruction": (
            "Write 2 technical questions about specific APIs, methods, parameters, "
            "or code patterns mentioned in the documentation. Expected answers should "
            "include code snippets or specific function names."
        ),
    },
    {
        "type": "misleading",
        "count": 2,
        "instruction": (
            "Write 2 questions that contain a false premise about this website's subject. "
            "The correct answer must correct the misconception before answering. "
            "Example pattern: 'How do I use [X]'s built-in [feature it doesn't have]?'"
        ),
    },
    {
        "type": "unanswerable",
        "count": 2,
        "instruction": (
            "Write 2 questions that CANNOT be answered from this website's content. "
            "Good examples: financial data, personal info, competitor comparisons, "
            "future roadmap details, or internal company decisions not published in docs."
        ),
    },
]


SYSTEM_PROMPT = """You are an evaluation dataset generator for a RAG system.
Given a sample of website content, generate questions that test different retrieval and reasoning abilities.
Output ONLY valid JSON — no markdown, no code fences, no extra text.
"""

QUESTION_SCHEMA = """{
  "questions": [
    {
      "question": "<the question text>",
      "expected_keywords": ["<keyword1>", "<keyword2>"],
      "note": "<brief note explaining why this tests the stated type>"
    }
  ]
}"""


def _sample_chunks(vector_store, n: int = 30) -> str:
    """Pull a diverse random sample of chunks spread across all pages."""
    try:
        collection = vector_store.vectorstore._collection
        total = collection.count()
        if total == 0:
            return ""

        # Fetch all IDs, pick random sample
        all_ids = collection.get(include=[])["ids"]
        sampled_ids = random.sample(all_ids, min(n * 3, len(all_ids)))

        result = collection.get(ids=sampled_ids, include=["documents", "metadatas"])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])

        # Dedupe by URL, keep one chunk per page, spread across pages
        parts = []
        seen_urls: set[str] = set()
        for doc, meta in zip(docs, metas):
            url = meta.get("url", "")
            if url not in seen_urls and len(doc.strip()) > 100:
                seen_urls.add(url)
                title = meta.get("title", "")
                parts.append(f"--- [{title}] ({url}) ---\n{doc[:800]}")
            if len(parts) >= n:
                break

        return "\n\n".join(parts)
    except Exception as e:
        print(f"[eval_gen] chunk sample error: {e}")
        return ""


def _generate_for_type(llm, context: str, qtype: dict, website_url: str, start_id: int) -> list[dict]:
    """Generate questions for one type. Runs in a thread."""
    prompt = f"""Website being evaluated: {website_url}

Sample content from the website:
{context}

Task: {qtype['instruction']}

Generate exactly {qtype['count']} questions.
For each question, provide 2-4 expected_keywords that should appear in a correct answer.
For "unanswerable" type, expected_keywords must be an empty list [].

Output format:
{QUESTION_SCHEMA}"""

    def _extract(content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif hasattr(block, "text") and block.text:
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("text"):
                    parts.append(block["text"])
            return "\n".join(parts).strip()
        return str(content).strip()

    try:
        result = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = _extract(result.content)
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        questions = data.get("questions", [])[:qtype["count"]]
        out = []
        for i, q in enumerate(questions):
            out.append({
                "id": start_id + i,
                "type": qtype["type"],
                "question": q.get("question", ""),
                "expected_keywords": q.get("expected_keywords", []),
                "note": q.get("note", ""),
            })
        return out
    except Exception as e:
        print(f"[eval_gen] failed for type={qtype['type']}: {e}")
        return []


def generate_eval_questions(pipeline, website_url: str) -> list[dict]:
    """
    Sample chunks and generate 12 eval questions in parallel (2 per type).
    Returns a list in the same format as EVAL_QUESTIONS.
    """
    context = _sample_chunks(pipeline.vector_store, n=30)
    if not context:
        return []

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.4, timeout=60)

    all_questions = []
    id_counter = 1

    # Assign start IDs before launching threads
    type_start_ids = {}
    for qt in QUESTION_TYPES:
        type_start_ids[qt["type"]] = id_counter
        id_counter += qt["count"]

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(
                _generate_for_type,
                llm,
                context,
                qt,
                website_url,
                type_start_ids[qt["type"]],
            ): qt["type"]
            for qt in QUESTION_TYPES
        }

        results_by_type = {}
        for future in as_completed(futures):
            qtype = futures[future]
            try:
                qs = future.result()
                results_by_type[qtype] = qs
                print(f"[eval_gen] {qtype}: {len(qs)} questions generated")
            except Exception as e:
                print(f"[eval_gen] {qtype} thread error: {e}")
                results_by_type[qtype] = []

    # Reassemble in canonical type order, renumber IDs 1..N
    counter = 1
    for qt in QUESTION_TYPES:
        for q in results_by_type.get(qt["type"], []):
            q["id"] = counter
            all_questions.append(q)
            counter += 1

    return all_questions
