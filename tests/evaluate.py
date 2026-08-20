"""
Evaluation script - runs all questions and generates a report.
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import Pipeline
from tests.eval_dataset import EVAL_QUESTIONS


def run_evaluation(pipeline: Pipeline, questions=None) -> dict:
    """Run eval questions and score results. Uses dynamic questions if provided."""
    questions = questions or EVAL_QUESTIONS
    results = []

    for item in questions:
        print(f"\nQ{item['id']} [{item['type']}]: {item['question']}")
        result = pipeline.query(item["question"])

        # Simple keyword-based scoring
        answer_lower = result["answer"].lower()
        matched = [kw for kw in item["expected_keywords"] if kw.lower() in answer_lower]

        score = 0
        if item["type"] == "unanswerable":
            # Should say "not enough info"
            if not result["enough_info"]:
                score = 1.0
            elif any(phrase in answer_lower for phrase in ["don't have enough", "not enough", "cannot answer", "no information"]):
                score = 0.8
            else:
                score = 0.0
        elif item["type"] == "misleading":
            # Refusing a false premise or honestly declaring insufficient info
            # both demonstrate correct grounding. Refusal-with-correction is best.
            corrected_premise = any(phrase in answer_lower for phrase in [
                "does not ship", "does not provide a built-in", "does not have",
                "no built-in", "integrations to external", "doesn't ship",
                "does not come with", "is not javascript", "is python",
                "primarily python", "is designed around", "pretrained",
                "is not designed to train from scratch", "fine-tun",
            ])
            if corrected_premise:
                score = 1.0
            elif not result["enough_info"]:
                score = 0.8
            else:
                if item["expected_keywords"]:
                    score = len(matched) / len(item["expected_keywords"])
                else:
                    score = 0.0
        else:
            if item["expected_keywords"]:
                score = len(matched) / len(item["expected_keywords"])
            else:
                score = 1.0 if result["answer"] else 0.0

        results.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "answer": result["answer"][:300],
            "sources": result.get("sources", []),
            "enough_info": result["enough_info"],
            "expected_keywords": item["expected_keywords"],
            "matched_keywords": matched,
            "score": round(score, 2),
            "tokens": result.get("token_usage", {}).get("total_input_tokens", 0)
                      + result.get("token_usage", {}).get("total_output_tokens", 0),
        })

        print(f"  Score: {score:.2f} | Sources: {len(result.get('sources', []))} | "
              f"Enough info: {result['enough_info']}")

    # Summary
    avg_score = sum(r["score"] for r in results) / len(results)
    type_scores = {}
    for r in results:
        t = r["type"]
        if t not in type_scores:
            type_scores[t] = []
        type_scores[t].append(r["score"])

    summary = {
        "total_questions": len(results),
        "average_score": round(avg_score, 3),
        "by_type": {t: round(sum(s) / len(s), 3) for t, s in type_scores.items()},
        "cost_summary": pipeline.get_cost_summary(),
    }

    return {"results": results, "summary": summary}


if __name__ == "__main__":
    pipeline = Pipeline(
        website_url="https://huggingface.co/docs/transformers/en/index",
        max_pages=40,
        top_k=5,
    )

    # Ensure content is indexed
    status = pipeline.ingest(force=False)
    print(f"Ingestion status: {status}")

    # Run evaluation
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)

    eval_report = run_evaluation(pipeline, questions=None)  # uses EVAL_QUESTIONS default

    # Save report
    report_path = Path(__file__).parent.parent / "data" / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    summary = eval_report["summary"]
    print(f"Total questions: {summary['total_questions']}")
    print(f"Average score: {summary['average_score']}")
    print(f"Scores by type:")
    for t, s in summary["by_type"].items():
        print(f"  {t}: {s}")

    # Cost analysis
    print("\n" + "=" * 60)
    print("COST ANALYSIS")
    print("=" * 60)
    cost = pipeline.estimate_scale_costs()
    print(f"Per query: ${cost['per_query_usd']:.6f}")
    print(f"100 queries: ${cost['100_queries']:.4f}")
    print(f"1,000 queries: ${cost['1000_queries']:.4f}")
    print(f"10,000 queries: ${cost['10000_queries']:.2f}")

    print(f"\nReport saved to: {report_path}")
