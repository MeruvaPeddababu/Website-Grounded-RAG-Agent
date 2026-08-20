"""
Token usage tracking and cost estimation for Gemini models.
"""
import time
from dataclasses import dataclass, field


# Gemini pricing (as of 2024)
PRICING = {
    "gemini-embedding-2": 0.025 / 1_000_000,   # $0.025/1M tokens
    "gemini-3.6-flash": {
        "input": 0.10 / 1_000_000,    # $0.10/1M input tokens
        "output": 0.40 / 1_000_000,   # $0.40/1M output tokens
    },
}

# Rough chars-per-token ratio for Gemini
CHARS_PER_TOKEN = 4


@dataclass
class TokenUsage:
    operation: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class TokenTracker:
    """Track token usage and estimate costs for Gemini pipeline."""

    def __init__(self):
        self.usage_log: list[TokenUsage] = []

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // CHARS_PER_TOKEN)

    def log_embedding(self, text: str, operation: str = "embedding"):
        tokens = self.count_tokens(text)
        cost = tokens * PRICING["gemini-embedding-2"]
        usage = TokenUsage(
            operation=operation,
            input_tokens=tokens,
            output_tokens=0,
            cost_usd=cost,
        )
        self.usage_log.append(usage)
        return usage

    def log_llm_call(
        self, input_text: str, output_text: str, operation: str = "llm_call"
    ):
        in_tokens = self.count_tokens(input_text)
        out_tokens = self.count_tokens(output_text)
        cost = (
            in_tokens * PRICING["gemini-3.6-flash"]["input"]
            + out_tokens * PRICING["gemini-3.6-flash"]["output"]
        )
        usage = TokenUsage(
            operation=operation,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=cost,
        )
        self.usage_log.append(usage)
        return usage

    def get_summary(self) -> dict:
        total_in = sum(u.input_tokens for u in self.usage_log)
        total_out = sum(u.output_tokens for u in self.usage_log)
        total_cost = sum(u.cost_usd for u in self.usage_log)

        by_operation = {}
        for u in self.usage_log:
            if u.operation not in by_operation:
                by_operation[u.operation] = {
                    "count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            by_operation[u.operation]["count"] += 1
            by_operation[u.operation]["input_tokens"] += u.input_tokens
            by_operation[u.operation]["output_tokens"] += u.output_tokens
            by_operation[u.operation]["cost_usd"] += u.cost_usd

        return {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cost_usd": total_cost,
            "by_operation": by_operation,
            "call_count": len(self.usage_log),
        }

    def estimate_cost_at_scale(self, query_tokens: int = 500) -> dict:
        """Estimate costs at 100, 1000, 10000 queries."""
        retrieval_tokens = 500
        prompt_tokens = query_tokens + 1500
        output_tokens = 300

        per_query = (
            retrieval_tokens * PRICING["gemini-embedding-2"]
            + prompt_tokens * PRICING["gemini-3.6-flash"]["input"]
            + output_tokens * PRICING["gemini-3.6-flash"]["output"]
        )

        return {
            "per_query_usd": per_query,
            "100_queries": per_query * 100,
            "1000_queries": per_query * 1000,
            "10000_queries": per_query * 10000,
            "breakdown": {
                "embedding_per_query": retrieval_tokens * PRICING["gemini-embedding-2"],
                "llm_input_per_query": prompt_tokens * PRICING["gemini-3.6-flash"]["input"],
                "llm_output_per_query": output_tokens * PRICING["gemini-3.6-flash"]["output"],
            },
        }
