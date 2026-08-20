"""
Evaluation dataset - 12 questions for HuggingFace Transformers docs.
Covers 6 types: straightforward, paraphrased, multi_page, technical, misleading, unanswerable.
"""
EVAL_QUESTIONS = [
    # Straightforward (direct fact retrieval)
    {
        "id": 1,
        "type": "straightforward",
        "question": "What is HuggingFace Transformers?",
        "expected_keywords": ["transformers", "models", "pretrained"],
        "note": "Basic definition from the Transformers overview/philosophy page",
    },
    {
        "id": 2,
        "type": "straightforward",
        "question": "How do you install the HuggingFace Transformers library?",
        "expected_keywords": ["pip install", "transformers"],
        "note": "Installation instructions from the getting started / installation page",
    },

    # Paraphrased (same intent, different wording)
    {
        "id": 3,
        "type": "paraphrased",
        "question": "What is the easiest way to get Transformers set up on my machine?",
        "expected_keywords": ["pip", "install", "transformers"],
        "note": "Paraphrase of Q2 — tests semantic retrieval across different wording",
    },
    {
        "id": 4,
        "type": "paraphrased",
        "question": "In plain terms, what problem does the Transformers library solve?",
        "expected_keywords": ["pretrained", "models", "NLP"],
        "note": "Paraphrase of Q1 — rephrased as 'what problem does it solve'",
    },

    # Multi-page (requires combining info from multiple pages)
    {
        "id": 5,
        "type": "multi_page",
        "question": "What pipeline tasks are available in Transformers and how do you use them?",
        "expected_keywords": ["pipeline", "task", "model"],
        "note": "Requires pipeline_tutorial + main_classes/pipelines pages",
    },
    {
        "id": 6,
        "type": "multi_page",
        "question": "How does the Trainer class work and what arguments does it accept?",
        "expected_keywords": ["Trainer", "TrainingArguments", "train"],
        "note": "Spans Trainer overview and TrainingArguments reference pages",
    },

    # Technical detail
    {
        "id": 7,
        "type": "technical",
        "question": "How do you load a pretrained model and tokenizer in Transformers?",
        "expected_keywords": ["from_pretrained", "tokenizer", "AutoModel"],
        "note": "Core API — expects from_pretrained pattern",
    },
    {
        "id": 8,
        "type": "technical",
        "question": "How do you run text generation with a language model using the Transformers pipeline?",
        "expected_keywords": ["pipeline", "text-generation", "generate"],
        "note": "llm_tutorial or pipeline_tutorial — specific generation workflow",
    },

    # Misleading (question implies incorrect premise)
    {
        "id": 9,
        "type": "misleading",
        "question": "How do I use HuggingFace Transformers to train a model from scratch without any pretrained weights?",
        "expected_keywords": ["pretrained", "from scratch", "weights"],
        "note": "Transformers is built around pretrained weights — answer should clarify the library's design philosophy",
    },
    {
        "id": 10,
        "type": "misleading",
        "question": "What is the default programming language for HuggingFace Transformers — is it JavaScript?",
        "expected_keywords": ["Python", "PyTorch", "TensorFlow"],
        "note": "False premise — Transformers is primarily Python, not JavaScript",
    },

    # Unanswerable (not on the website)
    {
        "id": 11,
        "type": "unanswerable",
        "question": "What is the total funding raised by HuggingFace Inc as of 2024?",
        "expected_keywords": [],
        "note": "Financial data not in Transformers documentation",
    },
    {
        "id": 12,
        "type": "unanswerable",
        "question": "How does HuggingFace Transformers compare to OpenAI's GPT-4 in benchmark tests?",
        "expected_keywords": [],
        "note": "Competitive benchmarks against OpenAI not in HF docs",
    },
]
