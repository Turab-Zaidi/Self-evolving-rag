"""Stage 3: Generator optimization via DSPy BootstrapFewShot."""

import os
import json
import sqlite3
import itertools
import dspy
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv

load_dotenv()

ALL_KEYS = [k.strip() for k in os.getenv("NVIDIA_API_KEYS", "").split(",") if k.strip()]
BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
_key_cycle = itertools.cycle(ALL_KEYS)


def get_rotated_lm():
    return dspy.LM('openai/openai/gpt-oss-120b', api_base=BASE_URL, api_key=next(_key_cycle))

nim_llm = get_rotated_lm()
dspy.settings.configure(lm=nim_llm)
print(f"[Generator Optimizer] Loaded {len(ALL_KEYS)} API keys for rotation.")


class GenerateAnswer(dspy.Signature):
    """Answer the user's question using ONLY the provided context. Include bracketed citations."""
    context = dspy.InputField(desc="Regulatory chunks retrieved from the database.")
    original_query = dspy.InputField(desc="The user's specific question.")
    generated_answer = dspy.OutputField(desc="A precise, accurate answer with citations.")


class RAGGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, context, original_query):
        return self.generate(context=context, original_query=original_query)


def load_generator_training_data():
    conn = sqlite3.connect("logs/traces.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT original_query, generated_answer, parent_chunk_ids FROM traces "
        "WHERE cycle_number=0 AND faithfulness >= 0.8 LIMIT 20"
    ).fetchall()
    if len(rows) < 10:
        rows = conn.execute(
            "SELECT original_query, generated_answer, parent_chunk_ids FROM traces "
            "WHERE cycle_number=0 AND faithfulness >= 0.5 ORDER BY faithfulness DESC LIMIT 20"
        ).fetchall()
    dataset = []
    for r in rows:
        parent_ids = json.loads(r["parent_chunk_ids"]) if r["parent_chunk_ids"] else []
        context_str = (f"[Retrieved {len(parent_ids)} regulatory sections]" if parent_ids
                       else "[Retrieved regulatory context from eCFR and FDA Guidance databases]")
        dataset.append(dspy.Example(
            original_query=r["original_query"], context=context_str, generated_answer=r["generated_answer"]
        ).with_inputs("context", "original_query"))
    conn.close()
    return dataset


def generator_metric(example, pred, trace=None):
    judge_prompt = (
        f"Does the predicted answer contain the same regulatory facts as the gold standard answer? "
        f"Focus on factual accuracy, not exact wording.\n\n"
        f"Gold standard: {example.generated_answer}\n\n"
        f"Predicted: {pred.generated_answer}\n\n"
        f"Answer YES, PARTIALLY, or NO."
    )
    judge_lm = get_rotated_lm()
    judge_response = judge_lm(judge_prompt)[0]
    judge_text = judge_response.strip().upper() if isinstance(judge_response, str) else str(judge_response).strip().upper()
    return "YES" in judge_text or "PARTIAL" in judge_text


def optimize_generator():
    print("Loading successful traces from Cycle 0 to train the Generator...")
    trainset = load_generator_training_data()
    print(f"Loaded {len(trainset)} gold-standard examples.")
    print("Compiling Generator via BootstrapFewShot...")
    compiled_generator = BootstrapFewShot(metric=generator_metric, max_bootstrapped_demos=3).compile(
        RAGGenerator(), trainset=trainset
    )
    compiled_generator.save("correction/optimized_generator.json")
    print("Generator optimization complete!")

if __name__ == "__main__":
    optimize_generator()
