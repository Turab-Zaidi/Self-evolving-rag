"""Stage 1: Query Planner optimization via DSPy BootstrapFewShot."""

import os
import json
import sqlite3
import itertools
import dspy
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv
from pipeline.nodes.retriever import retrieval_node

load_dotenv()

ALL_KEYS = [k.strip() for k in os.getenv("NVIDIA_API_KEYS", "").split(",") if k.strip()]
BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
_key_cycle = itertools.cycle(ALL_KEYS)


def get_rotated_lm():
    return dspy.LM('openai/openai/gpt-oss-120b', api_base=BASE_URL, api_key=next(_key_cycle))

nim_llm = get_rotated_lm()
dspy.settings.configure(lm=nim_llm)
print(f"[Planner Optimizer] Loaded {len(ALL_KEYS)} API keys for rotation.")


class PlanQuery(dspy.Signature):
    """Convert user question into an optimized search query and identify the CFR Part number."""
    original_query = dspy.InputField(desc="The user's raw question.")
    rewritten_query = dspy.OutputField(desc="A search string optimized for vector retrieval.")
    cfr_part = dspy.OutputField(desc="The 3-digit CFR Part number (e.g., '820'). Output 'None' if unknown.")


class QueryPlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.ChainOfThought(PlanQuery)

    def forward(self, original_query):
        return self.plan(original_query=original_query)


def load_planner_training_data():
    conn = sqlite3.connect("logs/traces.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT original_query, ground_truth FROM traces WHERE cycle_number=0 ORDER BY RANDOM() LIMIT 60"
    ).fetchall()
    dataset = [
        dspy.Example(original_query=r["original_query"], ground_truth=r["ground_truth"]).with_inputs("original_query")
        for r in rows
    ]
    conn.close()
    return dataset


def planner_metric(example, pred, trace=None):
    ecfr_filters = {"cfr_part": pred.cfr_part} if pred.cfr_part and pred.cfr_part != "None" else {}
    state = {
        "original_query": example.original_query,
        "retrieval_plan": type('Plan', (), {
            'rewritten_query': pred.rewritten_query, 'search_mode': 'parallel',
            'collections': ['ecfr_regulations', 'fda_guidance'],
            'ecfr_filters': ecfr_filters, 'guidance_filters': {}
        })()
    }
    try:
        new_state = retrieval_node(state)
        retrieved_text = " ".join([c.text for c in new_state.get("parent_chunks", [])])
        if not retrieved_text.strip():
            return False
        judge_prompt = (
            f"You are evaluating a retrieval system. Given the following retrieved context "
            f"and a ground truth answer, determine if the context contains sufficient information "
            f"to answer the question.\n\n"
            f"Question: {example.original_query}\n\n"
            f"Ground Truth Answer: {example.ground_truth}\n\n"
            f"Retrieved Context (first 2000 chars): {retrieved_text[:2000]}\n\n"
            f"Does the retrieved context contain the key facts needed to answer the question? "
            f"Answer only YES or NO."
        )
        judge_lm = get_rotated_lm()
        judge_response = judge_lm(judge_prompt)[0]
        judge_text = judge_response if isinstance(judge_response, str) else str(judge_response)
        return "YES" in judge_text.strip().upper()
    except Exception as e:
        print(f"  Metric error: {e}")
        return False


def optimize_planner():
    print("Loading traces from Cycle 0 to train the Planner...")
    trainset = load_planner_training_data()
    print(f"Loaded {len(trainset)} training examples.")
    print("Compiling Query Planner via BootstrapFewShot (LLM-as-Judge metric)...")
    compiled_planner = BootstrapFewShot(metric=planner_metric, max_bootstrapped_demos=3).compile(
        QueryPlanner(), trainset=trainset
    )
    compiled_planner.save("correction/optimized_planner.json")
    print("Planner optimization complete!")

if __name__ == "__main__":
    optimize_planner()
