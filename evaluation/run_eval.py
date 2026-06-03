"""Evaluation runner: runs questions through the RAG pipeline, scores with RAGAS, saves results."""

import sys
import os
import json
import math
import logging
import argparse
import uuid
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.graph import rag_pipeline
from pipeline.tracer import save_trace, update_eval_scores
from pipeline.state import RAGASScores
from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def compute_ragas_scores(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    from ragas.metrics import Faithfulness, ContextRecall, ContextPrecision, AnswerRelevancy
    from ragas import evaluate
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "question": [question], "answer": [answer],
        "contexts": [contexts], "ground_truth": [ground_truth],
    })
    try:
        import builtins; builtins.uuid = uuid
        results = evaluate(
            dataset,
            metrics=[Faithfulness(), ContextRecall(), ContextPrecision(), AnswerRelevancy()],
            llm=_get_ragas_llm(), embeddings=_get_ragas_embeddings(),
            raise_exceptions=False,
        )
        df = results.to_pandas()
        return {
            "faithfulness": float(df["faithfulness"].iloc[0]),
            "context_recall": float(df["context_recall"].iloc[0]),
            "context_precision": float(df["context_precision"].iloc[0]),
            "answer_relevancy": float(df["answer_relevancy"].iloc[0]),
            "answer_correctness": None,
        }
    except Exception as e:
        logger.warning(f"RAGAS scoring failed: {e}")
        return {k: None for k in ["faithfulness", "context_recall", "context_precision", "answer_relevancy", "answer_correctness"]}


def _get_ragas_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    import random
    return LangchainLLMWrapper(ChatOpenAI(
        model=cfg.nim.EVALUATION, api_key=random.choice(cfg.nim.API_KEYS),
        base_url=cfg.nim.BASE_URL, temperature=0.0,
    ))


def _get_ragas_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"},
    ))


def run_evaluation(eval_set_path: Path, cycle_number: int):
    logger.info(f"=== Starting Evaluation (Cycle {cycle_number}) ===")
    with open(eval_set_path, encoding="utf-8") as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} questions.")

    results = []
    pipeline_errors = 0

    for i, item in enumerate(questions):
        q_id = item.get("query_id", str(uuid.uuid4()))
        question = item["question"]
        gt = item.get("ground_truth", "")
        difficulty = item.get("difficulty", "unknown")
        logger.info(f"[{i+1}/{len(questions)}] [{difficulty.upper()}] {question[:70]}...")

        try:
            final_state = rag_pipeline.invoke({
                "query_id": q_id, "original_query": question, "ground_truth": gt,
                "cycle_number": cycle_number, "planner_version": 1, "generator_version": 1,
            })
        except Exception as e:
            logger.error(f"Pipeline failed for query {q_id}: {e}")
            pipeline_errors += 1
            continue

        row_id = save_trace(final_state)
        parent_chunks = final_state.get("parent_chunks", [])
        contexts = [c.text for c in parent_chunks] if parent_chunks else [""]
        answer = final_state.get("generated_answer", "")

        scores = compute_ragas_scores(question, answer, contexts, gt)
        ragas_obj = RAGASScores(
            faithfulness=scores["faithfulness"], context_recall=scores["context_recall"],
            context_precision=scores["context_precision"], answer_relevancy=scores["answer_relevancy"],
        )
        diagnosis = ragas_obj.diagnosis(high=cfg.evaluation.HIGH_THRESHOLD, low=cfg.evaluation.LOW_THRESHOLD)
        update_eval_scores(row_id, scores, diagnosis, diagnosis == "PASS")
        results.append({"query_id": q_id, "question": question, "difficulty": difficulty, "diagnosis": diagnosis, **scores})

        def _fmt(v): return f"{v:.2f}" if v is not None else "N/A"
        logger.info(f"  -> {diagnosis} | faith={_fmt(scores['faithfulness'])} | recall={_fmt(scores['context_recall'])} | "
                     f"prec={_fmt(scores['context_precision'])} | rel={_fmt(scores['answer_relevancy'])}")

    # Compute averages
    total = len(results)
    def _safe_avg(key):
        vals = [r[key] for r in results if r.get(key) is not None and not math.isnan(r.get(key))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    avgs = {m: _safe_avg(m) for m in ["faithfulness", "context_recall", "context_precision", "answer_relevancy"]}

    # Save results
    results_dir = cfg.ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "cycle_results.json"
    all_cycles = json.load(open(results_file, encoding="utf-8")) if results_file.exists() else {}
    all_cycles[f"cycle_{cycle_number}"] = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total, "pipeline_errors": pipeline_errors,
        "avg_faithfulness": avgs["faithfulness"], "avg_context_recall": avgs["context_recall"],
        "avg_context_precision": avgs["context_precision"], "avg_answer_relevancy": avgs["answer_relevancy"],
        "per_question": [
            {k: r.get(k) for k in ["query_id", "question", "difficulty", "faithfulness", "context_recall", "context_precision", "answer_relevancy"]}
            for r in results
        ],
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_cycles, f, indent=2, ensure_ascii=False)

    # Print report
    print(f"\n{'='*70}")
    print(f"  EVALUATION REPORT — Cycle {cycle_number}")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"  Total questions    : {total}")
    print(f"  Pipeline errors    : {pipeline_errors}\n")
    for name, key in [("Faithfulness", "faithfulness"), ("Context Recall", "context_recall"),
                      ("Context Precision", "context_precision"), ("Answer Relevancy", "answer_relevancy")]:
        print(f"  Avg {name:20s}: {avgs[key]:.4f}")
    diag_counts = Counter(r["diagnosis"] for r in results)
    if diag_counts:
        print("\n  Diagnosis Breakdown:")
        for diag, count in diag_counts.most_common():
            print(f"    {diag:<25} {count}")
    print(f"\n  Results saved to : {results_file}")
    print(f"  Traces saved to  : {cfg.storage.SQLITE_PATH}")
    print(f"{'='*70}")
    return avgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--set", type=str, default="dev", choices=["dev", "test"])
    args = parser.parse_args()
    eval_path = cfg.evaluation.DEV_SET_PATH if args.set == "dev" else cfg.evaluation.TEST_SET_PATH
    if not eval_path.exists():
        logger.error(f"Eval set not found at {eval_path}. Run generate_dataset.py first.")
        sys.exit(1)
    run_evaluation(eval_path, args.cycle)
