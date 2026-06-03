"""Stage 2: Hyperparameter optimization via Optuna (BM25 weight + cross-encoder top-k)."""

import os
import sqlite3
import itertools
import json
import optuna
import numpy as np
import dspy
from dotenv import load_dotenv
from config import cfg
from pipeline.nodes.retriever import retrieval_node
from ingestion.embedder import LocalEmbedder

load_dotenv()

ALL_KEYS = [k.strip() for k in os.getenv("NVIDIA_API_KEYS", "").split(",") if k.strip()]
BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
_key_cycle = itertools.cycle(ALL_KEYS)


def get_rotated_lm():
    return dspy.LM('openai/openai/gpt-oss-120b', api_base=BASE_URL, api_key=next(_key_cycle))

nim_llm = get_rotated_lm()
dspy.settings.configure(lm=nim_llm)
print(f"[Hyperparameter Optimizer] Loaded {len(ALL_KEYS)} API keys for rotation.")

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = LocalEmbedder()
    return _embedder


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


from correction.optimize_planner import QueryPlanner, PlanQuery

def load_optimized_planner():
    planner = QueryPlanner()
    planner_path = "correction/optimized_planner.json"
    if os.path.exists(planner_path):
        planner.load(planner_path)
        print(f"Loaded optimized planner from {planner_path}")
    else:
        print("WARNING: No optimized planner found. Using default planner.")
    return planner


def load_validation_set():
    conn = sqlite3.connect("logs/traces.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT original_query, ground_truth FROM traces WHERE cycle_number=0 ORDER BY RANDOM() LIMIT 60"
    ).fetchall()
    conn.close()
    return rows


def generate_new_queries(planner, validation_set):
    embedder = get_embedder()
    cached_queries = []
    for row in validation_set:
        try:
            result = planner(original_query=row["original_query"])
            gt_embedding = embedder.embed_batch([row["ground_truth"]])[0]
            cached_queries.append({
                "original_query": row["original_query"], "ground_truth": row["ground_truth"],
                "gt_embedding": gt_embedding, "rewritten_query": result.rewritten_query,
                "cfr_part": result.cfr_part if result.cfr_part != "None" else None
            })
            print(f"  Planner: '{row['original_query'][:50]}...' -> '{result.rewritten_query[:50]}...'")
        except Exception as e:
            gt_embedding = embedder.embed_batch([row["ground_truth"]])[0]
            cached_queries.append({
                "original_query": row["original_query"], "ground_truth": row["ground_truth"],
                "gt_embedding": gt_embedding, "rewritten_query": row["original_query"], "cfr_part": None
            })
            print(f"  Planner fallback for: '{row['original_query'][:50]}...' ({e})")
    return cached_queries


def create_objective(cached_queries):
    embedder = get_embedder()

    def objective(trial):
        bm25_weight = trial.suggest_float("bm25_weight", 0.3, 0.7)
        cross_encoder_top_k = trial.suggest_int("cross_encoder_top_k", 12, 15)
        cfg.retrieval.BM25_WEIGHT = bm25_weight
        cfg.retrieval.CHROMA_WEIGHT = 1.0 - bm25_weight
        cfg.retrieval.RERANK_TOP_K = cross_encoder_top_k
        total_score = 0.0

        for q in cached_queries:
            ecfr_filters = {"cfr_part": q["cfr_part"]} if q["cfr_part"] else {}
            state = {
                "original_query": q["original_query"],
                "retrieval_plan": type('Plan', (), {
                    'rewritten_query': q["rewritten_query"], 'search_mode': 'parallel',
                    'collections': ['ecfr_regulations', 'fda_guidance'],
                    'ecfr_filters': ecfr_filters, 'guidance_filters': {}
                })()
            }
            try:
                new_state = retrieval_node(state)
                retrieved_text = " ".join([c.text for c in new_state.get("parent_chunks", [])])
                if not retrieved_text.strip():
                    continue
                retrieved_embedding = embedder.embed_batch([retrieved_text[:1000]])[0]
                total_score += cosine_similarity(q["gt_embedding"], retrieved_embedding)
            except Exception:
                continue
        return total_score / len(cached_queries)

    return objective


def optimize_hyperparameters():
    print("=== Stage 2: Hyperparameter Optimization (with Trained Planner) ===")
    planner = load_optimized_planner()
    validation_set = load_validation_set()
    print(f"Loaded {len(validation_set)} validation questions.")

    print("\nGenerating new search queries using the trained Planner...")
    cached_queries = generate_new_queries(planner, validation_set)
    print(f"Cached {len(cached_queries)} queries. No more LLM calls from here.\n")

    study = optuna.create_study(direction="maximize")
    study.optimize(create_objective(cached_queries), n_trials=30, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nOptimal BM25_WEIGHT: {best_params['bm25_weight']:.3f}")
    print(f"Optimal CROSS_ENCODER_TOP_K: {best_params['cross_encoder_top_k']}")
    print(f"Best Semantic Retrieval Score: {study.best_value:.4f}")

    with open("correction/optimized_hyperparams.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Saved to correction/optimized_hyperparams.json")

if __name__ == "__main__":
    optimize_hyperparameters()
