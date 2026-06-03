"""
evaluation/generate_dataset.py
Generates a synthetic evaluation dataset by:
  1. Sampling random parent chunks from both ChromaDB collections
  2. Using gpt-oss-120b to generate realistic Q&A pairs per chunk
  3. Splitting into dev_set.json (80 Qs) and test_set.json (40 Qs)

Run: venv\Scripts\python evaluation\generate_dataset.py
"""

import json
import random
import uuid
import logging
import sys
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import chromadb

# ── Bootstrap path so we can import config ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg
from pipeline.llm_utils import get_nim_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_QUESTIONS   = 120
DEV_SIZE          = 80
TEST_SIZE         = 40
CHUNKS_TO_SAMPLE  = 130   # sample more than needed in case some fail
ECFR_RATIO        = 0.70  # 70% from eCFR, 30% from FDA guidance

# Difficulty distribution
DIFFICULTY_DIST = {
    "basic":        int(TOTAL_QUESTIONS * 0.40),   # 48
    "intermediate": int(TOTAL_QUESTIONS * 0.40),   # 48
    "advanced":     int(TOTAL_QUESTIONS * 0.20),   # 24
}

GENERATION_PROMPT = """You are an expert FDA regulatory compliance trainer.

You will be given a section of US FDA regulatory text. Your task is to generate ONE precise, realistic question that a medical device compliance officer would ask, and the ground truth answer based STRICTLY on the provided text.

Difficulty level: {difficulty}

Guidelines per difficulty:
- basic: Single fact from the text. e.g. "What is the reporting deadline for X?"
- intermediate: Requires connecting 2+ sentences or understanding a condition. e.g. "Under what conditions must a manufacturer do X?"
- advanced: Requires understanding the regulatory intent and cross-referencing. e.g. "How does requirement X relate to obligation Y?"

RULES:
- The answer must be fully supported by the provided text. Do not hallucinate.
- The question must be answerable using ONLY the provided text.
- Be specific — avoid vague questions like "What does this section say?"
- Return ONLY valid JSON. No explanation. No markdown.

Text:
{chunk_text}

Return exactly this JSON structure:
{{
  "question": "...",
  "ground_truth": "...",
  "difficulty": "{difficulty}"
}}"""


def get_random_chunks(client: chromadb.Client, collection_name: str, n: int) -> list[dict]:
    """Pull n random parent-only chunks from a ChromaDB collection."""
    try:
        collection = client.get_collection(collection_name)
        # Get total count
        total = collection.count()
        if total == 0:
            logger.warning(f"Collection {collection_name} is empty.")
            return []

        # Fetch a larger batch and randomly sample from it
        fetch_n = min(total, n * 5)
        results = collection.get(
            limit=fetch_n,
            where={"chunk_type": "section"},   # only parent chunks
            include=["documents", "metadatas"]
        )

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        ids = results.get("ids", [])

        combined = list(zip(ids, docs, metas))
        sampled = random.sample(combined, min(n, len(combined)))

        return [{"id": i, "text": d, "metadata": m} for i, d, m in sampled]
    except Exception as e:
        logger.error(f"Error fetching from {collection_name}: {e}")
        return []


def generate_qa_pair(llm, chunk: dict, difficulty: str) -> dict | None:
    """Ask the LLM to generate a Q&A pair from a chunk."""
    text = chunk["text"]
    if len(text.split()) < 30:
        return None   # too short to generate a good question

    prompt = GENERATION_PROMPT.format(
        difficulty=difficulty,
        chunk_text=text[:3000]   # cap to avoid token overflow
    )

    try:
        response = llm.invoke([
            SystemMessage(content="You are a precise regulatory exam writer. Output only valid JSON."),
            HumanMessage(content=prompt)
        ])
        raw = response.content.strip()

        # Strip markdown fences if LLM wraps in ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        return {
            "query_id": f"{difficulty[:3]}_{str(uuid.uuid4())[:8]}",
            "question": parsed["question"],
            "ground_truth": parsed["ground_truth"],
            "difficulty": difficulty,
            "source_chunk_id": chunk["id"],
            "source_collection": chunk["metadata"].get("source_type", "unknown"),
            "cfr_part": chunk["metadata"].get("cfr_part", None),
        }
    except Exception as e:
        logger.warning(f"Failed to parse LLM output for chunk {chunk['id']}: {e}")
        return None


def main():
    logger.info("=== Starting Synthetic Dataset Generation ===")

    # 1. Init ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(cfg.storage.CHROMA_DIR))

    # 2. Compute sample sizes per collection
    ecfr_n    = int(CHUNKS_TO_SAMPLE * ECFR_RATIO)
    guidance_n = CHUNKS_TO_SAMPLE - ecfr_n

    logger.info(f"Sampling {ecfr_n} eCFR chunks + {guidance_n} FDA guidance chunks...")

    ecfr_chunks    = get_random_chunks(chroma_client, cfg.storage.ECFR_COLLECTION, ecfr_n)
    guidance_chunks = get_random_chunks(chroma_client, cfg.storage.GUIDANCE_COLLECTION, guidance_n)
    all_chunks = ecfr_chunks + guidance_chunks

    random.shuffle(all_chunks)
    logger.info(f"Total chunks available: {len(all_chunks)}")

    # 3. Init LLM
    llm = get_nim_llm(temperature=0.7)   # slightly higher temp for question diversity

    # 4. Generate Q&A pairs across difficulty levels
    all_questions = []
    difficulty_queue = []
    for diff, count in DIFFICULTY_DIST.items():
        difficulty_queue.extend([diff] * count)
    random.shuffle(difficulty_queue)

    chunk_idx = 0
    for i, difficulty in enumerate(difficulty_queue):
        if chunk_idx >= len(all_chunks):
            logger.warning("Ran out of chunks before generating all questions.")
            break

        chunk = all_chunks[chunk_idx]
        chunk_idx += 1

        logger.info(f"[{i+1}/{TOTAL_QUESTIONS}] Generating '{difficulty}' question from {chunk['id'][:50]}...")
        qa = generate_qa_pair(llm, chunk, difficulty)

        if qa:
            all_questions.append(qa)
        else:
            # Try next chunk
            if chunk_idx < len(all_chunks):
                chunk = all_chunks[chunk_idx]
                chunk_idx += 1
                qa = generate_qa_pair(llm, chunk, difficulty)
                if qa:
                    all_questions.append(qa)

    logger.info(f"Successfully generated {len(all_questions)} questions.")

    # 5. Split into dev / test sets
    random.shuffle(all_questions)
    dev_set  = all_questions[:DEV_SIZE]
    test_set = all_questions[DEV_SIZE:DEV_SIZE + TEST_SIZE]

    # 6. Save to disk
    out_dir = cfg.evaluation.DEV_SET_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_path  = cfg.evaluation.DEV_SET_PATH
    test_path = cfg.evaluation.TEST_SET_PATH

    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_set, f, indent=2, ensure_ascii=False)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    logger.info(f"Dev set  ({len(dev_set)} questions) → {dev_path}")
    logger.info(f"Test set ({len(test_set)} questions) → {test_path}")

    # 7. Print a sample
    logger.info("\n=== Sample Questions ===")
    for q in random.sample(dev_set, min(3, len(dev_set))):
        print(f"\n[{q['difficulty'].upper()}] {q['question']}")
        print(f"  ↳ {q['ground_truth'][:120]}...")

    logger.info("\n=== Dataset Generation Complete ===")


if __name__ == "__main__":
    main()
