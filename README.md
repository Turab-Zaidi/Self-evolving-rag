# Self-Evolving RAG

A Retrieval-Augmented Generation (RAG) pipeline for FDA regulatory compliance that autonomously optimizes itself using DSPy, Optuna, and RAGAS.

The system answers questions about medical device regulations (21 CFR Title 21) by retrieving relevant passages from the eCFR database and FDA guidance documents, then generating precise, cited answers.

---

## What Makes It "Self-Evolving"

After each evaluation cycle, the pipeline runs a 3-stage optimization loop to improve its own performance:

| Stage | Tool | What It Optimizes |
|-------|------|--------------------|
| 1 — Query Planner | DSPy BootstrapFewShot | Rewrites user questions into better search queries |
| 2 — Hyperparameters | Optuna (Bayesian search) | BM25 weight and cross-encoder top-k |
| 3 — Generator | DSPy BootstrapFewShot | Answer quality and citation accuracy |

---

## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│   Planner   │  → rewrites query, selects collections, filters by CFR part
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retriever  │  → hybrid BM25 + ChromaDB search → RRF fusion → cross-encoder reranking
└──────┬──────┘       → parent-child chunk resolution
       │
       ▼
┌─────────────┐
│  Generator  │  → synthesizes answer with bracketed citations
└─────────────┘
       │
       ▼
  Final Answer
```

**Knowledge Base:**
- **eCFR**: 25 CFR Parts (801–892) fetched from the official eCFR API
- **FDA Guidance**: 5 key PDF guidance documents (Design Controls, Software Validation, 510(k), MDR, Cybersecurity)
- **Storage**: ChromaDB (dense) + BM25 (sparse) with parent-child chunking

---

## Evaluation Results

Each cycle runs RAGAS scoring across 40 test questions.

| Cycle | Description | Faithfulness | Context Recall | Context Precision | Answer Relevancy |
|-------|-------------|:---:|:---:|:---:|:---:|
| **0** | Baseline (no optimization) | 0.5753 | 0.7269 | 0.6508 | 0.5548 |
| **1** | DSPy + Optuna (unconstrained) | 0.4756 | 0.4432 | 0.4609 | 0.4122 |
| **2** | DSPy + Optuna (constrained) | 0.4662 | 0.5667 | 0.5364 | 0.4630 |
| **3** | Tuned prompt + golden hyperparams | **0.6862** | 0.6452 | 0.5376 | **0.5638** |

**Key finding:** Cycles 1 and 2 revealed a critical failure mode — unconstrained autonomous optimization can degrade performance. Optuna set BM25 weight to near-zero (disabling keyword search) and DSPy injected large few-shot examples that choked the model's context window. Cycle 3 applied guardrails: constrained search bounds, cleared noisy few-shot caches, and a tighter generation prompt — recovering and surpassing the baseline on Faithfulness (+11%) and Answer Relevancy (+0.9%).

---

## Project Structure

```
├── config.py                        # Central configuration
├── test_rag.py                      # Smoke test (run a question through the pipeline)
│
├── ingestion/
│   ├── indexer.py                   # Master ingestion script
│   ├── embedder.py                  # Local sentence-transformers embedder
│   ├── fetchers/                    # eCFR API + FDA PDF downloaders
│   └── parsers/                     # XML and PDF parsers with parent-child chunking
│
├── pipeline/
│   ├── graph.py                     # LangGraph orchestrator
│   ├── state.py                     # RAGState TypedDict
│   ├── tracer.py                    # SQLite trace logger
│   └── nodes/                       # Planner, Retriever, Generator nodes
│
├── correction/
│   ├── orchestrator.py              # Runs all 3 optimization stages in sequence
│   ├── optimize_planner.py          # Stage 1: DSPy query planner optimization
│   ├── optimize_hyperparams.py      # Stage 2: Optuna hyperparameter sweep
│   └── optimize_generator.py        # Stage 3: DSPy generator optimization
│
├── evaluation/
│   ├── run_eval.py                  # RAGAS evaluation runner
│   └── generate_dataset.py          # Synthetic Q&A dataset generator
│
├── prompts/                         # Planner and generator prompt templates
└── results/cycle_results.json       # Full evaluation history across all cycles
```

---

## Setup

```bash
git clone https://github.com/Turab-Zaidi/Self-evolving-rag.git
cd Self-evolving-rag
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # add your NVIDIA NIM API key
```

---

## Running

**1. Ingest data** (fetch eCFR + parse FDA PDFs):
```bash
python ingestion/indexer.py
```

**2. Generate evaluation dataset:**
```bash
python evaluation/generate_dataset.py
```

**3. Run baseline evaluation (Cycle 0):**
```bash
python evaluation/run_eval.py --cycle 0 --set test
```

**4. Run the self-evolution loop:**
```bash
python correction/orchestrator.py
```

**5. Evaluate the optimized pipeline:**
```bash
python evaluation/run_eval.py --cycle 1 --set test
```

**6. Smoke test a live query:**
```bash
python test_rag.py
```

---

## Tech Stack

- **LangGraph** — pipeline orchestration
- **ChromaDB** — vector store (dense search)
- **BM25** — sparse keyword search
- **sentence-transformers** — local embeddings (`all-MiniLM-L6-v2`)
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — reranking
- **DSPy** — automated prompt optimization via BootstrapFewShot
- **Optuna** — Bayesian hyperparameter search
- **RAGAS** — retrieval and generation quality metrics
- **NVIDIA NIM** — LLM inference (`gpt-oss-120b` via OpenAI-compatible API)

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | Your NVIDIA NIM API key |
| `NIM_BASE_URL` | API base URL (default: `https://integrate.api.nvidia.com/v1`) |
| `GENERATION_MODEL` | Model name for generation |
| `EVALUATION_MODEL` | Model name for RAGAS evaluation |
| `EMBEDDING_MODEL` | Local embedding model name |
