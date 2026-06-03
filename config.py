"""Central configuration for the Self-Evolving RAG system."""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent


class NIM:
    _raw_keys = os.getenv("NVIDIA_API_KEYS", os.getenv("NVIDIA_API_KEY", ""))
    API_KEYS      = [k.strip() for k in _raw_keys.split(",") if k.strip()]
    API_KEY       = API_KEYS[0] if API_KEYS else ""
    BASE_URL      = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    GENERATION    = os.getenv("GENERATION_MODEL", "openai/gpt-oss-120b")
    EVALUATION    = os.getenv("EVALUATION_MODEL", "openai/gpt-oss-120b")
    EMBEDDING     = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBED_DIM     = 384
    EMBED_BATCH   = 32
    MAX_TOKENS    = 2048
    TEMPERATURE   = 0.1


class DataSources:
    ECFR_API_BASE = "https://www.ecfr.gov/api/versioner/v1"
    ECFR_TITLE    = 21
    ECFR_PARTS    = [
        801, 803, 806, 807, 808, 809, 810, 812, 814,
        820, 821, 822, 860,
        870, 872, 874, 876, 878, 880, 882, 884, 886, 888, 890, 892
    ]
    ECFR_WITHHELD = [830]
    FDA_GUIDANCE_DIR = ROOT / "data" / "raw" / "fda_guidance"
    ECFR_RAW_DIR     = ROOT / "data" / "raw" / "ecfr"


class Chunking:
    ECFR_CHILD_MAX_TOKENS  = 200
    ECFR_PARENT_MAX_TOKENS = 600
    FDA_CHILD_MAX_TOKENS   = 250
    FDA_CHILD_OVERLAP      = 50
    FDA_PARENT_MAX_TOKENS  = 700


class Storage:
    CHROMA_DIR    = ROOT / os.getenv("CHROMA_PERSIST_DIR", "storage/chroma")
    BM25_DIR      = ROOT / os.getenv("BM25_INDEX_DIR", "storage/bm25")
    SQLITE_PATH   = ROOT / os.getenv("SQLITE_DB_PATH", "logs/traces.db")
    ECFR_COLLECTION     = "ecfr_regulations"
    GUIDANCE_COLLECTION = "fda_guidance"


class Retrieval:
    DENSE_TOP_K      = 20
    SPARSE_TOP_K     = 20
    PRE_RERANK_TOP_K = 15
    RERANK_TOP_K     = 7
    RRF_K            = 60
    RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Evaluation:
    DEV_SET_PATH   = ROOT / "evaluation" / "eval_set" / "dev_set.json"
    TEST_SET_PATH  = ROOT / "evaluation" / "eval_set" / "test_set.json"
    HIGH_THRESHOLD = 0.7
    LOW_THRESHOLD  = 0.4
    MIN_FAILURES_TO_FIX = 10


class Prompts:
    DIR = ROOT / os.getenv("PROMPTS_DIR", "prompts")

    @staticmethod
    def generation(version: int = 1) -> Path:
        return Prompts.DIR / f"generation_v{version}.txt"

    @staticmethod
    def planner(version: int = 1) -> Path:
        return Prompts.DIR / f"planner_v{version}.txt"


class Correction:
    MIN_ACCURACY_GAIN = 0.03
    MAX_REGRESSION    = 0.05
    REGRESSION_SAMPLE = 20
    FIX_PRIORITY = [
        "KNOWLEDGE_GAP", "RETRIEVAL_GAP", "GENERATION_FLAW",
        "REASONING_FAILURE", "RERANKING_ERROR",
    ]


class Config:
    ROOT       = ROOT
    nim        = NIM
    data       = DataSources
    chunking   = Chunking
    storage    = Storage
    retrieval  = Retrieval
    evaluation = Evaluation
    prompts    = Prompts
    correction = Correction

    def __init__(self):
        opt_path = ROOT / "correction" / "optimized_hyperparams.json"
        if opt_path.exists():
            try:
                with open(opt_path, "r") as f:
                    opt = json.load(f)
                    self.retrieval.BM25_WEIGHT = opt.get("bm25_weight", getattr(self.retrieval, "BM25_WEIGHT", 0.5))
                    self.retrieval.CHROMA_WEIGHT = 1.0 - self.retrieval.BM25_WEIGHT
                    self.retrieval.CROSS_ENCODER_TOP_K = opt.get("cross_encoder_top_k", self.retrieval.RERANK_TOP_K)
                    self.retrieval.RERANK_TOP_K = self.retrieval.CROSS_ENCODER_TOP_K
            except Exception as e:
                print(f"Warning: Failed to load optimized hyperparams: {e}")

cfg = Config()
