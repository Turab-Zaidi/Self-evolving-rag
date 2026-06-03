"""LangGraph state definitions for the RAG pipeline."""

from typing import TypedDict, Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id:        str
    parent_id:       Optional[str]
    text:            str
    source_type:     str
    collection:      str
    metadata:        dict = field(default_factory=dict)
    relevance_score: float = 0.0


@dataclass
class RetrievalPlan:
    rewritten_query:  str
    collections:      list[str]
    search_mode:      str
    ecfr_filters:     Optional[dict]
    guidance_filters: Optional[dict]
    reasoning:        str


@dataclass
class RAGASScores:
    faithfulness:       Optional[float] = None
    context_recall:     Optional[float] = None
    context_precision:  Optional[float] = None
    answer_relevancy:   Optional[float] = None
    answer_correctness: Optional[float] = None

    def diagnosis(self, high: float = 0.7, low: float = 0.4) -> str:
        f  = self.faithfulness       or 0.0
        cr = self.context_recall     or 0.0
        cp = self.context_precision  or 0.0
        ar = self.answer_relevancy   or 0.0

        if all(s < low for s in [f, cr, cp, ar]):
            return "KNOWLEDGE_GAP"
        if cr < low and f > high:
            return "RETRIEVAL_GAP"
        if cr > high and f < low:
            return "GENERATION_FLAW"
        if cr > high and f > high and ar < low:
            return "REASONING_FAILURE"
        if cp < low:
            return "RERANKING_ERROR"
        return "PASS"


class RAGState(TypedDict):
    query_id:           str
    original_query:     str
    ground_truth:       Optional[str]
    cycle_number:       int
    retrieval_plan:     Optional[RetrievalPlan]
    child_chunks:       list[Chunk]
    parent_chunks:      list[Chunk]
    assembled_context:  str
    generated_answer:   str
    prompt_version:     int
    planner_version:    int
    ragas_scores:       Optional[RAGASScores]
    diagnosis:          Optional[str]
    fix_applied:        Optional[str]


class CorrectionState(TypedDict):
    cycle_number:        int
    failed_traces:       list[dict]
    passing_traces:      list[dict]
    dominant_failure:    Optional[str]
    pre_fix_accuracy:    float
    passing_accuracy:    float
    prompt_version:      int
    planner_version:     int
    new_collection:      Optional[str]
    fix_description:     str
    post_fix_accuracy:   float
    improved:            bool
    committed:           bool
    mlflow_run_id:       Optional[str]
