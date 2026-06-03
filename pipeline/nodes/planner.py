"""Planner node: analyzes query and builds a structured retrieval plan."""

import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from pipeline.state import RAGState, RetrievalPlan
from pipeline.llm_utils import get_nim_llm
from config import cfg

logger = logging.getLogger(__name__)


class eCFRFilters(BaseModel):
    cfr_part: Optional[int] = Field(None, description="The specific CFR Part number to filter by (e.g., 820).")


class GuidanceFilters(BaseModel):
    document_id: Optional[str] = Field(None, description="The specific FDA guidance document ID to filter by.")


class PlannerOutput(BaseModel):
    rewritten_query: str
    collections: List[str]
    search_mode: str
    ecfr_filters: Optional[eCFRFilters] = None
    guidance_filters: Optional[GuidanceFilters] = None
    reasoning: str


def query_planner_node(state: RAGState) -> dict:
    query = state.get("original_query", "")
    planner_version = state.get("planner_version", 1)
    logger.info(f"QueryPlanner: Analyzing query -> '{query}'")

    prompt_path = cfg.prompts.planner(planner_version)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")

    system_prompt = prompt_path.read_text(encoding="utf-8")
    llm = get_nim_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(PlannerOutput)
    formatted_prompt = system_prompt.replace("{query}", query)

    # Inject DSPy few-shot demos if available
    opt_path = cfg.ROOT / "correction" / "optimized_planner.json"
    if opt_path.exists():
        try:
            with open(opt_path, "r", encoding="utf-8") as f:
                opt = json.load(f)
            demos = opt.get("plan.predict", {}).get("demos", [])
            if demos:
                few_shot_str = "\n\n=== EXAMPLES OF GOOD PLANS ===\n"
                for i, demo in enumerate(demos):
                    few_shot_str += f"\nExample {i+1}:\n"
                    few_shot_str += f"Original Query: {demo.get('original_query', '')}\n"
                    few_shot_str += f"Rewritten Query: {demo.get('rewritten_query', '')}\n"
                    few_shot_str += f"CFR Part: {demo.get('cfr_part', 'None')}\n"
                formatted_prompt += few_shot_str
        except Exception as e:
            logger.warning(f"Failed to inject optimized planner demos: {e}")

    try:
        result: PlannerOutput = structured_llm.invoke([SystemMessage(content=formatted_prompt)])
        plan = RetrievalPlan(
            rewritten_query=result.rewritten_query,
            collections=result.collections,
            search_mode=result.search_mode,
            ecfr_filters=result.ecfr_filters.model_dump(exclude_none=True) if result.ecfr_filters else None,
            guidance_filters=result.guidance_filters.model_dump(exclude_none=True) if result.guidance_filters else None,
            reasoning=result.reasoning
        )
        logger.info(f"QueryPlanner: Routing to {plan.collections} via {plan.search_mode} search.")
        logger.info(f"QueryPlanner: Reasoning: {plan.reasoning}")
        return {"retrieval_plan": plan, "planner_version": planner_version}
    except Exception as e:
        logger.error(f"QueryPlanner failed: {e}")
        fallback_plan = RetrievalPlan(
            rewritten_query=query,
            collections=[cfg.storage.ECFR_COLLECTION, cfg.storage.GUIDANCE_COLLECTION],
            search_mode="parallel",
            ecfr_filters=None, guidance_filters=None,
            reasoning="Fallback due to LLM parsing error."
        )
        return {"retrieval_plan": fallback_plan, "planner_version": planner_version}
