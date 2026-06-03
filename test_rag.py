"""Quick smoke test: runs two queries through the full RAG pipeline."""

import uuid
import logging
from pipeline.graph import rag_pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")

def test_query(query: str):
    print(f"\n{'='*80}\nQUERY: {query}\n{'='*80}")
    final_state = rag_pipeline.invoke({
        "query_id": str(uuid.uuid4()),
        "original_query": query,
        "cycle_number": 0,
        "planner_version": 1,
        "prompt_version": 1
    })
    plan = final_state.get('retrieval_plan')
    if plan:
        print(f"  Collections: {plan.collections} | Mode: {plan.search_mode}")
        print(f"  Filters: eCFR={plan.ecfr_filters}, Guidance={plan.guidance_filters}")
    print(f"  Retrieved {len(final_state.get('child_chunks', []))} children -> {len(final_state.get('parent_chunks', []))} parents")
    print(f"\nANSWER:\n{final_state.get('generated_answer')}\n{'='*80}")

if __name__ == "__main__":
    for q in [
        "What are the design control requirements for a Class II medical device?",
        "When do I need to submit a new 510(k) for a software change?",
    ]:
        test_query(q)
