"""Generator node: synthesizes the final answer from retrieved context."""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.state import RAGState
from pipeline.llm_utils import get_nim_llm
from config import cfg

logger = logging.getLogger(__name__)

def generate_node(state: RAGState) -> dict:
    query = state.get("original_query", "")
    parents = state.get("parent_chunks", [])
    gen_version = state.get("generator_version", 1)

    if not parents:
        logger.warning("Generator: No context provided.")
        context_str = "No regulatory context found."
    else:
        context_blocks = [f"[{p.collection.upper()} | {p.chunk_id}]\n{p.text}" for p in parents]
        context_str = "\n\n---\n\n".join(context_blocks)

    logger.info(f"Generator: Synthesizing answer using {len(parents)} source sections...")

    prompt_path = cfg.prompts.generation(gen_version)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")

    system_prompt = prompt_path.read_text(encoding="utf-8")
    formatted_system = system_prompt.replace("{context}", context_str)

    # Inject DSPy few-shot demos if available
    opt_path = cfg.ROOT / "correction" / "optimized_generator.json"
    if opt_path.exists():
        try:
            with open(opt_path, "r", encoding="utf-8") as f:
                opt = json.load(f)
            demos = opt.get("generate.predict", {}).get("demos", [])
            if demos:
                few_shot_str = "\n\n=== EXAMPLES OF HIGH-QUALITY ANSWERS ===\n"
                for i, demo in enumerate(demos):
                    few_shot_str += f"\nExample {i+1}:\n"
                    few_shot_str += f"Query: {demo.get('original_query', '')}\n"
                    few_shot_str += f"Reasoning: {demo.get('reasoning', '')}\n"
                    few_shot_str += f"Answer: {demo.get('generated_answer', '')}\n"
                formatted_system += few_shot_str
        except Exception as e:
            logger.warning(f"Failed to inject optimized generator demos: {e}")

    llm = get_nim_llm(temperature=0.3)

    try:
        response = llm.invoke([SystemMessage(content=formatted_system), HumanMessage(content=query)])
        return {"generated_answer": response.content}
    except Exception as e:
        logger.error(f"Generator failed: {e}")
        return {"generated_answer": f"Error generating answer: {e}"}
