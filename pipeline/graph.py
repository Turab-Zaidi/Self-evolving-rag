"""LangGraph pipeline: Planner → Retriever → Generator."""

from langgraph.graph import StateGraph, END
from pipeline.state import RAGState
from pipeline.nodes.planner import query_planner_node
from pipeline.nodes.retriever import retrieval_node
from pipeline.nodes.generator import generate_node

def build_rag_graph() -> StateGraph:
    workflow = StateGraph(RAGState)
    workflow.add_node("planner", query_planner_node)
    workflow.add_node("retriever", retrieval_node)
    workflow.add_node("generator", generate_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)
    return workflow.compile()

rag_pipeline = build_rag_graph()
