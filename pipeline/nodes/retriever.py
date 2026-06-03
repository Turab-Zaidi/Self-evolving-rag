"""Retriever node: hybrid search (ChromaDB + BM25), cross-encoder reranking, parent-child resolution."""

import pickle
import logging
from typing import List
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import chromadb
from config import cfg
from pipeline.state import RAGState, Chunk
from ingestion.embedder import LocalEmbedder

logger = logging.getLogger(__name__)

_chroma_client = None
_bm25_model = None
_bm25_chunks = None
_cross_encoder = None
_embedder = None


def _init_models():
    global _chroma_client, _bm25_model, _bm25_chunks, _cross_encoder, _embedder
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(cfg.storage.CHROMA_DIR))
    if _bm25_model is None:
        bm25_path = cfg.storage.BM25_DIR / "bm25_index.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                _bm25_model = pickle.load(f)
            with open(cfg.storage.BM25_DIR / "chunk_mapping.pkl", "rb") as f:
                _bm25_chunks = pickle.load(f)
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(model_name=cfg.retrieval.RERANKER_MODEL)
    if _embedder is None:
        _embedder = LocalEmbedder()


def _reciprocal_rank_fusion(dense_results: List[Chunk], sparse_results: List[Chunk], k: int = 60) -> List[Chunk]:
    scores = {}
    chunk_map = {}
    dense_weight = getattr(cfg.retrieval, "CHROMA_WEIGHT", 0.5)
    for rank, chunk in enumerate(dense_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + dense_weight * (1.0 / (k + rank + 1))
        chunk_map[chunk.chunk_id] = chunk
    sparse_weight = getattr(cfg.retrieval, "BM25_WEIGHT", 0.5)
    for rank, chunk in enumerate(sparse_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + sparse_weight * (1.0 / (k + rank + 1))
        chunk_map[chunk.chunk_id] = chunk
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids]


def hybrid_search(query: str, collection_name: str, filters: dict = None) -> List[Chunk]:
    if _bm25_model is None or _chroma_client is None:
        return []

    # Sparse search (BM25)
    tokenized_query = query.lower().split(" ")
    sparse_candidates = _bm25_model.get_top_n(tokenized_query, _bm25_chunks, n=cfg.retrieval.SPARSE_TOP_K * 2)
    sparse_results = []
    for c in sparse_candidates:
        if c.collection != collection_name:
            continue
        if filters:
            if not all(v is None or c.metadata.get(k) == v for k, v in filters.items()):
                continue
        sparse_results.append(c)
        if len(sparse_results) >= cfg.retrieval.SPARSE_TOP_K:
            break

    # Dense search (ChromaDB)
    collection = _chroma_client.get_collection(collection_name)
    query_embedding = _embedder.embed_batch([query])[0]
    chroma_filters = None
    if filters:
        valid_pairs = [(k, v) for k, v in filters.items() if v is not None]
        if len(valid_pairs) == 1:
            chroma_filters = {valid_pairs[0][0]: valid_pairs[0][1]}
        elif len(valid_pairs) > 1:
            chroma_filters = {"$and": [{k: v} for k, v in valid_pairs]}

    dense_res = collection.query(query_embeddings=[query_embedding], n_results=cfg.retrieval.DENSE_TOP_K, where=chroma_filters)
    dense_results = []
    if dense_res['ids'] and dense_res['ids'][0]:
        for i in range(len(dense_res['ids'][0])):
            dense_results.append(Chunk(
                chunk_id=dense_res['ids'][0][i],
                parent_id=None,
                text=dense_res['documents'][0][i],
                source_type=dense_res['metadatas'][0][i].get("source_type", ""),
                collection=collection_name,
                metadata=dense_res['metadatas'][0][i]
            ))

    return _reciprocal_rank_fusion(dense_results, sparse_results, k=cfg.retrieval.RRF_K)[:cfg.retrieval.PRE_RERANK_TOP_K]


def retrieval_node(state: RAGState) -> dict:
    _init_models()
    plan = state["retrieval_plan"]
    query = plan.rewritten_query
    all_candidates = []

    if "ecfr_regulations" in plan.collections:
        logger.info(f"Retriever: Searching eCFR with filters: {plan.ecfr_filters}")
        all_candidates.extend(hybrid_search(query, cfg.storage.ECFR_COLLECTION, plan.ecfr_filters))
    if "fda_guidance" in plan.collections:
        logger.info(f"Retriever: Searching Guidance with filters: {plan.guidance_filters}")
        all_candidates.extend(hybrid_search(query, cfg.storage.GUIDANCE_COLLECTION, plan.guidance_filters))

    if not all_candidates:
        logger.warning("Retriever: No documents found.")
        return {"child_chunks": [], "parent_chunks": []}

    # Rerank with cross-encoder
    logger.info(f"Retriever: Reranking {len(all_candidates)} candidates...")
    pairs = [[query, chunk.text] for chunk in all_candidates]
    scores = _cross_encoder.score(pairs)
    for chunk, score in zip(all_candidates, scores):
        chunk.relevance_score = float(score)
    all_candidates.sort(key=lambda x: x.relevance_score, reverse=True)
    top_children = all_candidates[:cfg.retrieval.RERANK_TOP_K]

    # Parent-child resolution
    parent_chunks = {}
    for child in top_children:
        chunk_type = child.metadata.get("chunk_type")
        if chunk_type == "paragraph":
            parent_id = child.chunk_id.rsplit("-p", 1)[0]
            if parent_id not in parent_chunks:
                collection = _chroma_client.get_collection(child.collection)
                parent_res = collection.get(ids=[parent_id])
                if parent_res and parent_res['ids']:
                    parent_chunks[parent_id] = Chunk(
                        chunk_id=parent_res['ids'][0], parent_id=None,
                        text=parent_res['documents'][0],
                        source_type=parent_res['metadatas'][0].get("source_type", ""),
                        collection=child.collection, metadata=parent_res['metadatas'][0]
                    )
        elif chunk_type == "section":
            parent_chunks[child.chunk_id] = child

    logger.info(f"Retriever: Returning {len(top_children)} children mapped to {len(parent_chunks)} parents.")
    return {"child_chunks": top_children, "parent_chunks": list(parent_chunks.values())}
