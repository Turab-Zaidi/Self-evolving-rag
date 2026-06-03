"""Master ingestion: fetch, parse, embed, and index into ChromaDB + BM25."""

import sys
import logging
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from rank_bm25 import BM25Okapi
import chromadb
from config import cfg
from ingestion.fetchers.ecfr_fetcher import fetch_ecfr_part
from ingestion.parsers.ecfr_parser import parse_ecfr_xml
from ingestion.fetchers.fda_fetcher import FDA_GUIDANCE_DOCS
from ingestion.parsers.fda_parser import parse_fda_pdf
from ingestion.embedder import LocalEmbedder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion():
    embedder = LocalEmbedder()
    cfg.storage.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(cfg.storage.CHROMA_DIR))

    try:
        ecfr_collection = chroma_client.get_or_create_collection(name=cfg.storage.ECFR_COLLECTION, metadata={"hnsw:space": "cosine"})
        guidance_collection = chroma_client.get_or_create_collection(name=cfg.storage.GUIDANCE_COLLECTION, metadata={"hnsw:space": "cosine"})
    except Exception as e:
        logger.error(f"Failed to initialize Chroma collections: {e}")
        return

    all_chunks = []

    # Ingest eCFR
    for part in cfg.data.ECFR_PARTS:
        if part in cfg.data.ECFR_WITHHELD:
            continue
        xml_content = fetch_ecfr_part(part)
        if not xml_content:
            continue
        chunks = parse_ecfr_xml(xml_content, part)
        if not chunks:
            continue
        all_chunks.extend(chunks)
        embeddings = embedder.embed_batch([c.text for c in chunks])
        ecfr_collection.upsert(
            ids=[c.chunk_id for c in chunks], embeddings=embeddings,
            documents=[c.text for c in chunks], metadatas=[c.metadata for c in chunks]
        )

    # Ingest FDA Guidance PDFs
    for pdf_path in cfg.data.FDA_GUIDANCE_DIR.glob("*.pdf"):
        doc_id = pdf_path.stem
        doc_title = doc_id.replace("_", " ").replace("-", " ").title()
        chunks = parse_fda_pdf(pdf_path, doc_id, doc_title)
        if not chunks:
            continue
        all_chunks.extend(chunks)
        embeddings = embedder.embed_batch([c.text for c in chunks])
        guidance_collection.upsert(
            ids=[c.chunk_id for c in chunks], embeddings=embeddings,
            documents=[c.text for c in chunks], metadatas=[c.metadata for c in chunks]
        )

    # Build BM25 index
    cfg.storage.BM25_DIR.mkdir(parents=True, exist_ok=True)
    tokenized_corpus = [c.text.lower().split(" ") for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(cfg.storage.BM25_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(cfg.storage.BM25_DIR / "chunk_mapping.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    logger.info(f"Ingestion complete! Total chunks indexed: {len(all_chunks)}")

if __name__ == "__main__":
    run_ingestion()
