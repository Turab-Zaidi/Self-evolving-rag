"""Local sentence-transformers embedder."""

import logging
from sentence_transformers import SentenceTransformer
from config import cfg

logger = logging.getLogger(__name__)


class LocalEmbedder:
    def __init__(self):
        logger.info(f"Loading local embedding model: {cfg.nim.EMBEDDING}")
        self.model = SentenceTransformer(cfg.nim.EMBEDDING)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
