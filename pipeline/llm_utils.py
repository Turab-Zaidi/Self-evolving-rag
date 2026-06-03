"""NVIDIA NIM LLM wrapper with multi-key failover."""

from langchain_openai import ChatOpenAI
from config import cfg
import logging

logger = logging.getLogger(__name__)

def get_nim_llm(temperature: float = 0.0):
    if not cfg.nim.API_KEYS:
        raise ValueError("No NVIDIA API keys found in environment.")

    primary_llm = ChatOpenAI(
        model=cfg.nim.GENERATION,
        api_key=cfg.nim.API_KEYS[0],
        base_url=cfg.nim.BASE_URL,
        temperature=temperature
    )

    if len(cfg.nim.API_KEYS) == 1:
        return primary_llm

    fallbacks = [
        ChatOpenAI(model=cfg.nim.GENERATION, api_key=key, base_url=cfg.nim.BASE_URL, temperature=temperature)
        for key in cfg.nim.API_KEYS[1:]
    ]
    logger.info(f"LLM initialized with {len(fallbacks) + 1} API keys for rotation/failover.")
    return primary_llm.with_fallbacks(fallbacks)
