from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    """
    Return a cached OpenAIEmbeddings instance.

    Cached so that repeated calls within a process don't re-initialize
    the client on every ingestion step. The cache is intentionally
    maxsize=1 — if the model string changes (e.g. in tests), the cache
    should be cleared explicitly via get_embedding_model.cache_clear().
    """
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
