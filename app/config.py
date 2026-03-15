from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str

    # Model selection — kept as config so judge and generator can be swapped
    # independently for benchmarking
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o"
    judge_model: str = "gpt-4o"

    # Chunking — treated as tunable parameters, not constants
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k_retrieval: int = 6

    # Evaluation
    gap_score_threshold: int = 1  # scores <= this are flagged as gaps (0–3 scale)

    # Storage
    session_log_dir: Path = Path("./logs")
    chroma_persist_dir: Path = Path("./chroma_db")


settings = Settings()
