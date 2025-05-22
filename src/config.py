# src/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MAX_RETRIES_PER_CHUNK: int = 1  # 청크당 최대 재시도 횟수
    GEMINI_API_KEY: str
    VECTOR_API_URL: str
    CHUNK_SIZE: int = 512          # 토큰 기준 상한
    CHUNK_OVERLAP: int = 64
    MAX_RETRY: int = 2             # 재적합 루프
    VECTOR_TOP_K: int = 5
    RELEVANT_K: int = 3
    SCORE_THRESHOLD: float = 0.5
    TIMEOUT_SEC: int = 300         # 전 파이프라인 5 분

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

@lru_cache
def get_settings() -> Settings:
    return Settings()
