from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_size_mb: int = 25
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 180
    top_k_dense: int = 8
    top_k_lexical: int = 8
    top_k_final: int = 5

    @property
    def data_dir(self) -> Path:
        import os
        custom = os.environ.get("DATA_DIR")
        if custom:
            return Path(custom)
        if os.environ.get("VERCEL"):
            return Path("/tmp/data")
        return PROJECT_ROOT / "data"

    @property
    def documents_dir(self) -> Path:
        import os
        custom = os.environ.get("DOCUMENTS_DIR")
        if custom:
            return Path(custom)
        bundled = PROJECT_ROOT / "data" / "documents"
        if bundled.exists():
            return bundled
        return self.data_dir / "documents"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"


settings = Settings()
