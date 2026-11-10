"""Runtime configuration, read from environment variables.

Two modes are supported:

* ``offline`` (default) - deterministic, dependency-light stand-ins replace the
  heavy models so the full pipeline runs on any machine without a GPU.
* ``real`` - BGE-M3 embeddings, a cross-encoder re-ranker, a biomedical image
  encoder and a vision-language model (Qwen2.5-VL served by vLLM through an
  OpenAI-compatible endpoint) are used for genuine reasoning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    mode: str = field(default_factory=lambda: _env("MEDRAG_MODE", "offline"))
    data_dir: Path = field(default_factory=lambda: Path(_env("MEDRAG_DATA_DIR", str(ROOT_DIR / "var"))))
    kb_dir: Path = field(
        default_factory=lambda: Path(_env("MEDRAG_KB_DIR", str(ROOT_DIR / "data" / "knowledge_base")))
    )
    secret_key: str = field(default_factory=lambda: _env("MEDRAG_SECRET_KEY", "dev-secret-change-me"))
    patient_salt: str = field(default_factory=lambda: _env("MEDRAG_PATIENT_SALT", "dev-salt-change-me"))
    token_ttl_seconds: int = field(default_factory=lambda: int(_env("MEDRAG_TOKEN_TTL", "28800")))

    # Retrieval
    chunk_tokens: int = field(default_factory=lambda: int(_env("MEDRAG_CHUNK_TOKENS", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("MEDRAG_CHUNK_OVERLAP", "60")))
    top_k: int = field(default_factory=lambda: int(_env("MEDRAG_TOP_K", "5")))
    candidate_k: int = field(default_factory=lambda: int(_env("MEDRAG_CANDIDATE_K", "20")))
    rrf_k: int = field(default_factory=lambda: int(_env("MEDRAG_RRF_K", "60")))
    vector_backend: str = field(default_factory=lambda: _env("MEDRAG_VECTOR_BACKEND", "memory"))
    qdrant_url: str = field(default_factory=lambda: _env("MEDRAG_QDRANT_URL", "http://localhost:6333"))

    # Agent
    max_iterations: int = field(default_factory=lambda: int(_env("MEDRAG_MAX_ITERATIONS", "3")))
    support_threshold: float = field(default_factory=lambda: float(_env("MEDRAG_SUPPORT_THRESHOLD", "0.5")))

    # Models (real mode)
    embed_model: str = field(default_factory=lambda: _env("MEDRAG_EMBED_MODEL", "BAAI/bge-m3"))
    rerank_model: str = field(default_factory=lambda: _env("MEDRAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    image_encoder: str = field(
        default_factory=lambda: _env(
            "MEDRAG_IMAGE_ENCODER", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
    )
    llm_url: str = field(default_factory=lambda: _env("MEDRAG_LLM_URL", "http://localhost:8001/v1"))
    llm_model: str = field(default_factory=lambda: _env("MEDRAG_LLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    llm_api_key: str = field(default_factory=lambda: _env("MEDRAG_LLM_API_KEY", "EMPTY"))
    llm_timeout: float = field(default_factory=lambda: float(_env("MEDRAG_LLM_TIMEOUT", "120")))

    # Seed demo users on first start (development only)
    seed_demo: bool = field(default_factory=lambda: _env("MEDRAG_SEED_DEMO", "1") == "1")

    @property
    def offline(self) -> bool:
        return self.mode != "real"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "medrag.sqlite3"

    @property
    def object_dir(self) -> Path:
        return self.data_dir / "objects"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
