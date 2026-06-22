import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    openai_embed_model: str = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chroma_dir: str = os.environ.get("CHROMA_DIR", "./data/chroma")
    collection_name: str = os.environ.get("EIH_COLLECTION", "airflow_w2")


cfg = Config()
