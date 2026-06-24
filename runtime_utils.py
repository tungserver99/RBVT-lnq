from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = ROOT / ".env"


def load_runtime_env(env_path: str | Path | None = None):
    env_file = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    if not env_file.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
        return
    except ImportError:
        pass

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def resolve_hf_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )
