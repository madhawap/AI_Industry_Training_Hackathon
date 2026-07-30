"""Application configuration loaded from config.yaml with env-var expansion."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _expand_env(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} placeholders."""

    def repl(match: re.Match[str]) -> str:
        var, default = match.group(1), match.group(2)
        env_val = os.environ.get(var)
        if env_val is not None:
            return env_val
        return default if default is not None else ""

    return _ENV_PATTERN.sub(repl, value)


def _expand_tree(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _expand_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_tree(v) for v in node]
    if isinstance(node, str):
        return _expand_env(node)
    return node


class Settings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001

    litellm_base_url: str = Field(alias="litellm_base_url")
    litellm_key: str = ""

    brain_model: str = "agent-brain"
    max_agent_steps: int = 5

    domain_ft_model: str = "domain-ft"
    domain_predict_mode: str = "chat"

    request_timeout_seconds: float = 60.0
    temperature_brain: float = 0.1
    temperature_synthesis: float = 0.2
    max_tokens_brain: int = 4096
    max_tokens_synthesis: int = 8192

    max_concurrent_queries: int = 16
    http_max_connections: int = 32

    default_tool_timeout_seconds: float = 15.0

    @field_validator("max_agent_steps", "port", "max_concurrent_queries", mode="before")
    @classmethod
    def _coerce_int(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip():
            return int(v)
        return v

    @field_validator(
        "request_timeout_seconds",
        "temperature_brain",
        "temperature_synthesis",
        "default_tool_timeout_seconds",
        mode="before",
    )
    @classmethod
    def _coerce_float(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip():
            return float(v)
        return v

    @field_validator("domain_predict_mode")
    @classmethod
    def _normalize_mode(cls, v: str) -> str:
        mode = (v or "chat").strip().lower()
        # Always calls real domain-ft via LiteLLM; this only selects the API shape.
        # "llm" is accepted as an alias for chat.
        aliases = {
            "chat": "chat",
            "llm": "chat",
            "completion": "completion",
        }
        if mode not in aliases:
            raise ValueError(
                "domain_predict_mode must be one of: chat, llm, completion"
            )
        return aliases[mode]

    @field_validator("litellm_base_url")
    @classmethod
    def _normalize_litellm_url(cls, v: str) -> str:
        """Ensure OpenAI client hits LiteLLM's /v1 surface."""
        url = (v or "http://localhost:4000").rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        return url

    model_config = {"populate_by_name": True}


def _default_config_path() -> Path:
    # Prefer project-root config.yaml; fall back to env override.
    override = os.environ.get("APP_CONFIG")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    path = Path(config_path) if config_path else _default_config_path()
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config root must be a mapping: {path}")
            raw = _expand_tree(loaded)
    # Env vars win over file for the critical LiteLLM knobs.
    env_overrides = {
        "litellm_base_url": os.environ.get("LITELLM_BASE_URL"),
        "litellm_key": os.environ.get("LITELLM_KEY"),
        "brain_model": os.environ.get("BRAIN_MODEL"),
        "max_agent_steps": os.environ.get("MAX_AGENT_STEPS"),
        "domain_ft_model": os.environ.get("DOMAIN_FT_MODEL"),
        "domain_predict_mode": os.environ.get("DOMAIN_PREDICT_MODE"),
    }
    for key, value in env_overrides.items():
        if value is not None and value != "":
            raw[key] = value
    return Settings(**raw)
