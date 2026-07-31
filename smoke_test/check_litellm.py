#!/usr/bin/env python3
"""Smoke-test LiteLLM aliases (Qwen / Nemotron).

Usage (from repo root):

  python smoke_test/check_litellm.py              # both models
  python smoke_test/check_litellm.py qwen         # agent-brain only
  python smoke_test/check_litellm.py nemotron     # domain-ft only
  python smoke_test/check_litellm.py list         # GET /v1/models

Env (loaded from repo .env if present):
  LITELLM_BASE_URL  default http://localhost:4000
  LITELLM_KEY
  BRAIN_MODEL       default agent-brain
  DOMAIN_FT_MODEL   default domain-ft
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

import os  # noqa: E402  — after dotenv load


def _base_url() -> str:
    raw = (os.getenv("LITELLM_BASE_URL") or "http://localhost:4000").rstrip("/")
    return raw if raw.endswith("/v1") else f"{raw}/v1"


def _client() -> OpenAI:
    return OpenAI(
        base_url=_base_url(),
        api_key=os.getenv("LITELLM_KEY") or "EMPTY",
        timeout=60.0,
        max_retries=0,
    )


def list_models(client: OpenAI) -> int:
    print(f"GET {_base_url()}/models")
    try:
        models = client.models.list()
    except Exception as exc:
        print(f"FAIL  list models: {exc}")
        return 1
    ids = sorted(m.id for m in models.data)
    if not ids:
        print("OK    proxy reachable, but no models returned")
        return 0
    for mid in ids:
        print(f"  - {mid}")
    print(f"OK    {len(ids)} model(s)")
    return 0


def ping_chat(client: OpenAI, *, label: str, model: str, prompt: str) -> int:
    print(f"\n=== {label} ({model}) ===")
    print(f"POST {_base_url()}/chat/completions")
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1
    elapsed = time.perf_counter() - t0
    content = (resp.choices[0].message.content or "").strip()
    preview = content.replace("\n", " ")[:120] or "<empty>"
    print(f"OK    {elapsed:.1f}s  reply: {preview}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test LiteLLM Qwen / Nemotron aliases")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all", "qwen", "brain", "nemotron", "domain", "list"],
        help="What to check (default: all)",
    )
    args = parser.parse_args(argv)

    brain = os.getenv("BRAIN_MODEL") or "agent-brain"
    domain = os.getenv("DOMAIN_FT_MODEL") or "domain-ft"

    print(f"LiteLLM base: {_base_url()}")
    print(f"brain_model:  {brain}")
    print(f"domain_ft:    {domain}")

    client = _client()
    failed = 0

    if args.target == "list":
        return list_models(client)

    if args.target in ("all", "qwen", "brain"):
        failed |= ping_chat(
            client,
            label="Qwen / agent-brain",
            model=brain,
            prompt="hi",
        )

    if args.target in ("all", "nemotron", "domain"):
        failed |= ping_chat(
            client,
            label="Nemotron / domain-ft",
            model=domain,
            prompt="hi",
        )

    print("\nPASS" if failed == 0 else "\nFAIL")
    return failed


if __name__ == "__main__":
    sys.exit(main())
