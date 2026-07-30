"""Async OpenAI-compatible clients pointed at the org LiteLLM proxy.

Traffic path:

  LangGraph/FastAPI agent
        |  OpenAI-compatible request
        |  model="agent-brain" | model="domain-ft"
        v
  LiteLLM proxy  (LITELLM_BASE_URL, default http://localhost:4000)
        |
        +--> agent-brain  --> Qwen vLLM        (:8000)
        +--> domain-ft    --> fine-tuned Nemotron vLLM (:8001)
"""

from __future__ import annotations

from typing import Any

import httpx
from openai import AsyncOpenAI

from src.config import Settings


class LLMClients:
    """Shared AsyncOpenAI client; model name selects the LiteLLM alias."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        limits = httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=max(4, settings.http_max_connections // 2),
        )
        self._http = httpx.AsyncClient(
            limits=limits,
            timeout=settings.request_timeout_seconds,
        )
        # settings.litellm_base_url is normalized to .../v1
        self._client = AsyncOpenAI(
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_key or "EMPTY",
            timeout=settings.request_timeout_seconds,
            max_retries=2,
            http_client=self._http,
        )

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    async def aclose(self) -> None:
        await self._client.close()
        if not self._http.is_closed:
            await self._http.aclose()

    async def brain_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Call LiteLLM alias BRAIN_MODEL (default: agent-brain → Qwen)."""
        kwargs: dict[str, Any] = {
            "model": self.settings.brain_model,
            "messages": messages,
            "temperature": self.settings.temperature_brain,
            "max_tokens": self.settings.max_tokens_brain,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return await self._client.chat.completions.create(**kwargs)

    async def synthesize(self, prompt: str, *, system: str | None = None) -> str:
        """
        Always call DOMAIN_FT_MODEL (default: domain-ft) via LiteLLM.

        DOMAIN_PREDICT_MODE only selects the OpenAI API shape:
          - chat / llm: chat.completions
          - completion: completions
        """
        mode = self.settings.domain_predict_mode
        model = self.settings.domain_ft_model

        if mode == "completion":
            full = prompt if system is None else f"{system}\n\n{prompt}"
            resp = await self._client.completions.create(
                model=model,
                prompt=full,
                temperature=self.settings.temperature_synthesis,
                max_tokens=self.settings.max_tokens_synthesis,
            )
            return (resp.choices[0].text or "").strip()

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.settings.temperature_synthesis,
            max_tokens=self.settings.max_tokens_synthesis,
        )
        content = resp.choices[0].message.content
        return (content or "").strip()
