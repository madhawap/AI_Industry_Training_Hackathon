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

from typing import Any, NoReturn

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError

from src.config import Settings


class UpstreamModelError(RuntimeError):
    """LiteLLM alias / upstream vLLM is unreachable or returned a fatal error."""

    def __init__(self, model: str, *, role: str, cause: BaseException) -> None:
        self.model = model
        self.role = role
        self.cause = cause
        tip = (
            "Check that LiteLLM is up and its route for this alias points at a "
            "running vLLM (for domain-ft: fine-tuning node :8001)."
        )
        super().__init__(
            f"{role} model '{model}' is not reachable via LiteLLM. {tip}"
        )


def _is_unreachable(exc: BaseException) -> bool:
    """True when LiteLLM/proxy cannot reach the upstream model backend."""
    if isinstance(exc, (APIConnectionError, httpx.ConnectError, httpx.TimeoutException)):
        return True
    text = str(exc).lower()
    markers = (
        "connection error",
        "connecterror",
        "failed to connect",
        "connection refused",
        "name or service not known",
        "nodename nor servname",
        "timed out",
        "timeout",
    )
    if any(m in text for m in markers):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in {500, 502, 503, 504}:
        # LiteLLM wraps upstream connection failures as 500 InternalServerError.
        return any(m in text for m in markers) or "model group" in text
    return False


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

    def _extra_body(self) -> dict[str, Any]:
        """Vendor options passed straight through to vLLM.

        Suppresses Qwen3's reasoning trace unless explicitly enabled. Measured
        on a two-operation plan: 83s with thinking on, 5.4s off, identical
        output. The tool catalogue already constrains what the planner may
        emit, so the reasoning tokens buy nothing and the latency is the
        difference between scoring and timing out.
        """
        if self.settings.enable_thinking:
            return {}
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def _reraise_unreachable(
        self, exc: BaseException, *, model: str, role: str
    ) -> NoReturn:
        if _is_unreachable(exc):
            raise UpstreamModelError(model, role=role, cause=exc) from exc
        raise exc

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
        model = self.settings.brain_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature_brain,
            "max_tokens": self.settings.max_tokens_brain,
            "extra_body": self._extra_body(),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            return await self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            self._reraise_unreachable(exc, model=model, role="Brain")

    async def synthesize(self, prompt: str, *, system: str | None = None) -> str:
        """
        Always call DOMAIN_FT_MODEL (default: domain-ft) via LiteLLM.

        DOMAIN_PREDICT_MODE only selects the OpenAI API shape:
          - chat / llm: chat.completions
          - completion: completions
        """
        mode = self.settings.domain_predict_mode
        model = self.settings.domain_ft_model

        try:
            if mode == "completion":
                full = prompt if system is None else f"{system}\n\n{prompt}"
                resp = await self._client.completions.create(
                    model=model,
                    prompt=full,
                    temperature=self.settings.temperature_synthesis,
                    max_tokens=self.settings.max_tokens_synthesis,
                    extra_body=self._extra_body(),
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
                extra_body=self._extra_body(),
            )
            content = resp.choices[0].message.content
            return (content or "").strip()
        except OpenAIError as exc:
            self._reraise_unreachable(exc, model=model, role="Domain-ft / Nemotron")
