"""
Async FastAPI entrypoint.

Workflow: POST /query → Qwen Agent ⇄ Tools → Fine-tuned Synthesis → response
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

from src.config import Settings, get_settings
from src.graph.workflow import QueryWorkflow
from src.llm.client import LLMClients
from src.models import HealthResponse, QueryRequest, QueryResponse, ToolTraceItem
from src.tools import register_all_tools, registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("app")


class AppState:
    settings: Settings
    llm: LLMClients
    workflow: QueryWorkflow
    query_semaphore: asyncio.Semaphore


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state.settings = settings
    register_all_tools(registry, default_timeout=settings.default_tool_timeout_seconds)
    state.llm = LLMClients(settings)
    state.workflow = QueryWorkflow(settings, state.llm, registry)
    # Bound in-flight /query work while still allowing ≥3 concurrent requests.
    state.query_semaphore = asyncio.Semaphore(settings.max_concurrent_queries)
    logger.info(
        "Started on planned bind %s:%s | brain=%s | domain=%s | max_steps=%s | concurrency=%s",
        settings.host,
        settings.port,
        settings.brain_model,
        settings.domain_ft_model,
        settings.max_agent_steps,
        settings.max_concurrent_queries,
    )
    yield
    await state.llm.aclose()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Cognitivo Hackathon Query API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    try:
        async with state.query_semaphore:
            result: dict[str, Any] = await state.workflow.run(question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=502, detail=f"Upstream / workflow error: {exc}") from exc

    return QueryResponse(
        answer=result["answer"],
        steps=result["steps"],
        tool_trace=[ToolTraceItem(**item) for item in result["tool_trace"]],
    )


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        # Multiple workers are optional; async + semaphore already supports ≥3 concurrent queries.
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
