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
from src.tfql import Store
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
    store: Store | None = None
    ready: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state.settings = settings

    # Load the warehouse and precompute every in-memory series before serving.
    # Requests run against a 60-second clock; startup does not, so all of the
    # expensive work belongs here. /health stays 503 until this finishes.
    state.store = await asyncio.to_thread(Store.build)
    logger.info(
        "TFQL store warm | rba=%s | asx=%s (%s tickers) | afr=%s",
        state.store.rba.coverage.describe(),
        state.store.asx_coverage().describe(),
        len(state.store.tickers),
        state.store.afr_coverage.describe(),
    )

    register_all_tools(
        registry,
        default_timeout=settings.default_tool_timeout_seconds,
        store=state.store,
    )
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
    state.ready = True
    yield
    state.ready = False
    await state.llm.aclose()
    if state.store is not None:
        state.store.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Cognitivo Hackathon Query API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness, not liveness.

    The evaluation harness treats this as a hard gate and starts sending
    questions the moment it sees 200. Reporting ok while the warehouse is still
    loading would hand it a server that cannot answer yet, so this stays 503
    until startup has completed.
    """
    if not state.ready:
        raise HTTPException(status_code=503, detail="warming up")
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
