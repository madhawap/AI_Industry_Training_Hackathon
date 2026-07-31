"""Minimal HTTP wrapper around finance_agent.graph, for testing
llm_judge_grader.py's ``--endpoint`` mode against a real running service.

Implements the same contract as the hackathon submission
(``AI_Industry_Training_Hackathon/README.md`` "Agent Contract" and
``cognitivo_prep/src/main.py``):

    GET  /health  -> {"status": "ok"}
    POST /query   -> {"question": "..."}
                  <- {"answer": "...", "steps": N, "tool_trace": [...]}

This is a convenience double for testing this repo's finance_agent.py over
HTTP exactly like the real submission endpoint -- it is NOT the official
submission service (that's cognitivo_prep/src/main.py, which additionally
runs the real Qwen agent-brain + fine-tuned Nemotron via a LiteLLM proxy).
Anything that speaks this same contract works with
``llm_judge_grader.py --endpoint``, including the real submission service
once its model backend is reachable.

Run:

    .venv/bin/uvicorn api:app --app-dir evals-hackathon --port 8001

Then grade against it:

    .venv/bin/python evals-hackathon/llm_judge_grader.py --endpoint http://127.0.0.1:8001
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from finance_agent import extract_tool_trace, final_answer_text, graph  # noqa: E402
from finance_data import list_asx_tickers, load_afr, load_asx, load_rba  # noqa: E402

_ready = False


def _warm_datasets() -> None:
    """Parse every dataset once, synchronously, before serving traffic.

    Without this, the first few concurrent /query requests that happen to
    need AFR data all race to cold-load the ~780MB corpus at once -- in
    testing this produced 300s+ stalls under 3-way concurrency (the exact
    load the brief requires handling), even though each AFR tool call is
    fast once the corpus is warm. Same fix cognitivo_prep/src/main.py uses
    for its own warehouse load.
    """
    load_rba()
    for ticker in list_asx_tickers():
        load_asx(ticker)
    load_afr()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _ready
    await asyncio.to_thread(_warm_datasets)
    _ready = True
    yield
    _ready = False


app = FastAPI(title="finance-hackathon-agent (evals-hackathon demo)", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    steps: int
    tool_trace: list[dict]


@app.get("/health")
async def health() -> dict[str, str]:
    if not _ready:
        raise HTTPException(status_code=503, detail="datasets still warming up")
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    result = await graph.ainvoke({"messages": [{"role": "user", "content": request.question}]})
    messages = result["messages"]
    trace = extract_tool_trace(messages)
    return QueryResponse(answer=final_answer_text(messages), steps=len(trace), tool_trace=trace)
