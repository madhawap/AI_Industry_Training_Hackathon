"""Pydantic request / response models for the FastAPI surface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body of ``POST /query``."""

    question: str = Field(..., min_length=1, description="User question")


class ToolTraceItem(BaseModel):
    """One executed tool call, reported back to the caller for transparency."""

    tool: str
    args: dict[str, Any]
    result: Any


class QueryResponse(BaseModel):
    """Response of ``POST /query`` — the harness grades only ``answer``."""

    answer: str
    steps: int = Field(..., ge=0, description="Number of agent / tool loop iterations")
    tool_trace: list[ToolTraceItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response of ``GET /health`` once the service is ready."""

    status: str = "ok"
