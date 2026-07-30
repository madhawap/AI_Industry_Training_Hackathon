"""Pydantic request / response models for the FastAPI surface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class ToolTraceItem(BaseModel):
    tool: str
    args: dict[str, Any]
    result: Any


class QueryResponse(BaseModel):
    answer: str
    steps: int = Field(..., ge=0, description="Number of agent / tool loop iterations")
    tool_trace: list[ToolTraceItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
