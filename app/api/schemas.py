"""
Pydantic schemas for FastAPI request/response payloads.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field
from fastapi import UploadFile


class ChatQueryRequest(BaseModel):
    user_query: str = Field(
        ...,
        min_length=3,
        description="Natural language financial question (min 3 chars).",
        examples=["What are Apple's reportable business segments?"],
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Optional ticker filter (e.g. AAPL, MSFT, NVDA).",
    )
    fiscal_year: Optional[str] = Field(
        default=None,
        description="Optional fiscal year filter (e.g. 2025).",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for conversation memory.",
    )


class SourceCitation(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0–1).")
    ticker: str = Field(default="UNKNOWN")
    fiscal_year: str = Field(default="UNKNOWN")
    section: str = Field(default="General")
    text_snippet: str = Field(default="", max_length=300)


class GuardrailStatus(BaseModel):
    passed: bool = Field(default=True)
    verified_claims: list[str] = Field(default_factory=list)
    failed_claims: list[str] = Field(default_factory=list)
    detail: str = Field(default="")


class ChatQueryResponse(BaseModel):
    answer: str = Field(..., description="Generated financial answer.")
    sources: list[SourceCitation] = Field(
        default_factory=list, description="Source chunks used."
    )
    execution_time_ms: float = Field(
        ..., description="End-to-end execution time in milliseconds."
    )
    guardrail_status: GuardrailStatus = Field(
        default_factory=GuardrailStatus, description="Guardrail verification result."
    )
    model_used: str = Field(
        default="unknown", description="LLM model used for generation."
    )
    cache_hit: bool = Field(default=False)
    task_id: Optional[str] = Field(
        default=None, description="Background task identifier if applicable."
    )


class HealthServiceStatus(BaseModel):
    mongodb: str = Field(default="unknown")
    qdrant: str = Field(default="unknown")
    redis: str = Field(default="unknown")


class HealthCheckResponse(BaseModel):
    status: str = Field(default="ok")
    services: HealthServiceStatus = Field(default_factory=HealthServiceStatus)
    warmup_completed: bool = Field(default=False, description="Whether the startup warm-up routine finished successfully.")


class DocumentUploadResponse(BaseModel):
    filename: str
    ticker: str
    fiscal_year: str
    chunks_created: int = Field(default=0)
    mongo_count: int = Field(default=0)
    qdrant_count: int = Field(default=0)
    elapsed_seconds: float = Field(default=0.0)
    task_id: Optional[str] = Field(default=None)


class CacheFlushResponse(BaseModel):
    status: str = Field(default="ok")
    keys_removed: int = Field(default=0, description="Number of cache keys deleted.")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error description.")
    error_code: str = Field(default="INTERNAL_ERROR")
