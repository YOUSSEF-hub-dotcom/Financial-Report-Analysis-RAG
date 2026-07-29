"""
Structured Pydantic Output Schemas for the Financial RAG Generation Engine.

Enforces strict JSON schema compliance on LLM outputs before returning
to the user.  Every field is validated with clear error messages so that
malformed generations are caught early and the fallback path is triggered.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator


class ConsolidatedFinancialAnswer(BaseModel):
    """
    The canonical output schema for every RAG generation.

    The LLM is prompted to produce JSON matching this structure exactly.
    Pydantic validation ensures no partial or hallucinated output reaches
    the end user.
    """

    internal_thought: str = Field(
        ...,
        description=(
            "Step-by-step financial reasoning. Must include unit-scaling "
            "verification (M/B/K), currency symbol checks, and cross-document "
            "consistency notes."
        ),
        min_length=1,
    )
    extracted_raw_data: str = Field(
        ...,
        description=(
            "Exact verbatim numerical and financial facts copied directly "
            "from the XML context. No interpretation or rounding allowed."
        ),
        min_length=1,
    )
    answer: str = Field(
        ...,
        description=(
            "Executive-level answer synthesized exclusively from "
            "extracted_raw_data. Must reference specific figures."
        ),
        min_length=1,
    )
    sources: List[str] = Field(
        ...,
        description=(
            "List of exact citations formatted as "
            "'ticker - fiscal_year - section - page_number'."
        ),
        min_length=1,
    )

    @field_validator("sources")
    @classmethod
    def sources_must_be_non_empty_strings(cls, v: list[str]) -> list[str]:
        """Reject any empty or whitespace-only citation strings."""
        cleaned: list[str] = []
        for src in v:
            stripped = src.strip()
            if not stripped:
                continue
            cleaned.append(stripped)
        if not cleaned:
            raise ValueError("At least one non-empty source citation is required")
        return cleaned

    @field_validator("internal_thought")
    @classmethod
    def thought_must_reference_numbers(cls, v: str) -> str:
        """Soft check: internal thought should reference at least one number."""
        if not any(c.isdigit() for c in v):
            raise ValueError(
                "internal_thought should contain at least one numeric reference "
                "for financial reasoning verification"
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "internal_thought": (
                    "Revenue for fiscal year 2025 is reported as $394,328M in "
                    "the consolidated statements. Converting M to full scale: "
                    "394,328,000,000. No conflicting figures found across "
                    "segments."
                ),
                "extracted_raw_data": (
                    "Net sales for 2025: $394,328 million. "
                    "Net income for 2025: $93,736 million."
                ),
                "answer": (
                    "Apple Inc. reported total net sales of $394.3 billion "
                    "(approximately $394,328 million) for fiscal year 2025, "
                    "with net income of $93.7 billion ($93,736 million)."
                ),
                "sources": [
                    "AAPL - 2025 - Consolidated Statements of Operations - 30",
                    "AAPL - 2025 - Revenue by Segment - 35",
                ],
            }
        }


class GuardrailVerdict(BaseModel):
    """Result of an asynchronous hallucination verification pass."""

    passed: bool = Field(
        ...,
        description="True if all numerical claims in the answer match extracted_raw_data.",
    )
    verified_claims: List[str] = Field(
        default_factory=list,
        description="List of numerical claims that were successfully verified.",
    )
    failed_claims: List[str] = Field(
        default_factory=list,
        description="List of numerical claims that could not be verified (hallucination).",
    )
    detail: str = Field(
        default="",
        description="Human-readable summary of the verification outcome.",
    )


class CacheEntry(BaseModel):
    """A single entry in the Redis semantic cache."""

    query_hash: str
    answer_json: str
    guardrail_passed: bool
    timestamp_iso: str
