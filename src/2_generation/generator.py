"""
Generation Engine — Groq-powered LLM with primary/fallback execution.

Handles:
  - XML context enclosure for retrieved documents
  - Conversation memory management (truncated to last K messages)
  - Primary-fallback execution with exponential-backoff retry
  - CFO-grade system prompt with zero-hallucination instructions
  - Token streaming support
  - MLflow experiment tracking for generation events
"""

import hashlib
import json
import re
import time
import uuid
from typing import Any, AsyncIterator

import mlflow
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import GenerationChunk
from langchain_groq import ChatGroq

from config.logging_config import get_logger
from config.settings import (
    GROQ_API_KEY,
    GROQ_FALLBACK_MODEL,
    GROQ_PRIMARY_MODEL,
    LLM_HISTORY_K,
    LLM_MAX_TOKENS,
    LLM_SEED,
    LLM_TEMPERATURE,
)

logger = get_logger("generation.engine")

# ---------------------------------------------------------------------------
# System Prompt — CFO-grade zero-hallucination instructions
# ---------------------------------------------------------------------------
_CFO_SYSTEM_PROMPT = """\
You are a senior financial analyst assistant for SEC 10-K filings.

STRICT RULES — ZERO HALLUCINATION:
1. ONLY use numbers and facts present in the <CONTEXT> documents provided.
2. NEVER fabricate, estimate, or interpolate financial figures.
3. Preserve ALL financial notation exactly: $ for currency, % for percentages,
   parentheses (X) for negatives, and scale suffixes M (millions), B (billions), K (thousands).
4. When scaling is mentioned (e.g., "in millions"), compute the full number explicitly
   in your internal_thought before answering.
5. If the requested information is NOT available in the provided reports, respond with:
   "The requested financial information is not available in the provided reports."
6. Always provide sources in the format: "ticker - fiscal_year - section - page_number"

OUTPUT FORMAT — You MUST respond with ONLY valid JSON. No markdown, no code fences,
no conversational preambles, no explanations outside the JSON. Output pure JSON
matching this exact schema:
{
  "internal_thought": "Step-by-step reasoning with number verification",
  "extracted_raw_data": "Exact verbatim facts from the context documents",
  "answer": "Executive-level answer using only extracted_raw_data",
  "sources": ["TICKER - YEAR - SECTION - PAGE"]
}
"""

# ---------------------------------------------------------------------------
# Context Formatting
# ---------------------------------------------------------------------------

def format_context_xml(documents: list[dict]) -> str:
    """
    Enclose retrieved candidate documents in structured XML tags.

    Each document is wrapped in <DOCUMENT> with metadata attributes and
    its text content in <CONTENT>.

    Args:
        documents: List of dicts with keys: text, metadata (ticker, fiscal_year,
                   section, contains_table, chunk_id, page_number).

    Returns:
        XML-formatted context string.
    """
    if not documents:
        return "<CONTEXT>\nNo documents retrieved.\n</CONTEXT>"

    parts: list[str] = ["<CONTEXT>"]
    for i, doc in enumerate(documents, start=1):
        meta = doc.get("metadata", {})
        ticker = meta.get("ticker", "UNKNOWN")
        year = meta.get("fiscal_year", "UNKNOWN")
        section = meta.get("section", "General")
        has_table = meta.get("contains_table", False)
        page = meta.get("page_number", "N/A")
        chunk_id = doc.get("chunk_id", f"doc_{i}")

        parts.append(
            f'<DOCUMENT id="{chunk_id}" ticker="{ticker}" '
            f'fiscal_year="{year}" section="{section}" '
            f'page="{page}" contains_table="{str(has_table).lower()}">'
        )
        parts.append(doc.get("text", ""))
        parts.append("</DOCUMENT>")
        parts.append("")

    parts.append("</CONTEXT>")
    return "\n".join(parts)


def _hash_query(query: str) -> str:
    """Deterministic short hash for cache keys."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Memory Management
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    Sliding-window conversation memory.

    Retains the system prompt plus the last K (human, assistant) message pairs.
    Older messages are discarded to stay within token budgets.
    """

    def __init__(self, k: int = LLM_HISTORY_K):
        self._k = k
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def add_human(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))

    def add_ai(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))

    def set_system(self, content: str) -> None:
        """Prepend or replace the system message."""
        if self._messages and isinstance(self._messages[0], SystemMessage):
            self._messages[0] = SystemMessage(content=content)
        else:
            self._messages.insert(0, SystemMessage(content=content))

    def truncate(self) -> list[BaseMessage]:
        """
        Return messages respecting the K-pair window.

        Always keeps the system prompt (index 0), then the last 2*K
        non-system messages.
        """
        if not self._messages:
            return []

        system_msgs = [m for m in self._messages if isinstance(m, SystemMessage)]
        non_system = [m for m in self._messages if not isinstance(m, SystemMessage)]

        # Keep last 2*K non-system messages (K pairs = K human + K AI)
        max_non_system = self._k * 2
        trimmed = non_system[-max_non_system:] if len(non_system) > max_non_system else non_system

        return system_msgs + trimmed

    def clear(self) -> None:
        self._messages.clear()


# ---------------------------------------------------------------------------
# Generation Engine
# ---------------------------------------------------------------------------

class FinancialRAGGenerator:
    """
    Groq-powered generation engine with primary/fallback, retry, and streaming.

    Flow:
        1. Build XML context from retrieved chunks
        2. Manage conversation memory (truncate to last K pairs)
        3. Call primary model with exponential-backoff retry
        4. On exhausted retries, silently fallback to secondary model
        5. Parse JSON output into ConsolidatedFinancialAnswer via Pydantic
        6. Log generation metrics to MLflow
    """

    def __init__(
        self,
        primary_model: str = GROQ_PRIMARY_MODEL,
        fallback_model: str = GROQ_FALLBACK_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        seed: int = LLM_SEED,
        history_k: int = LLM_HISTORY_K,
    ):
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._memory = ConversationMemory(k=history_k)
        self._run_id: str = ""
        self._mlflow_active = False

    # --- Internal helpers ---------------------------------------------------

    def _build_llm(self, model_name: str) -> ChatGroq:
        """Construct a ChatGroq instance for the given model."""
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            model_kwargs={
                "seed": self._seed,
                "response_format": {"type": "json_object"},
            },
        )

    def _start_mlflow(self) -> None:
        """Start an MLflow run if not already active."""
        try:
            mlflow.set_experiment("financial_rag_generation")
            self._run_id = str(uuid.uuid4())[:8]
            mlflow.start_run(run_name=f"gen_{self._run_id}", nested=True)
            mlflow.log_param("primary_model", self._primary_model)
            mlflow.log_param("fallback_model", self._fallback_model)
            mlflow.log_param("temperature", self._temperature)
            mlflow.log_param("max_tokens", self._max_tokens)
            mlflow.log_param("seed", self._seed)
            mlflow.log_param("history_k_truncated", self._memory._k)
            self._mlflow_active = True
        except Exception as exc:
            logger.warning("MLflow start failed (non-blocking): %s", exc)
            self._mlflow_active = False

    def _end_mlflow(
        self,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """End the active MLflow run, logging optional metrics."""
        if not self._mlflow_active:
            return
        try:
            if metrics:
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
            mlflow.end_run()
        except Exception as exc:
            logger.warning("MLflow end failed (non-blocking): %s", exc)
        finally:
            self._mlflow_active = False

    def _log_mlflow_param(self, key: str, value: Any) -> None:
        if not self._mlflow_active:
            return
        try:
            mlflow.log_param(key, value)
        except Exception:
            pass

    def _log_mlflow_metric(self, key: str, value: Any) -> None:
        if not self._mlflow_active:
            return
        try:
            mlflow.log_metric(key, value)
        except Exception:
            pass

    # --- Public API ---------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieved_docs: list[dict],
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Synchronous generation with primary/fallback and retry.

        Args:
            query: User question string.
            retrieved_docs: List of chunk dicts from vector retrieval.
            stream: If True, tokens are printed as they arrive (no JSON parsing).

        Returns:
            Dict with keys: raw_output, parsed (ConsolidatedFinancialAnswer or None),
            model_used, fallback_triggered, ttft_ms.
        """
        self._start_mlflow()
        t_start = time.time()
        fallback_triggered = False

        # Build context and memory
        context_xml = format_context_xml(retrieved_docs)
        self._memory.set_system(_CFO_SYSTEM_PROMPT)
        self._memory.add_human(f"{context_xml}\n\nQuestion: {query}")
        messages = self._memory.truncate()

        # Try primary model with retry
        raw_output, model_used, ttft_ms = self._call_with_retry(
            self._primary_model, messages, stream=stream
        )

        # If primary returned error, try fallback
        if raw_output is None:
            logger.warning(
                "Primary model %s exhausted — falling back to %s",
                self._primary_model,
                self._fallback_model,
            )
            fallback_triggered = True
            raw_output, model_used, ttft_ms = self._call_with_retry(
                self._fallback_model, messages, stream=stream
            )

        # If both failed, return safe fallback
        if raw_output is None:
            safe_msg = (
                "The requested financial information is not available "
                "in the provided reports."
            )
            total_ms = (time.time() - t_start) * 1000
            self._memory.add_ai(safe_msg)
            self._log_mlflow_metric("fallback_triggered", 1)
            self._log_mlflow_metric("guardrail_passed", 0)
            self._log_mlflow_metric("ttft_ms", 0)
            self._log_mlflow_metric("total_generation_tokens", 0)
            self._end_mlflow()
            return {
                "raw_output": safe_msg,
                "parsed": None,
                "model_used": "none",
                "fallback_triggered": True,
                "ttft_ms": 0,
            }

        # Parse JSON from LLM output
        parsed = self._parse_json_output(raw_output)
        total_ms = (time.time() - t_start) * 1000
        total_tokens = self._estimate_tokens(raw_output)

        # Update memory
        self._memory.add_ai(raw_output)

        # Log to MLflow
        self._log_mlflow_metric("fallback_triggered", int(fallback_triggered))
        self._log_mlflow_metric("ttft_ms", round(ttft_ms, 2))
        self._log_mlflow_metric("total_generation_tokens", total_tokens)
        self._end_mlflow()

        logger.info(
            "Generation complete: model=%s fallback=%s ttft=%.0fms tokens=%d",
            model_used,
            fallback_triggered,
            ttft_ms,
            total_tokens,
        )

        return {
            "raw_output": raw_output,
            "parsed": parsed,
            "model_used": model_used,
            "fallback_triggered": fallback_triggered,
            "ttft_ms": round(ttft_ms, 2),
        }

    async def agenerate(
        self,
        query: str,
        retrieved_docs: list[dict],
    ) -> dict[str, Any]:
        """
        Async generation with primary/fallback and retry.

        Same logic as generate() but uses ainvoke for non-blocking execution.
        """
        self._start_mlflow()
        t_start = time.time()
        fallback_triggered = False

        context_xml = format_context_xml(retrieved_docs)
        self._memory.set_system(_CFO_SYSTEM_PROMPT)
        self._memory.add_human(f"{context_xml}\n\nQuestion: {query}")
        messages = self._memory.truncate()

        # Primary model
        raw_output, model_used, ttft_ms = await self._acall_with_retry(
            self._primary_model, messages
        )

        if raw_output is None:
            logger.warning("Primary exhausted — fallback to %s", self._fallback_model)
            fallback_triggered = True
            raw_output, model_used, ttft_ms = await self._acall_with_retry(
                self._fallback_model, messages
            )

        if raw_output is None:
            safe_msg = (
                "The requested financial information is not available "
                "in the provided reports."
            )
            self._memory.add_ai(safe_msg)
            self._log_mlflow_metric("fallback_triggered", 1)
            self._log_mlflow_metric("guardrail_passed", 0)
            self._end_mlflow()
            return {
                "raw_output": safe_msg,
                "parsed": None,
                "model_used": "none",
                "fallback_triggered": True,
                "ttft_ms": 0,
            }

        parsed = self._parse_json_output(raw_output)
        total_ms = (time.time() - t_start) * 1000
        total_tokens = self._estimate_tokens(raw_output)

        self._memory.add_ai(raw_output)
        self._log_mlflow_metric("fallback_triggered", int(fallback_triggered))
        self._log_mlflow_metric("ttft_ms", round(ttft_ms, 2))
        self._log_mlflow_metric("total_generation_tokens", total_tokens)
        self._end_mlflow()

        return {
            "raw_output": raw_output,
            "parsed": parsed,
            "model_used": model_used,
            "fallback_triggered": fallback_triggered,
            "ttft_ms": round(ttft_ms, 2),
        }

    async def stream_tokens(
        self,
        query: str,
        retrieved_docs: list[dict],
    ) -> AsyncIterator[str]:
        """
        Async token-by-token streaming via astream.

        Yields individual token strings as they are generated.
        """
        context_xml = format_context_xml(retrieved_docs)
        self._memory.set_system(_CFO_SYSTEM_PROMPT)
        self._memory.add_human(f"{context_xml}\n\nQuestion: {query}")
        messages = self._memory.truncate()

        llm = self._build_llm(self._primary_model)
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content

    # --- Retry helpers ------------------------------------------------------

    def _call_with_retry(
        self,
        model_name: str,
        messages: list[BaseMessage],
        max_retries: int = 3,
        stream: bool = False,
    ) -> tuple[str | None, str, float]:
        """
        Call the Groq API with exponential-backoff retry.

        Returns (raw_output, model_name, ttft_ms) or (None, model_name, 0)
        on exhausted retries.
        """
        llm = self._build_llm(model_name)
        t_first = 0.0
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                t0 = time.time()
                response = llm.invoke(messages)
                t_first = (time.time() - t0) * 1000

                content = response.content if hasattr(response, "content") else str(response)
                if content and content.strip():
                    return content, model_name, t_first

                last_error = ValueError("Empty response from model")
            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    "Groq call attempt %d/%d failed (%s): %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    model_name,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error(
            "All %d retries exhausted for model %s: %s",
            max_retries,
            model_name,
            last_error,
        )
        return None, model_name, 0.0

    async def _acall_with_retry(
        self,
        model_name: str,
        messages: list[BaseMessage],
        max_retries: int = 3,
    ) -> tuple[str | None, str, float]:
        """Async variant of _call_with_retry."""
        import asyncio

        llm = self._build_llm(model_name)
        t_first = 0.0
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                t0 = time.time()
                response = await llm.ainvoke(messages)
                t_first = (time.time() - t0) * 1000

                content = response.content if hasattr(response, "content") else str(response)
                if content and content.strip():
                    return content, model_name, t_first

                last_error = ValueError("Empty response from model")
            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    "Async Groq attempt %d/%d failed (%s): %s — retry in %ds",
                    attempt + 1,
                    max_retries,
                    model_name,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        return None, model_name, 0.0

    # --- Output parsing -----------------------------------------------------

    def _parse_json_output(self, raw: str) -> "ConsolidatedFinancialAnswer | None":
        """
        Extract and validate JSON from the LLM's raw text output.

        Handles cases where the model wraps JSON in markdown code fences
        or thinking tags. Falls back to wrapping raw text into a default
        ConsolidatedFinancialAnswer when JSON parsing fails entirely.
        """
        from pydantic import ValidationError
        from schemas import ConsolidatedFinancialAnswer

        text = raw.strip()

        # Strip thinking tags (Qwen / DeepSeek style)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = text.strip()

        # Strip markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        # Try direct parse first
        try:
            data = json.loads(text)
            return ConsolidatedFinancialAnswer.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass
        except Exception:
            pass

        # Try to find a JSON object in the text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return ConsolidatedFinancialAnswer.model_validate(data)
            except (json.JSONDecodeError, ValidationError):
                pass
            except Exception:
                pass

        logger.warning("Failed to parse JSON from LLM output (length=%d) — using raw-text fallback", len(raw))

        # Fallback: wrap raw text into a default structured answer
        answer_text = raw.strip()
        if not answer_text:
            answer_text = "The requested financial information is not available in the provided reports."

        try:
            return ConsolidatedFinancialAnswer(
                internal_thought="1. Fallback: raw LLM output could not be parsed as structured JSON.",
                extracted_raw_data="No structured data was extracted from the LLM response.",
                answer=answer_text,
                sources=["unknown"],
            )
        except ValidationError:
            logger.error("Fallback ConsolidatedFinancialAnswer also failed validation")
            return None

    # --- Utility ------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (4 chars per token for English)."""
        return max(1, len(text) // 4)

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self._memory.clear()
