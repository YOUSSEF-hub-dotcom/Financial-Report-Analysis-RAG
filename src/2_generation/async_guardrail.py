"""
Asynchronous Numerical Hallucination Guardrail & Semantic Cache Handler.

Runs a lightweight background verification pass that cross-checks every
numerical claim in the generated answer against the extracted_raw_data.

  - PASS → writes query, context hash, and verified JSON to Redis cache.
  - FAIL → bypasses cache, aborts output, returns safe fallback response.
"""

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

import mlflow

from config.logging_config import get_logger
from config.settings import REDIS_URL

logger = get_logger("generation.guardrail")

# ---------------------------------------------------------------------------
# Hashing utility (shared with generator.py)
# ---------------------------------------------------------------------------


def _hash_query(query: str) -> str:
    """Deterministic short hash for cache keys."""
    import hashlib
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Safe fallback response returned when guardrail fails
# ---------------------------------------------------------------------------
SAFE_FALLBACK = (
    "Direct arithmetic verification failed for the numbers present "
    "in the provided reports."
)

# ---------------------------------------------------------------------------
# Numerical extraction helpers
# ---------------------------------------------------------------------------

# Matches: $1,234.56  $1234  (1,234)  -1234  12.5%  1234M  1234B  1234K
_NUM_PATTERN = re.compile(
    r"(?:[\$]?\s*[\(]?\s*-?\s*)"          # optional currency, parens, minus
    r"[\d,]+"                              # integer part with commas
    r"(?:\.\d+)?"                          # optional decimal
    r"(?:\s*[%])"                          # optional percent
    r"|"
    r"(?:[\$]?\s*[\(]?\s*-?\s*)"          # currency prefix
    r"[\d,]+"                              # digits
    r"(?:\.\d+)?"                          # optional decimal
    r"(?:\s*[MBKmbk])?"                   # optional scale suffix
    r"|"
    r"\([\d,]+(?:\.\d+)?\)"               # parenthesized negative
)

# Simplified: extract all number-like tokens
_NUM_EXTRACT = re.compile(
    r"\(?\s*-?\s*\d[\d,]*\.?\d*\s*[%MBKmbk]?\s*\)?"
)


def _extract_numbers(text: str) -> list[str]:
    """Extract all number-like tokens from text, normalized to stripped form."""
    matches = _NUM_EXTRACT.findall(text)
    normalized: list[str] = []
    for m in matches:
        stripped = m.strip()
        # Collapse internal whitespace
        stripped = re.sub(r"\s+", "", stripped)
        if stripped:
            normalized.append(stripped)
    return normalized


def _normalize_number(num_str: str) -> str:
    """
    Normalize a number string for comparison.

    Removes currency symbols, spaces, and standardizes formatting.
    """
    s = num_str.strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    s = s.replace("M", "").replace("B", "").replace("K", "")
    s = s.replace("m", "").replace("b", "").replace("k", "")
    s = s.replace("%", "")
    # Keep parens for negatives
    return s


# ---------------------------------------------------------------------------
# Async Verification Loop
# ---------------------------------------------------------------------------

async def verify_numerical_claims(
    answer: str,
    extracted_raw_data: str,
) -> tuple[bool, list[str], list[str]]:
    """
    Asynchronously verify that every number in the answer exists in the raw data.

    This is a lightweight regex-based check (no LLM call) that runs in a
    background thread to avoid blocking the main generation path.

    Args:
        answer: The generated answer text.
        extracted_raw_data: The verbatim extracted facts from context.

    Returns:
        (passed, verified_claims, failed_claims)
    """
    # Run CPU-bound regex work in a thread to stay async
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _verify_sync, answer, extracted_raw_data
    )


def _verify_sync(
    answer: str,
    extracted_raw_data: str,
) -> tuple[bool, list[str], list[str]]:
    """Synchronous verification implementation."""
    answer_numbers = _extract_numbers(answer)
    raw_numbers = _extract_numbers(extracted_raw_data)

    if not answer_numbers:
        # No numbers to verify — pass by default
        return True, [], []

    raw_normalized = {_normalize_number(n) for n in raw_numbers}

    verified: list[str] = []
    failed: list[str] = []

    for num in answer_numbers:
        norm = _normalize_number(num)
        if norm in raw_normalized or norm.lstrip("0") in {n.lstrip("0") for n in raw_normalized}:
            verified.append(num)
        else:
            failed.append(num)

    passed = len(failed) == 0
    return passed, verified, failed


# ---------------------------------------------------------------------------
# Redis Semantic Cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Redis-backed semantic cache for verified RAG outputs.

    Stores query→answer mappings keyed by SHA-256 of the lowercased query.
    Entries include guardrail status and timestamp for TTL management.
    """

    def __init__(self, redis_url: str = REDIS_URL, ttl_seconds: int = 3600):
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._client = None

    def _get_client(self):
        """Lazy-load Redis client."""
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(
                    self._redis_url,
                    socket_connect_timeout=3,
                    decode_responses=True,
                )
                self._client.ping()
                logger.info("Redis cache connected: %s", self._redis_url)
            except Exception as exc:
                logger.warning(
                    "Redis unavailable (cache disabled): %s", exc
                )
                self._client = False  # type: ignore[assignment]
        return self._client

    def get(self, query: str) -> dict | None:
        """
        Retrieve a cached answer for the given query.

        Returns:
            Cached dict with keys: query_hash, answer_json, guardrail_passed,
            timestamp_iso, or None on miss.
        """
        client = self._get_client()
        if not client:
            return None

        key = _hash_query(query)
        try:
            raw = client.get(f"rag_cache:{key}")
            if raw:
                entry = json.loads(raw)
                logger.info("Cache HIT for query hash=%s", key)
                return entry
        except Exception as exc:
            logger.warning("Cache GET failed: %s", exc)
        return None

    def put(
        self,
        query: str,
        answer_json: str,
        guardrail_passed: bool,
    ) -> bool:
        """
        Write a verified answer to the cache.

        Args:
            query: Original user query.
            answer_json: JSON string of the ConsolidatedFinancialAnswer.
            guardrail_passed: Whether the guardrail verification passed.

        Returns:
            True on success, False on failure.
        """
        client = self._get_client()
        if not client:
            return False

        key = _hash_query(query)
        entry = {
            "query_hash": key,
            "answer_json": answer_json,
            "guardrail_passed": guardrail_passed,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        }

        try:
            client.setex(
                f"rag_cache:{key}",
                self._ttl,
                json.dumps(entry),
            )
            logger.info("Cache SET for query hash=%s (ttl=%ds)", key, self._ttl)
            return True
        except Exception as exc:
            logger.warning("Cache SET failed: %s", exc)
            return False

    def invalidate(self, query: str) -> bool:
        """Remove a cached entry for the given query."""
        client = self._get_client()
        if not client:
            return False

        key = _hash_query(query)
        try:
            client.delete(f"rag_cache:{key}")
            return True
        except Exception as exc:
            logger.warning("Cache DELETE failed: %s", exc)
            return False

    def flush_all(self) -> int:
        """Delete all rag_cache:* keys from Redis.

        Returns:
            Number of keys deleted, or -1 on error / unavailable.
        """
        client = self._get_client()
        if not client:
            return -1
        try:
            keys = client.keys("rag_cache:*")
            if not keys:
                return 0
            count = client.delete(*keys)
            logger.info("Cache flushed: %d keys removed", count)
            return count
        except Exception as exc:
            logger.warning("Cache flush failed: %s", exc)
            return -1

    def close(self) -> None:
        """Close Redis connection."""
        if self._client and self._client is not False:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


# ---------------------------------------------------------------------------
# Guardrail Orchestrator
# ---------------------------------------------------------------------------

class AsyncGuardrail:
    """
    Orchestrates the full verification + cache flow:

    1. Verify numerical claims in the answer against extracted_raw_data
    2. If PASS → write to Redis semantic cache, log MLflow, return answer
    3. If FAIL → bypass cache, log MLflow, return safe fallback response
    """

    def __init__(self, cache: SemanticCache | None = None):
        self._cache = cache or SemanticCache()

    async def check(
        self,
        query: str,
        answer: str,
        extracted_raw_data: str,
        parsed_output: Any = None,
        skip_cache: bool = False,
    ) -> dict[str, Any]:
        """
        Run the full guardrail verification pipeline.

        Args:
            query: Original user query.
            answer: Generated answer text.
            extracted_raw_data: Verbatim extracted facts from context.
            parsed_output: Optional parsed ConsolidatedFinancialAnswer for caching.
            skip_cache: If True, skip writing to Redis cache even on pass.

        Returns:
            Dict with: passed, verified_claims, failed_claims, final_output,
            cache_written, detail.
        """
        t0 = time.time()

        # Step 1: Verify numerical claims
        passed, verified, failed = await verify_numerical_claims(
            answer, extracted_raw_data
        )

        elapsed_ms = (time.time() - t0) * 1000
        detail = (
            f"Verified {len(verified)} claims, {len(failed)} failed. "
            f"Elapsed: {elapsed_ms:.1f}ms"
        )

        # Log to MLflow
        try:
            mlflow.log_metric("guardrail_passed", int(passed))
            mlflow.log_metric("guardrail_verified_count", len(verified))
            mlflow.log_metric("guardrail_failed_count", len(failed))
            mlflow.log_metric("guardrail_elapsed_ms", round(elapsed_ms, 2))
        except Exception:
            pass

        cache_written = False

        # Auto-detect invalid / fallback answers that should never be cached
        is_not_available = "not available" in answer.lower()
        auto_skip = is_not_available
        effective_skip = skip_cache or auto_skip

        if passed:
            if not effective_skip:
                # Step 2a: PASS — write to cache
                if parsed_output is not None:
                    try:
                        answer_json = (
                            parsed_output.model_dump_json()
                            if hasattr(parsed_output, "model_dump_json")
                            else json.dumps(str(parsed_output))
                        )
                    except Exception:
                        answer_json = json.dumps({"answer": answer})
                else:
                    answer_json = json.dumps({"answer": answer})

                cache_written = self._cache.put(query, answer_json, guardrail_passed=True)
                logger.info("Guardrail PASSED — %s (cache written)", detail)
            else:
                logger.info("Guardrail PASSED — %s (cache skipped: skip_cache=%s)", detail, effective_skip)
            final_output = answer
        else:
            # Step 2b: FAIL — abort output, return safe fallback
            final_output = SAFE_FALLBACK
            logger.warning("Guardrail FAILED — %s", detail)

        return {
            "passed": passed,
            "verified_claims": verified,
            "failed_claims": failed,
            "final_output": final_output,
            "cache_written": cache_written,
            "detail": detail,
        }

    def close(self) -> None:
        """Close underlying cache connections."""
        self._cache.close()
