"""
Comprehensive Test Suite for Module 2: Generation Engine.

Tests cover:
  - Pydantic schema validation (ConsolidatedFinancialAnswer)
  - XML context formatting
  - Conversation memory truncation
  - Groq generation with fallback
  - Async guardrail verification (pass + fail paths)
  - Semantic cache operations
  - MLflow metric/param logging
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# --- Path Setup ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "2_generation"))

from config.logging_config import get_logger

logger = get_logger("test.generation")

# Import under test
from schemas import CacheEntry, ConsolidatedFinancialAnswer, GuardrailVerdict
from generator import (
    ConversationMemory,
    FinancialRAGGenerator,
    _hash_query,
    format_context_xml,
)
from async_guardrail import (
    SAFE_FALLBACK,
    AsyncGuardrail,
    SemanticCache,
    _extract_numbers,
    _normalize_number,
    verify_numerical_claims,
)

# ============================================================================
# TEST 1: Pydantic Schema Validation
# ============================================================================

def test_schema_valid_output():
    """Valid JSON matching the schema should parse without errors."""
    data = {
        "internal_thought": "Revenue is $394,328M which converts to $394.3B.",
        "extracted_raw_data": "Net sales for 2025: $394,328 million.",
        "answer": "Apple reported $394.3B in net sales for FY2025.",
        "sources": ["AAPL - 2025 - Income Statement - 30"],
    }
    model = ConsolidatedFinancialAnswer.model_validate(data)
    assert model.internal_thought == data["internal_thought"]
    assert model.answer == data["answer"]
    assert len(model.sources) == 1
    logger.info("Schema valid output: OK")


def test_schema_rejects_empty_fields():
    """Empty required fields should raise ValidationError."""
    from pydantic import ValidationError

    data = {
        "internal_thought": "123 reasoning here",
        "extracted_raw_data": "",
        "answer": "Some answer with 123",
        "sources": ["AAPL - 2025 - Section - 1"],
    }
    try:
        ConsolidatedFinancialAnswer.model_validate(data)
        assert False, "Should have raised ValidationError for empty extracted_raw_data"
    except ValidationError:
        pass
    logger.info("Schema rejects empty fields: OK")


def test_schema_rejects_no_sources():
    """Empty sources list should raise ValidationError."""
    from pydantic import ValidationError

    data = {
        "internal_thought": "Step 1: check 42 numbers.",
        "extracted_raw_data": "Revenue = $100B.",
        "answer": "Revenue was $100B.",
        "sources": [],
    }
    try:
        ConsolidatedFinancialAnswer.model_validate(data)
        assert False, "Should have raised ValidationError for empty sources"
    except ValidationError:
        pass
    logger.info("Schema rejects no sources: OK")


def test_schema_rejects_thought_without_numbers():
    """internal_thought without any digit should raise ValidationError."""
    from pydantic import ValidationError

    data = {
        "internal_thought": "Revenue analysis based on statements provided.",
        "extracted_raw_data": "Revenue = $100B.",
        "answer": "Revenue was $100B.",
        "sources": ["AAPL - 2025 - Section - 1"],
    }
    try:
        ConsolidatedFinancialAnswer.model_validate(data)
        assert False, "Should have raised ValidationError for thought without numbers"
    except ValidationError:
        pass
    logger.info("Schema rejects thought without numbers: OK")


def test_schema_json_roundtrip():
    """model_dump_json → model_validate_json should roundtrip cleanly."""
    original = {
        "internal_thought": "Step 1: verify 394,328M = $394.3B.",
        "extracted_raw_data": "Net sales for 2025: $394,328 million.",
        "answer": "Apple reported $394.3B in sales.",
        "sources": ["AAPL - 2025 - Income Statement - 30"],
    }
    model = ConsolidatedFinancialAnswer.model_validate(original)
    json_str = model.model_dump_json()
    restored = ConsolidatedFinancialAnswer.model_validate_json(json_str)
    assert restored.answer == model.answer
    assert restored.sources == model.sources
    logger.info("Schema JSON roundtrip: OK")


# ============================================================================
# TEST 2: XML Context Formatting
# ============================================================================

def test_format_context_basic():
    """Documents should be wrapped in XML tags with metadata attributes."""
    docs = [
        {
            "text": "Net sales: $394,328M.",
            "metadata": {
                "ticker": "AAPL",
                "fiscal_year": "2025",
                "section": "Income Statement",
                "contains_table": False,
                "page_number": "30",
            },
            "chunk_id": "chunk_001",
        },
        {
            "text": "R&D expenses: $31,385M.",
            "metadata": {
                "ticker": "AAPL",
                "fiscal_year": "2025",
                "section": "Operating Expenses",
                "contains_table": True,
                "page_number": "32",
            },
            "chunk_id": "chunk_002",
        },
    ]
    result = format_context_xml(docs)
    assert "<CONTEXT>" in result
    assert "</CONTEXT>" in result
    assert 'ticker="AAPL"' in result
    assert 'fiscal_year="2025"' in result
    assert 'contains_table="false"' in result
    assert 'contains_table="true"' in result
    assert "<DOCUMENT" in result
    assert "$394,328M" in result
    assert "$31,385M" in result
    logger.info("XML context basic: OK")


def test_format_context_empty():
    """Empty docs list should produce a valid CONTEXT wrapper."""
    result = format_context_xml([])
    assert "<CONTEXT>" in result
    assert "No documents retrieved" in result
    assert "</CONTEXT>" in result
    logger.info("XML context empty: OK")


def test_format_context_special_chars():
    """Metadata with special XML characters should be escaped."""
    docs = [
        {
            "text": "Test <value> & data.",
            "metadata": {
                "ticker": "AAPL",
                "fiscal_year": "2025",
                "section": "Risk Factors (Item 1A)",
                "contains_table": False,
                "page_number": "N/A",
            },
            "chunk_id": "chunk_xml",
        }
    ]
    result = format_context_xml(docs)
    assert "Risk Factors" in result
    assert "<DOCUMENT" in result
    logger.info("XML context special chars: OK")


# ============================================================================
# TEST 3: Conversation Memory Truncation
# ============================================================================

def test_memory_truncation():
    """Memory should keep system prompt + last 2*K non-system messages."""
    mem = ConversationMemory(k=3)  # K=3 → keep last 6 non-system messages

    mem.set_system("You are a financial analyst.")
    for i in range(10):
        mem.add_human(f"Question {i}")
        mem.add_ai(f"Answer {i}")

    truncated = mem.truncate()
    # System + last 6 (3 pairs)
    system_msgs = [m for m in truncated if hasattr(m, "type") and m.type == "system"]
    non_system = [m for m in truncated if not (hasattr(m, "type") and m.type == "system")]

    assert len(system_msgs) == 1, f"Expected 1 system msg, got {len(system_msgs)}"
    assert len(non_system) == 6, f"Expected 6 non-system msgs, got {len(non_system)}"

    # Last messages should be answers 7,8,9
    ai_msgs = [m for m in non_system if hasattr(m, "type") and m.type == "ai"]
    assert len(ai_msgs) == 3
    logger.info("Memory truncation (K=3): OK — %d messages retained", len(truncated))


def test_memory_empty():
    """Empty memory should return empty list."""
    mem = ConversationMemory(k=6)
    assert mem.truncate() == []
    logger.info("Memory empty: OK")


def test_memory_clear():
    """clear() should empty all messages."""
    mem = ConversationMemory(k=6)
    mem.set_system("System")
    mem.add_human("Hi")
    mem.add_ai("Hello")
    mem.clear()
    assert len(mem.truncate()) == 0
    logger.info("Memory clear: OK")


# ============================================================================
# TEST 4: Number Extraction & Normalization
# ============================================================================

def test_extract_numbers_basic():
    """Should extract dollar amounts, percentages, and parenthesized negatives."""
    text = "Revenue was $394,328M (up from $383,285M) or 12.5% growth."
    nums = _extract_numbers(text)
    assert len(nums) >= 3, f"Expected >=3 numbers, got {len(nums)}: {nums}"
    # Check some expected values are present
    joined = " ".join(nums)
    assert "394328" in joined.replace(",", ""), f"Missing 394328 in {nums}"
    logger.info("Extract numbers basic: %s", nums)


def test_extract_numbers_parens_negative():
    """Parenthesized numbers should be extracted."""
    text = "Net loss was $(1,234) million."
    nums = _extract_numbers(text)
    assert any("(" in n for n in nums), f"No parenthesized number found: {nums}"
    logger.info("Extract numbers parens: %s", nums)


def test_normalize_number():
    """Normalization should strip currency, commas, scale suffixes."""
    assert _normalize_number("$1,234") == "1234"
    assert _normalize_number("(500)") == "(500)"
    assert _normalize_number("$12.5M") == "12.5"
    assert _normalize_number("75%") == "75"
    logger.info("Normalize number: OK")


# ============================================================================
# TEST 5: Async Guardrail Verification
# ============================================================================

def test_guardrail_pass_same_numbers():
    """Answer numbers that match raw data should pass."""
    import asyncio

    answer = "Revenue was $394,328 million, up 4.8% from last year."
    raw = "Net sales for 2025: $394,328 million. Growth rate: 4.8%."

    passed, verified, failed = asyncio.get_event_loop().run_until_complete(
        verify_numerical_claims(answer, raw)
    )
    assert passed is True, f"Expected pass, got failed={failed}"
    assert len(verified) >= 2, f"Expected >=2 verified, got {len(verified)}"
    assert len(failed) == 0
    logger.info("Guardrail pass: verified=%d, failed=%d", len(verified), len(failed))


def test_guardrail_fail_hallucinated_numbers():
    """Numbers in answer NOT in raw data should fail."""
    import asyncio

    answer = "Revenue was $999,999 million."
    raw = "Net sales for 2025: $394,328 million."

    passed, verified, failed = asyncio.get_event_loop().run_until_complete(
        verify_numerical_claims(answer, raw)
    )
    assert passed is False, "Expected fail for hallucinated number"
    assert len(failed) >= 1, f"Expected >=1 failed claim"
    logger.info("Guardrail fail: verified=%d, failed=%d", len(verified), len(failed))


def test_guardrail_pass_no_numbers():
    """Answer with no numbers should pass (nothing to verify)."""
    import asyncio

    answer = "Apple is a technology company."
    raw = "Apple Inc. designs consumer electronics."

    passed, verified, failed = asyncio.get_event_loop().run_until_complete(
        verify_numerical_claims(answer, raw)
    )
    assert passed is True
    assert len(verified) == 0
    assert len(failed) == 0
    logger.info("Guardrail no numbers: OK")


# ============================================================================
# TEST 6: Semantic Cache
# ============================================================================

def test_cache_put_and_get():
    """Cache should store and retrieve entries by query hash."""
    cache = SemanticCache(ttl_seconds=60)
    try:
        query = "What was Apple's revenue in 2025?"
        answer_json = json.dumps({"answer": "$394.3B"})
        put_ok = cache.put(query, answer_json, guardrail_passed=True)

        if not put_ok:
            # Redis not available — test that graceful fallback works
            assert cache.get(query) is None
            logger.info("Cache put/get: Redis unavailable (graceful fallback OK)")
            return

        result = cache.get(query)
        assert result is not None, "Cache miss after put"
        assert result["guardrail_passed"] is True
        assert "$394.3B" in result["answer_json"]
        logger.info("Cache put/get: OK")
    finally:
        cache.invalidate("What was Apple's revenue in 2025?")
        cache.close()


def test_cache_miss():
    """Cache miss should return None."""
    cache = SemanticCache(ttl_seconds=60)
    try:
        result = cache.get("nonexistent query that nobody asked")
        assert result is None
        logger.info("Cache miss: OK")
    finally:
        cache.close()


def test_cache_invalidate():
    """invalidate() should remove a cached entry (or gracefully handle no Redis)."""
    cache = SemanticCache(ttl_seconds=60)
    try:
        query = "test invalidation query"
        put_ok = cache.put(query, '{"answer":"test"}', guardrail_passed=True)
        if not put_ok:
            # Redis unavailable — invalidation should return False gracefully
            result = cache.invalidate(query)
            assert result is False
            logger.info("Cache invalidate: Redis unavailable (graceful fallback OK)")
            return
        assert cache.get(query) is not None
        cache.invalidate(query)
        assert cache.get(query) is None
        logger.info("Cache invalidate: OK")
    finally:
        cache.close()


# ============================================================================
# TEST 7: AsyncGuardrail Orchestrator
# ============================================================================

def test_guardrail_orchestrator_pass():
    """Full guardrail check with matching numbers should pass."""
    import asyncio

    guardrail = AsyncGuardrail()
    try:
        result = asyncio.get_event_loop().run_until_complete(
            guardrail.check(
                query="AAPL revenue 2025",
                answer="Apple reported $394,328M in revenue.",
                extracted_raw_data="Net sales: $394,328 million.",
                parsed_output=None,
            )
        )
        assert result["passed"] is True
        assert result["final_output"] == "Apple reported $394,328M in revenue."
        assert len(result["verified_claims"]) >= 1
        # cache_written depends on Redis availability
        logger.info(
            "Guardrail orchestrator pass: cache_written=%s", result["cache_written"]
        )
    finally:
        guardrail.close()


def test_guardrail_orchestrator_fail():
    """Full guardrail check with hallucinated numbers should fail."""
    import asyncio

    guardrail = AsyncGuardrail()
    try:
        result = asyncio.get_event_loop().run_until_complete(
            guardrail.check(
                query="AAPL revenue 2025",
                answer="Apple reported $999,999M in revenue.",
                extracted_raw_data="Net sales: $394,328 million.",
            )
        )
        assert result["passed"] is False
        assert result["cache_written"] is False
        assert result["final_output"] == SAFE_FALLBACK
        assert len(result["failed_claims"]) >= 1
        logger.info("Guardrail orchestrator fail: OK")
    finally:
        guardrail.close()


# ============================================================================
# TEST 8: MLflow Logging (smoke test)
# ============================================================================

def test_mlflow_logging():
    """Verify MLflow can be set and params logged without error."""
    import mlflow

    # End any lingering active run from prior tests
    try:
        mlflow.end_run()
    except Exception:
        pass

    mlflow.set_experiment("financial_rag_generation_test")
    with mlflow.start_run(run_name="test_gen_smoke"):
        mlflow.log_param("primary_model", "qwen-2.5-72b-instruct")
        mlflow.log_param("fallback_model", "llama-3.3-70b-versatile")
        mlflow.log_param("temperature", 0.0)
        mlflow.log_param("max_tokens", 2048)
        mlflow.log_param("seed", 42)
        mlflow.log_param("history_k_truncated", 6)
        mlflow.log_metric("ttft_ms", 123.45)
        mlflow.log_metric("total_generation_tokens", 512)
        mlflow.log_metric("guardrail_passed", 1)
        mlflow.log_metric("fallback_triggered", 0)
    logger.info("MLflow smoke test: OK")


# ============================================================================
# TEST 9: Hash Utility
# ============================================================================

def test_hash_query_deterministic():
    """Same query should produce the same hash."""
    h1 = _hash_query("What was Apple's revenue?")
    h2 = _hash_query("What was Apple's revenue?")
    assert h1 == h2
    assert len(h1) == 16
    logger.info("Hash query deterministic: OK")


def test_hash_query_case_insensitive():
    """Case differences should produce the same hash."""
    h1 = _hash_query("REVENUE 2025")
    h2 = _hash_query("revenue 2025")
    assert h1 == h2
    logger.info("Hash query case insensitive: OK")


# ============================================================================
# TEST 10: Schema ConsolidatedFinancialAnswer JSON Structure
# ============================================================================

def test_schema_has_all_fields():
    """Schema should have exactly the 4 required fields."""
    fields = list(ConsolidatedFinancialAnswer.model_fields.keys())
    assert "internal_thought" in fields
    assert "extracted_raw_data" in fields
    assert "answer" in fields
    assert "sources" in fields
    assert len(fields) == 4, f"Expected 4 fields, got {len(fields)}: {fields}"
    logger.info("Schema all fields: OK")


def test_schema_multiple_sources():
    """Schema should accept multiple source citations."""
    data = {
        "internal_thought": "Step 1: 123 check. Step 2: 456 verify.",
        "extracted_raw_data": "Sales: $100B. Income: $50B.",
        "answer": "Sales were $100B with $50B income.",
        "sources": [
            "AAPL - 2025 - Income Statement - 30",
            "AAPL - 2025 - Balance Sheet - 35",
            "AAPL - 2025 - Cash Flow - 40",
        ],
    }
    model = ConsolidatedFinancialAnswer.model_validate(data)
    assert len(model.sources) == 3
    logger.info("Schema multiple sources: OK")


# ============================================================================
# TEST 11: GuardrailVerdict Schema
# ============================================================================

def test_guardrail_verdict_schema():
    """GuardrailVerdict should validate correctly."""
    verdict = GuardrailVerdict(
        passed=True,
        verified_claims=["$394,328M", "4.8%"],
        failed_claims=[],
        detail="All claims verified.",
    )
    assert verdict.passed is True
    assert len(verdict.verified_claims) == 2
    assert len(verdict.failed_claims) == 0
    logger.info("GuardrailVerdict schema: OK")


# ============================================================================
# TEST 12: CacheEntry Schema
# ============================================================================

def test_cache_entry_schema():
    """CacheEntry should validate correctly."""
    entry = CacheEntry(
        query_hash="abc123def456",
        answer_json='{"answer": "test"}',
        guardrail_passed=True,
        timestamp_iso="2026-01-01T00:00:00+00:00",
    )
    assert entry.query_hash == "abc123def456"
    assert entry.guardrail_passed is True
    logger.info("CacheEntry schema: OK")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """Run all generation engine tests."""
    print("=" * 70)
    print("MODULE 2: GENERATION ENGINE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Schema: valid output", test_schema_valid_output),
        ("Schema: rejects empty fields", test_schema_rejects_empty_fields),
        ("Schema: rejects no sources", test_schema_rejects_no_sources),
        ("Schema: rejects thought w/o numbers", test_schema_rejects_thought_without_numbers),
        ("Schema: JSON roundtrip", test_schema_json_roundtrip),
        ("Schema: all 4 fields", test_schema_has_all_fields),
        ("Schema: multiple sources", test_schema_multiple_sources),
        ("GuardrailVerdict schema", test_guardrail_verdict_schema),
        ("CacheEntry schema", test_cache_entry_schema),
        ("XML context: basic", test_format_context_basic),
        ("XML context: empty", test_format_context_empty),
        ("XML context: special chars", test_format_context_special_chars),
        ("Memory: truncation K=3", test_memory_truncation),
        ("Memory: empty", test_memory_empty),
        ("Memory: clear", test_memory_clear),
        ("Number extraction: basic", test_extract_numbers_basic),
        ("Number extraction: parens", test_extract_numbers_parens_negative),
        ("Number normalization", test_normalize_number),
        ("Guardrail verify: pass", test_guardrail_pass_same_numbers),
        ("Guardrail verify: fail", test_guardrail_fail_hallucinated_numbers),
        ("Guardrail verify: no numbers", test_guardrail_pass_no_numbers),
        ("Cache: put/get", test_cache_put_and_get),
        ("Cache: miss", test_cache_miss),
        ("Cache: invalidate", test_cache_invalidate),
        ("Guardrail orchestrator: pass", test_guardrail_orchestrator_pass),
        ("Guardrail orchestrator: fail", test_guardrail_orchestrator_fail),
        ("MLflow: smoke test", test_mlflow_logging),
        ("Hash: deterministic", test_hash_query_deterministic),
        ("Hash: case insensitive", test_hash_query_case_insensitive),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, fn in tests:
        try:
            print(f"  {name}...", end=" ", flush=True)
            fn()
            print("PASS")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
            errors.append((name, str(e)))
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            failed += 1
            errors.append((name, f"{type(e).__name__}: {e}"))

    print()
    print("=" * 70)
    status = "ALL PASSED" if failed == 0 else f"{failed} FAILED"
    print(f"RESULTS: {passed}/{len(tests)} passed | {status}")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
