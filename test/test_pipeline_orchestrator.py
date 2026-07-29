"""
Unit tests for FinancialRAGPipeline orchestrator.

Tests the full query flow with mocked external services (Groq, Qdrant, MongoDB, Redis).
Verifies:
  - Cache hit returns cached answer without LLM call
  - Cache miss triggers retrieval → generation → guardrail
  - Metadata filtering (ticker, fiscal_year) is passed to Qdrant
  - Empty retrieval returns safe fallback
  - Guardrail failure returns safe fallback
  - Streaming yields tokens correctly
  - MLflow logging is triggered
"""

import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_ingestion"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "2_generation"))

from schemas import ConsolidatedFinancialAnswer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parsed_answer(
    answer: str = "Apple has two reportable segments: Products and Services.",
    extracted_raw_data: str = "Products and Services are the reportable segments. Americas revenue $167.0B.",
    sources: list[str] | None = None,
) -> ConsolidatedFinancialAnswer:
    if sources is None:
        sources = ["AAPL - 2025 - Business Overview - 10"]
    return ConsolidatedFinancialAnswer(
        internal_thought="Test internal thought with $167.0B figure.",
        extracted_raw_data=extracted_raw_data,
        answer=answer,
        sources=sources,
    )


def _make_qdrant_results(count: int = 3) -> list[dict]:
    results = []
    for i in range(count):
        results.append({
            "chunk_id": f"chunk_{i:03d}",
            "score": 0.85 - i * 0.05,
            "payload": {
                "chunk_id": f"chunk_{i:03d}",
                "ticker": "AAPL",
                "fiscal_year": "2025",
                "section": "Business Overview",
                "doc_type": "10-K",
                "contains_table": i == 2,
            },
        })
    return results


def _make_mongo_docs(qdrant_results: list[dict]) -> dict[str, dict]:
    docs = {}
    for r in qdrant_results:
        cid = r["chunk_id"]
        docs[cid] = {
            "chunk_id": cid,
            "raw_text": f"Apple Inc. is a global technology company. Chunk {cid} with detailed financial information about fiscal year 2025. Revenue figures and segment breakdown are reported in the annual 10-K filing. " * 3,
            "chunk_type": "text",
            "ticker": "AAPL",
            "fiscal_year": "2025",
            "section": "Business Overview",
        }
    return docs


def _make_gen_result(parsed: ConsolidatedFinancialAnswer | None = None) -> dict[str, Any]:
    return {
        "raw_output": json.dumps({
            "internal_thought": "Test $100B figure.",
            "extracted_raw_data": "Test data $100B.",
            "answer": "Test answer",
            "sources": ["AAPL - 2025 - Business Overview - 10"],
        }),
        "parsed": parsed or _make_parsed_answer(),
        "model_used": "llama-3.3-70b-versatile",
        "fallback_triggered": False,
        "ttft_ms": 234.5,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_components():
    """Mock all external dependencies and return patched classes."""
    with (
        patch("pipeline.EmbeddingEngine") as MockEmbed,
        patch("pipeline.QdrantIndexer") as MockQdrant,
        patch("pipeline.MongoDBIndexer") as MockMongo,
        patch("pipeline.FinancialRAGGenerator") as MockGen,
        patch("pipeline.AsyncGuardrail") as MockGuard,
        patch("pipeline.SemanticCache") as MockCache,
        patch("pipeline.mlflow.set_experiment"),
        patch("pipeline.mlflow.start_run"),
        patch("pipeline.mlflow.end_run"),
        patch("pipeline.mlflow.log_param"),
        patch("pipeline.mlflow.log_metric"),
    ):
        # EmbeddingEngine
        embed_instance = MockEmbed.return_value
        embed_instance.embed_single.return_value = [0.1] * 768

        # QdrantIndexer
        qdrant_instance = MockQdrant.return_value
        qdrant_instance.search.return_value = _make_qdrant_results(3)
        qdrant_instance.count_points.return_value = 110

        # MongoDBIndexer
        mongo_instance = MockMongo.return_value
        mongo_docs = _make_mongo_docs(_make_qdrant_results(3))
        mongo_instance.get_chunks_by_ids.return_value = mongo_docs
        mongo_instance.count_documents.return_value = 110

        # Generator
        gen_instance = MockGen.return_value
        gen_instance.generate.return_value = _make_gen_result()

        # SemanticCache
        cache_instance = MockCache.return_value
        cache_instance.get.return_value = None  # cache miss
        cache_instance.put.return_value = True

        # AsyncGuardrail
        guard_instance = MockGuard.return_value
        guard_instance.check.return_value = {
            "passed": True,
            "verified_claims": ["167.0"],
            "failed_claims": [],
            "final_output": "Answer verified.",
            "cache_written": True,
            "detail": "OK",
        }

        yield {
            "embed": embed_instance,
            "qdrant": qdrant_instance,
            "mongo": mongo_instance,
            "generator": gen_instance,
            "guardrail": guard_instance,
            "cache": cache_instance,
        }


@pytest.fixture
def pipeline(mock_components):
    """Create a FinancialRAGPipeline with all components mocked."""
    from pipeline import FinancialRAGPipeline

    p = FinancialRAGPipeline(
        qdrant_path="/tmp/test_qdrant",
        mongo_db="test_db",
        mongo_collection="test_collection",
        qdrant_collection="test_vectors",
        top_k=5,
        enable_cache=True,
        enable_guardrail=True,
    )
    return p


# ============================================================================
# TEST: Cache Hit
# ============================================================================

class TestCacheHit:
    """When cache has a hit, return cached answer without LLM call."""

    def test_cache_hit_returns_immediately(self, pipeline, mock_components):
        cached_entry = {
            "query_hash": "abc123",
            "answer_json": json.dumps({"answer": "Cached answer"}),
            "guardrail_passed": True,
            "timestamp_iso": "2026-01-01T00:00:00Z",
        }
        mock_components["cache"].get.return_value = cached_entry

        result = pipeline.query("What is Apple's revenue?", ticker="AAPL")

        assert result["cache_hit"] is True
        assert result["model_used"] == "cache"
        mock_components["generator"].generate.assert_not_called()

    def test_cache_hit_skips_retrieval(self, pipeline, mock_components):
        cached_entry = {"answer_json": "{}", "guardrail_passed": True}
        mock_components["cache"].get.return_value = cached_entry

        pipeline.query("What are Apple's segments?")

        mock_components["qdrant"].search.assert_not_called()
        mock_components["mongo"].get_chunks_by_ids.assert_not_called()


# ============================================================================
# TEST: Cache Miss → Full Pipeline
# ============================================================================

class TestCacheMiss:
    """When cache misses, execute the full retrieval → generation → guardrail flow."""

    def test_full_flow_returns_parsed_answer(self, pipeline, mock_components):
        result = pipeline.query("What are Apple's reportable segments?")

        assert result["parsed"] is not None
        assert result["model_used"] == "llama-3.3-70b-versatile"
        assert result["fallback_triggered"] is False
        assert result["cache_hit"] is False

    def test_qdrant_search_called_with_query(self, pipeline, mock_components):
        pipeline.query("What is Apple's revenue?")

        mock_components["qdrant"].search.assert_called_once()
        call_args = mock_components["qdrant"].search.call_args
        assert call_args[0][0] == [0.1] * 768  # query embedding
        assert call_args[1]["top_k"] == 5

    def test_mongo_enrichment_called(self, pipeline, mock_components):
        pipeline.query("What is Apple's revenue?")

        mock_components["mongo"].get_chunks_by_ids.assert_called_once()
        call_args = mock_components["mongo"].get_chunks_by_ids.call_args
        chunk_ids = call_args[0][0]
        assert len(chunk_ids) == 3
        assert all(cid.startswith("chunk_") for cid in chunk_ids)

    def test_generator_called_with_retrieved_docs(self, pipeline, mock_components):
        pipeline.query("What is Apple's revenue?")

        mock_components["generator"].generate.assert_called_once()
        call_kwargs = mock_components["generator"].generate.call_args[1]
        assert call_kwargs["query"] == "What is Apple's revenue?"
        assert len(call_kwargs["retrieved_docs"]) == 3
        assert call_kwargs["stream"] is False

    def test_guardrail_called_for_parsed_answer(self, pipeline, mock_components):
        pipeline.query("What is Apple's revenue?")

        mock_components["guardrail"].check.assert_called_once()
        call_kwargs = mock_components["guardrail"].check.call_args[1]
        assert "Apple" in call_kwargs["answer"]

    def test_pipeline_run_id_is_generated(self, pipeline, mock_components):
        result = pipeline.query("What is Apple's revenue?")
        assert "pipeline_run_id" in result
        assert len(result["pipeline_run_id"]) == 8

    def test_context_docs_have_text_and_metadata(self, pipeline, mock_components):
        pipeline.query("What is Apple's revenue?")

        call_kwargs = mock_components["generator"].generate.call_args[1]
        docs = call_kwargs["retrieved_docs"]
        for doc in docs:
            assert "text" in doc
            assert "metadata" in doc
            assert doc["metadata"]["ticker"] == "AAPL"
            assert doc["metadata"]["fiscal_year"] == "2025"
            assert len(doc["text"]) > 50


# ============================================================================
# TEST: Metadata Filtering
# ============================================================================

class TestMetadataFiltering:
    """Verify that ticker and fiscal_year filters are passed to Qdrant search."""

    def test_ticker_filter_passed_to_qdrant(self, pipeline, mock_components):
        pipeline.query("What is Apple's revenue?", ticker="AAPL")

        call_args = mock_components["qdrant"].search.call_args
        assert call_args[1]["ticker"] == "AAPL"

    def test_fiscal_year_filter_passed_to_qdrant(self, pipeline, mock_components):
        pipeline.query("What is Apple's revenue?", fiscal_year="2025")

        call_args = mock_components["qdrant"].search.call_args
        assert call_args[1]["fiscal_year"] == "2025"

    def test_both_filters_passed(self, pipeline, mock_components):
        pipeline.query("Revenue?", ticker="MSFT", fiscal_year="2024")

        call_args = mock_components["qdrant"].search.call_args
        assert call_args[1]["ticker"] == "MSFT"
        assert call_args[1]["fiscal_year"] == "2024"

    def test_no_filters_when_none(self, pipeline, mock_components):
        pipeline.query("Revenue?")

        call_args = mock_components["qdrant"].search.call_args
        assert call_args[1]["ticker"] is None
        assert call_args[1]["fiscal_year"] is None


# ============================================================================
# TEST: Empty Retrieval → Safe Fallback
# ============================================================================

class TestEmptyRetrieval:
    """When Qdrant returns 0 results, return safe fallback without LLM call."""

    def test_empty_qdrant_returns_fallback(self, pipeline, mock_components):
        mock_components["qdrant"].search.return_value = []

        result = pipeline.query("What is Apple's revenue?")

        assert result["parsed"] is None
        assert "not available" in result["raw_output"].lower()
        assert result["model_used"] == "none"
        mock_components["generator"].generate.assert_not_called()

    def test_empty_mongo_returns_fallback(self, pipeline, mock_components):
        mock_components["mongo"].get_chunks_by_ids.return_value = {}

        result = pipeline.query("What is Apple's revenue?")

        assert result["parsed"] is None
        assert "not available" in result["raw_output"].lower()


# ============================================================================
# TEST: Generator Returns None (LLM Failure)
# ============================================================================

class TestGeneratorFailure:
    """When LLM returns None (both models exhausted), pipeline returns safe fallback."""

    def test_none_parsed_returns_fallback(self, pipeline, mock_components):
        mock_components["generator"].generate.return_value = {
            "raw_output": "The requested financial information is not available in the provided reports.",
            "parsed": None,
            "model_used": "none",
            "fallback_triggered": True,
            "ttft_ms": 0,
        }

        result = pipeline.query("Revenue?")

        assert result["parsed"] is None
        assert result["model_used"] == "none"
        assert result["fallback_triggered"] is True


# ============================================================================
# TEST: Streaming
# ============================================================================

class TestStreaming:
    """Verify streaming yields tokens correctly."""

    def test_stream_returns_async_iterator(self, pipeline, mock_components):
        async def mock_stream(query, retrieved_docs):
            yield "token1"
            yield "token2"
            yield "token3"

        mock_components["generator"].stream_tokens = mock_stream

        async def run_stream():
            tokens = []
            async for token in pipeline.query_stream("Revenue?"):
                tokens.append(token)
            return tokens

        tokens = __import__("asyncio").run(run_stream())
        assert tokens == ["token1", "token2", "token3"]

    def test_stream_with_empty_docs(self, pipeline, mock_components):
        mock_components["qdrant"].search.return_value = []

        async def run_stream():
            tokens = []
            async for token in pipeline.query_stream("Revenue?"):
                tokens.append(token)
            return tokens

        tokens = __import__("asyncio").run(run_stream())
        assert len(tokens) == 1
        assert "not available" in tokens[0].lower()


# ============================================================================
# TEST: MLflow Logging
# ============================================================================

class TestMLflowLogging:
    """Verify MLflow is called with pipeline metrics."""

    def test_mlflow_experiment_set(self, pipeline, mock_components):
        pipeline.query("Revenue?")

        import mlflow
        mlflow.set_experiment.assert_called_with("financial_rag_pipeline")

    def test_mlflow_metrics_logged(self, pipeline, mock_components):
        pipeline.query("Revenue?")

        import mlflow
        mlflow.log_metric.assert_any_call("documents_retrieved", 3)
        mlflow.log_metric.assert_any_call("cache_hit", 0)
        mlflow.log_metric.assert_any_call("fallback_triggered", 0)

    def test_mlflow_params_logged(self, pipeline, mock_components):
        pipeline.query("Revenue?", ticker="AAPL", fiscal_year="2025")

        import mlflow
        mlflow.log_param.assert_any_call("ticker", "AAPL")
        mlflow.log_param.assert_any_call("fiscal_year", "2025")

    def test_mlflow_run_ended(self, pipeline, mock_components):
        pipeline.query("Revenue?")

        import mlflow
        mlflow.end_run.assert_called()


# ============================================================================
# TEST: Stats
# ============================================================================

class TestStats:
    """Verify get_stats returns correct counts."""

    def test_stats_returns_qdrant_and_mongo(self, pipeline, mock_components):
        stats = pipeline.get_stats()
        assert stats["qdrant_total"] == 110
        assert stats["mongo_total"] == 110


# ============================================================================
# TEST: Memory Reset
# ============================================================================

class TestMemoryReset:
    """Verify reset_memory delegates to generator."""

    def test_reset_memory_calls_generator(self, pipeline, mock_components):
        pipeline.reset_memory()
        mock_components["generator"].reset_memory.assert_called_once()
