"""
End-to-End Integration Test: Module 1 (Ingestion) + Module 2 (Generation).

Connects the full pipeline:
  SEC Filing -> Parser -> Cleaner -> Chunker -> Qdrant/MongoDB Indexing
  -> Vector Query -> XML Context Formatting -> Groq LLM Generation
  -> Pydantic Validation -> Async Guardrail -> MLflow

Uses a real AAPL 10-K filing and live Groq API (temperature=0.0).
Tests TWO scenarios:
  A) Answerable question (narrative data in context) -> full JSON validation path
  B) Unanswerable question (no exact data in context) -> guardrail "not available" path
"""

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_ingestion"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "2_generation"))

from config.logging_config import get_logger
from config.settings import DATA_DIR

logger = get_logger("test.e2e_integration")

from html_table_parser import parse_sec_filing
from cleaning import clean_financial_text
from metadata_extractor import extract_metadata
from hybrid_chunker import chunk_document
from database_indexer import EmbeddingEngine, MongoDBIndexer, QdrantIndexer
from schemas import ConsolidatedFinancialAnswer
from generator import FinancialRAGGenerator, format_context_xml
from async_guardrail import AsyncGuardrail

_TEST_FILE = DATA_DIR / "AAPL" / "10-K" / "0000320193-25-000079" / "full-submission.txt"
_QDRANT_PATH = str(DATA_DIR / "qdrant_db_e2e_gen_test")
_MONGO_DB = "financial_rag_e2e_gen_test"
_MONGO_COLLECTION = "raw_chunks_e2e_gen"

_QUERY_ANSWERABLE = "What are Apple's reportable business segments and how did Americas perform in fiscal year 2025?"
_QUERY_UNANSWERABLE = "What was Apple's exact total net revenue figure for fiscal year 2025 in dollars?"


def _make_mock_mongo(chunks=None):
    """Create an in-memory mock MongoDB indexer for offline testing."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    _chunks = chunks or []
    mock.upsert_chunks.return_value = len(_chunks)
    mock.count_documents.return_value = len(_chunks)
    mock.get_chunk.side_effect = lambda cid: next(
        (c for c in _chunks if c.get("chunk_id") == cid), None
    )
    mock.get_chunks_by_ids.side_effect = lambda cids: {
        c["chunk_id"]: {"raw_text": c["text"]}
        for c in _chunks if c.get("chunk_id") in cids
    }
    mock.close.return_value = None
    return mock


# ============================================================================
# SESSION-SCOPED FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def ingestion_indexers():
    """Run the full Module 1 ingestion pipeline once for the entire test session."""
    import torch

    assert _TEST_FILE.exists(), f"Test file not found: {_TEST_FILE}"

    # Parse
    parsed = parse_sec_filing(_TEST_FILE)
    assert len(parsed["text_content"]) > 0, "Parser returned empty text"

    # Clean
    cleaned = clean_financial_text(parsed["text_content"])

    # Metadata
    base_metadata = extract_metadata(
        file_path=_TEST_FILE,
        content=parsed["raw_html"],
        chunk_text=cleaned[:2000],
        chunk_index=0,
    )
    assert base_metadata["ticker"] == "AAPL"

    # Chunk
    chunks = chunk_document(
        text=cleaned,
        tables=parsed["tables"],
        file_path=_TEST_FILE,
        metadata_base=base_metadata,
    )
    assert len(chunks) > 0, "Chunker produced zero chunks"

    # Clean prior test data
    if os.path.exists(_QDRANT_PATH):
        shutil.rmtree(_QDRANT_PATH)

    # Embed
    embedding_engine = EmbeddingEngine()
    qdrant_indexer = QdrantIndexer(path=_QDRANT_PATH)

    # MongoDB with graceful fallback
    import pymongo
    _mongo_available = False
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        client[_MONGO_DB][_MONGO_COLLECTION].drop()
        client.close()
        _mongo_available = True
    except Exception:
        pass

    if _mongo_available:
        mongo_indexer = MongoDBIndexer(db_name=_MONGO_DB, collection_name=_MONGO_COLLECTION)
        logger.info("MongoDB available — using real indexer")
    else:
        mongo_indexer = _make_mock_mongo(chunks=chunks)
        logger.info("MongoDB not available — using in-memory mock")

    all_embeddings = []
    batch_size = 8
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_embs = embedding_engine.embed([c["text"] for c in batch], batch_size=batch_size)
        all_embeddings.extend(batch_embs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert len(all_embeddings) == len(chunks)

    # Index
    qdrant_count = qdrant_indexer.upsert_vectors(chunks, all_embeddings)
    assert qdrant_count == len(chunks)
    mongo_count = mongo_indexer.upsert_chunks(chunks)

    logger.info(
        "Ingestion complete: %d chunks, %d vectors, %d mongo docs",
        len(chunks), qdrant_count, mongo_count,
    )

    yield qdrant_indexer, mongo_indexer

    # Teardown
    qdrant_indexer.close()
    if _mongo_available:
        mongo_indexer.close()
    if os.path.exists(_QDRANT_PATH):
        try:
            shutil.rmtree(_QDRANT_PATH)
        except PermissionError:
            pass


@pytest.fixture(scope="session")
def answerable_docs(ingestion_indexers):
    """Retrieve and format documents for the answerable query."""
    qdrant_indexer, mongo_indexer = ingestion_indexers
    return _retrieve_documents(qdrant_indexer, mongo_indexer, _QUERY_ANSWERABLE)


@pytest.fixture(scope="session")
def unanswerable_docs(ingestion_indexers):
    """Retrieve and format documents for the unanswerable query."""
    qdrant_indexer, mongo_indexer = ingestion_indexers
    return _retrieve_documents(qdrant_indexer, mongo_indexer, _QUERY_UNANSWERABLE)


def _retrieve_documents(qdrant_indexer, mongo_indexer, query, top_k=5):
    """Retrieve documents from Qdrant + MongoDB for a query."""
    embedding_engine = EmbeddingEngine()
    query_embedding = embedding_engine.embed_single(query)

    qdrant_results = qdrant_indexer.search(query_embedding, top_k=top_k, ticker="AAPL")
    assert len(qdrant_results) > 0, f"Vector search returned no results for: {query}"

    chunk_ids = [r["chunk_id"] for r in qdrant_results]
    mongo_docs = mongo_indexer.get_chunks_by_ids(chunk_ids)

    documents = []
    for r in qdrant_results:
        cid = r["chunk_id"]
        meta = r.get("payload", {})
        mongo_doc = mongo_docs.get(cid, {})
        doc_text = mongo_doc.get("raw_text", "")
        assert len(doc_text) > 100, f"Chunk {cid} too short ({len(doc_text)} chars)"
        documents.append({
            "text": doc_text,
            "metadata": {
                "ticker": meta.get("ticker", "UNKNOWN"),
                "fiscal_year": meta.get("fiscal_year", "UNKNOWN"),
                "section": meta.get("section", "General"),
                "contains_table": meta.get("contains_table", False),
                "page_number": meta.get("page_number", "N/A"),
            },
            "chunk_id": cid,
            "score": r["score"],
        })

    return documents


@pytest.fixture(scope="session")
def answerable_result(answerable_docs):
    """Run Groq LLM generation for the answerable query."""
    generator = FinancialRAGGenerator()
    return generator.generate(query=_QUERY_ANSWERABLE, retrieved_docs=answerable_docs, stream=False)


@pytest.fixture(scope="session")
def unanswerable_result(unanswerable_docs):
    """Run Groq LLM generation for the unanswerable query."""
    generator = FinancialRAGGenerator()
    return generator.generate(query=_QUERY_UNANSWERABLE, retrieved_docs=unanswerable_docs, stream=False)


# ============================================================================
# PHASE 1: INGESTION
# ============================================================================

class TestPhase1Ingestion:
    """Module 1: Parse, embed, and index the AAPL 10-K filing."""

    def test_ingestion_completes(self, ingestion_indexers):
        qdrant_indexer, mongo_indexer = ingestion_indexers
        assert qdrant_indexer is not None
        assert mongo_indexer is not None

    def test_qdrant_vector_count(self, ingestion_indexers):
        qdrant_indexer, _ = ingestion_indexers
        count = qdrant_indexer.count_points(ticker="AAPL")
        assert count == 110, f"Expected 110 vectors, got {count}"

    def test_mongo_chunk_count(self, ingestion_indexers):
        _, mongo_indexer = ingestion_indexers
        count = mongo_indexer.count_documents(ticker="AAPL")
        assert count == 110, f"Expected 110 chunks, got {count}"


# ============================================================================
# PHASE 2: ANSWERABLE QUESTION (Happy Path)
# ============================================================================

class TestPhase2Answerable:
    """Test the full generation pipeline with a question answerable from narrative context."""

    def test_vector_retrieval(self, answerable_docs):
        assert len(answerable_docs) == 5
        for doc in answerable_docs:
            assert doc["metadata"]["ticker"] == "AAPL"
            assert doc["metadata"]["fiscal_year"] == "2025"
            assert len(doc["text"]) > 100

    def test_context_xml_formatting(self, answerable_docs):
        context_xml = format_context_xml(answerable_docs)
        assert "<CONTEXT>" in context_xml
        assert "</CONTEXT>" in context_xml
        assert 'ticker="AAPL"' in context_xml
        assert len(context_xml) > 2000, f"Context too short ({len(context_xml)} chars)"

    def test_llm_returns_valid_json(self, answerable_result):
        assert answerable_result["parsed"] is not None, (
            f"LLM output failed validation. Raw:\n{answerable_result['raw_output'][:500]}"
        )
        assert answerable_result["model_used"] == "llama-3.3-70b-versatile"
        assert answerable_result["fallback_triggered"] is False

    def test_schema_validation(self, answerable_result):
        parsed = answerable_result["parsed"]
        assert isinstance(parsed, ConsolidatedFinancialAnswer)
        assert len(parsed.internal_thought.strip()) > 0
        assert len(parsed.extracted_raw_data.strip()) > 0
        assert len(parsed.answer.strip()) > 0
        assert len(parsed.sources) > 0

    def test_answer_contains_segments(self, answerable_result):
        parsed = answerable_result["parsed"]
        answer_lower = parsed.answer.lower()
        assert "americas" in answer_lower, f"Answer should mention Americas: {parsed.answer[:200]}"
        assert any(seg in answer_lower for seg in ["europe", "greater china", "japan", "asia"]), (
            f"Answer should mention business segments: {parsed.answer[:200]}"
        )

    def test_guardrail_runs(self, answerable_result):
        guardrail = AsyncGuardrail()
        try:
            verdict = asyncio.get_event_loop().run_until_complete(
                guardrail.check(
                    query=_QUERY_ANSWERABLE,
                    answer=answerable_result["raw_output"],
                    extracted_raw_data=answerable_result["parsed"].extracted_raw_data,
                    parsed_output=answerable_result["parsed"],
                )
            )
            assert "passed" in verdict
            assert "verified_claims" in verdict
            assert "failed_claims" in verdict
        finally:
            guardrail.close()


# ============================================================================
# PHASE 3: UNANSWERABLE QUESTION (Guardrail Path)
# ============================================================================

class TestPhase3Unanswerable:
    """Test that the LLM correctly returns 'not available' when data is absent."""

    def test_llm_returns_not_available(self, unanswerable_result):
        raw = unanswerable_result["raw_output"].strip().lower()
        assert "not available" in raw or "cannot be determined" in raw or "not found" in raw, (
            f"Expected 'not available' response, got: {unanswerable_result['raw_output'][:200]}"
        )

    def test_parsed_is_consolidated_answer(self, unanswerable_result):
        parsed = unanswerable_result["parsed"]
        from schemas import ConsolidatedFinancialAnswer
        assert isinstance(parsed, ConsolidatedFinancialAnswer), (
            f"Expected ConsolidatedFinancialAnswer, got {type(parsed)}"
        )
        assert "not available" in parsed.answer.lower()

    def test_model_used(self, unanswerable_result):
        assert unanswerable_result["model_used"] in ("llama-3.3-70b-versatile", "qwen/qwen3.6-27b")

    def test_guardrail_passes_on_not_available(self, unanswerable_result):
        guardrail = AsyncGuardrail()
        try:
            verdict = asyncio.get_event_loop().run_until_complete(
                guardrail.check(
                    query=_QUERY_UNANSWERABLE,
                    answer=unanswerable_result["raw_output"],
                    extracted_raw_data=unanswerable_result["raw_output"],
                    parsed_output=None,
                )
            )
            assert verdict["passed"] is True, f"Guardrail should pass on 'not available': {verdict}"
        finally:
            guardrail.close()


# ============================================================================
# PHASE 4: MLFLOW
# ============================================================================

class TestPhase4MLflow:
    """Verify MLflow experiment tracking."""

    def test_mlflow_experiment_exists(self):
        import mlflow
        experiment = mlflow.get_experiment_by_name("financial_rag_generation")
        assert experiment is not None, "MLflow experiment 'financial_rag_generation' not found"

    def test_mlflow_has_runs(self):
        import mlflow
        experiment = mlflow.get_experiment_by_name("financial_rag_generation")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0, "No MLflow runs found"
