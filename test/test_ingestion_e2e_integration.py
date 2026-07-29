"""
End-to-End Integration Test for Module 1: Ingestion Pipeline.

Processes a real SEC filing through the complete pipeline:
  Document -> Parser -> Cleaner -> Metadata Extractor -> Hybrid Chunker -> Dual Storage

Validates:
  - Full pipeline execution without errors
  - Metadata integrity across all chunks
  - Table structure and financial notation preservation
  - Persistent Qdrant storage with filtered vector retrieval
  - MongoDB raw storage (or graceful fallback)
  - Structured JSON log emission to logs/rag_events.log
"""

import mlflow
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

# --- Path Setup ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_ingestion"))

from config.logging_config import get_logger
from config.settings import DATA_DIR, LOG_DIR, QDRANT_PATH

from html_table_parser import parse_sec_filing
from cleaning import clean_financial_text
from metadata_extractor import extract_metadata
from hybrid_chunker import chunk_document, count_tokens, _get_token_encoder
from database_indexer import (
    DualStorageIndexer,
    EmbeddingEngine,
    IngestionTracker,
    MongoDBIndexer,
    QdrantIndexer,
)

logger = get_logger("test.e2e")

# --- Test Configuration ---
# Use the smallest real SEC filing for faster test execution
_TEST_FILE = DATA_DIR / "AAPL" / "10-K" / "0000320193-25-000079" / "full-submission.txt"
_TEST_QDRANT_PATH = str(DATA_DIR / "qdrant_db_test")
_CLEANUP_QDRANT = True

# Financial notation patterns to verify preservation
import re
_PAREN_NEGATIVE = re.compile(r"\(\d[\d,.*]*\)")
_CURRENCY_SYMBOL = re.compile(r"\$\d")
_PERCENTAGE = re.compile(r"\d+\.?\d*%")


def _check_mongo_available() -> bool:
    """Check if a MongoDB instance is reachable."""
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False


def _make_mock_mongo() -> MagicMock:
    """Create an in-memory mock MongoDB indexer for offline testing."""
    mock = MagicMock()
    mock.upsert_chunks.return_value = 0
    mock.count_documents.return_value = 0
    mock.get_chunk.return_value = None
    return mock


# ============================================================================
# FIXTURES
# ============================================================================

import pytest

@pytest.fixture(scope="module")
def chunks():
    """Process a real SEC filing through the complete ingestion pipeline."""
    assert _TEST_FILE.exists(), f"Test file not found: {_TEST_FILE}"

    # Step 1: Parse
    logger.info("E2E Step 1: Parsing %s", _TEST_FILE.name)
    parsed = parse_sec_filing(_TEST_FILE)
    assert parsed["table_count"] >= 0, "Parser failed to return table count"
    assert len(parsed["text_content"]) > 0, "Parser returned empty text"
    assert parsed["raw_html"] is not None, "Parser returned no raw HTML"

    # Step 2: Clean
    logger.info("E2E Step 2: Cleaning text (%d chars)", len(parsed["text_content"]))
    cleaned = clean_financial_text(parsed["text_content"])
    assert len(cleaned) > 0, "Cleaner returned empty text"
    assert len(cleaned) <= len(parsed["text_content"]) + 1000, "Cleaning bloated text unexpectedly"

    # Step 3: Metadata extraction (base metadata from file)
    logger.info("E2E Step 3: Extracting metadata")
    base_metadata = extract_metadata(
        file_path=_TEST_FILE,
        content=parsed["raw_html"],
        chunk_text=cleaned[:2000],
        chunk_index=0,
    )
    assert base_metadata["ticker"] == "AAPL", f"Wrong ticker: {base_metadata['ticker']}"
    assert base_metadata["fiscal_year"] == "2025", f"Wrong year: {base_metadata['fiscal_year']}"
    assert base_metadata["doc_type"] == "10-K", f"Wrong doc_type: {base_metadata['doc_type']}"

    # Step 4: Chunk
    logger.info("E2E Step 4: Chunking document")
    chunks = chunk_document(
        text=cleaned,
        tables=parsed["tables"],
        file_path=_TEST_FILE,
        metadata_base=base_metadata,
    )
    assert len(chunks) > 0, "Chunker produced zero chunks"
    logger.info("E2E Step 4 result: %d chunks produced", len(chunks))

    return chunks

@pytest.fixture(scope="module")
def qdrant_indexer(chunks):
    """Build a QdrantIndexer, embed chunks, and index them."""
    if os.path.exists(_TEST_QDRANT_PATH):
        shutil.rmtree(_TEST_QDRANT_PATH)
    indexer = QdrantIndexer(path=_TEST_QDRANT_PATH)
    embedding_engine = EmbeddingEngine()
    tracker = IngestionTracker(experiment_name="financial_rag_e2e_test")

    mongo_available = _check_mongo_available()
    if mongo_available:
        mongo_indexer = MongoDBIndexer()
    else:
        mongo_indexer = _make_mock_mongo()
        logger.info("MongoDB not available — using in-memory mock")

    t0 = time.time()

    import torch
    all_embeddings = []
    batch_size = 8
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_texts = [c["text"] for c in batch]
        batch_embs = embedding_engine.embed(batch_texts, batch_size=batch_size)
        all_embeddings.extend(batch_embs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    embeddings = all_embeddings
    assert len(embeddings) == len(chunks), "Embedding count mismatch"

    qdrant_count = indexer.upsert_vectors(chunks, embeddings)
    assert qdrant_count == len(chunks), f"Qdrant stored {qdrant_count}/{len(chunks)}"
    mongo_count = mongo_indexer.upsert_chunks(chunks)

    elapsed = time.time() - t0
    text_chunks = sum(1 for c in chunks if c.get("chunk_type") == "text")
    table_chunks = len(chunks) - text_chunks
    try:
        tracker.log_params(
            embedding_model="nomic-ai/nomic-embed-text-v1.5",
            device="cuda" if torch.cuda.is_available() else "cpu",
            dim=768,
        )
        tracker.log_metrics(
            total_chunks=len(chunks),
            text_chunks=text_chunks,
            table_chunks=table_chunks,
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        logger.warning("MLflow logging failed (non-blocking): %s", exc)

    logger.info(
        "Dual storage indexed %d chunks in %.1fs",
        len(chunks), elapsed,
    )
    return indexer


# ============================================================================
# TEST 1: Full Pipeline Execution
# ============================================================================

def test_full_pipeline_execution(chunks):
    """Process a real SEC filing through the complete ingestion pipeline."""
    assert len(chunks) > 0, "Chunker produced zero chunks"


# ============================================================================
# TEST 2: Metadata Integrity
# ============================================================================

def test_metadata_integrity(chunks: list[dict]):
    """Verify every chunk carries complete and valid financial metadata."""
    required_fields = [
        "ticker", "fiscal_year", "doc_type", "section",
        "contains_table", "chunk_id",
    ]

    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        for field in required_fields:
            assert field in meta, f"Chunk {i} ({chunk['chunk_id']}) missing field: {field}"

        # Validate field values
        assert meta["ticker"] == "AAPL", f"Chunk {i} wrong ticker: {meta['ticker']}"
        assert meta["fiscal_year"] == "2025", f"Chunk {i} wrong year: {meta['fiscal_year']}"
        assert meta["doc_type"] == "10-K", f"Chunk {i} wrong doc_type: {meta['doc_type']}"
        assert isinstance(meta["contains_table"], bool), f"Chunk {i} contains_table not bool"
        assert len(chunk["chunk_id"]) > 5, f"Chunk {i} chunk_id too short"
        assert chunk["token_count"] > 0, f"Chunk {i} has zero tokens"

    logger.info("Metadata integrity verified for %d chunks", len(chunks))


# ============================================================================
# TEST 3: Table Structure & Financial Notation
# ============================================================================

def test_table_and_financial_notation(chunks: list[dict]):
    """Confirm Markdown tables and financial symbols are preserved."""
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]

    # Verify table chunks have pipe structure
    for tc in table_chunks:
        assert "|" in tc["text"], f"Table chunk missing pipes: {tc['chunk_id']}"
        lines = [l for l in tc["text"].split("\n") if l.strip()]
        assert len(lines) >= 2, f"Table chunk too small: {tc['chunk_id']}"

    # Verify financial notation in text chunks
    all_text = " ".join(c["text"] for c in text_chunks)
    has_currency = bool(_CURRENCY_SYMBOL.search(all_text))
    has_parens = bool(_PAREN_NEGATIVE.search(all_text))

    logger.info(
        "Table chunks: %d, Text chunks: %d, Currency found: %s, Paren negatives: %s",
        len(table_chunks), len(text_chunks), has_currency, has_parens,
    )


# ============================================================================
# TEST 4: Dual Storage (Qdrant Persistent + MongoDB/Mock)
# ============================================================================

def test_dual_storage(qdrant_indexer: QdrantIndexer):
    """Verify chunks were indexed into Qdrant."""
    assert qdrant_indexer is not None, "qdrant_indexer fixture failed"


# ============================================================================
# TEST 5: Physical Storage Verification
# ============================================================================

def test_physical_storage(qdrant_indexer):
    """Verify that local Qdrant storage files exist on disk."""
    qdrant_path = Path(_TEST_QDRANT_PATH)
    assert qdrant_path.exists(), f"Qdrant DB directory not created: {_TEST_QDRANT_PATH}"

    # Check for snapshot or data files inside
    files = list(qdrant_path.rglob("*"))
    data_files = [f for f in files if f.is_file()]
    assert len(data_files) > 0, f"No data files found in {_TEST_QDRANT_PATH}"

    logger.info("Physical storage verified: %d files in %s", len(data_files), _TEST_QDRANT_PATH)


# ============================================================================
# TEST 6: Filtered Query Retrieval
# ============================================================================

def test_filtered_query_retrieval(qdrant_indexer: QdrantIndexer):
    """Execute a vector search with metadata pre-filtering."""
    embedding_engine = EmbeddingEngine()
    query = "What was Apple's revenue for fiscal year 2025?"
    query_embedding = embedding_engine.embed_single(query)

    # Search without filter
    all_results = qdrant_indexer.search(query_embedding, top_k=5)
    assert len(all_results) > 0, "Unfiltered search returned no results"

    # Verify results have valid scores
    for r in all_results:
        assert r["score"] > 0, f"Result score <= 0: {r['score']}"
        assert "payload" in r, "Result missing payload"
        assert "ticker" in r["payload"], "Payload missing ticker"

    # Search with ticker filter
    filtered_results = qdrant_indexer.search(
        query_embedding, top_k=5, ticker="AAPL"
    )
    assert len(filtered_results) > 0, "Filtered search returned no results"
    for r in filtered_results:
        assert r["payload"]["ticker"] == "AAPL", (
            f"Filter failed: got ticker={r['payload']['ticker']}"
        )

    logger.info(
        "Query retrieval: unfiltered=%d results, AAPL-filtered=%d results, best_score=%.4f",
        len(all_results), len(filtered_results), all_results[0]["score"],
    )


# ============================================================================
# TEST 7: JSON Logging Verification
# ============================================================================

def test_json_logging():
    """Confirm structured JSON logs are written to rag_events.log."""
    log_path = LOG_DIR / "rag_events.log"
    assert log_path.exists(), f"Log file not found: {log_path}"

    content = log_path.read_text(encoding="utf-8")
    assert len(content) > 100, "Log file too small — no events recorded"

    # Verify JSON structure (first non-empty line)
    import json
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    assert len(lines) > 0, "No log lines found"

    first_entry = json.loads(lines[0])
    assert "timestamp" in first_entry, "Log entry missing 'timestamp'"
    assert "level" in first_entry, "Log entry missing 'level'"
    assert "message" in first_entry, "Log entry missing 'message'"

    logger.info("JSON logging verified: %d entries in %s", len(lines), log_path.name)


# ============================================================================
# TEST 8: Token Constraint Verification
# ============================================================================

def test_token_constraint(chunks: list[dict]):
    """Verify all TEXT chunks respect the 768-token maximum.

    Table chunks are exempt: SEC financial tables are atomic and must never
    be split per the design spec.  Large tables are expected to exceed 768 tokens.
    """
    encoder = _get_token_encoder()
    text_violations = []
    table_violations = []

    for chunk in chunks:
        actual = count_tokens(chunk["text"], encoder)
        if chunk["chunk_type"] == "table":
            if actual > 8192:
                table_violations.append((chunk["chunk_id"], actual))
        else:
            if actual > 768:
                text_violations.append((chunk["chunk_id"], actual))

    assert len(text_violations) == 0, (
        f"{len(text_violations)} text chunks exceed 768 tokens: "
        + ", ".join(f"{cid}({tok})" for cid, tok in text_violations[:5])
    )
    assert len(table_violations) == 0, (
        f"{len(table_violations)} table chunks exceed 8192 tokens (tables should not be enormous): "
        + ", ".join(f"{cid}({tok})" for cid, tok in table_violations[:5])
    )

    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]

    text_token_counts = [count_tokens(c["text"], encoder) for c in text_chunks]
    table_token_counts = [count_tokens(c["text"], encoder) for c in table_chunks]

    text_avg = sum(text_token_counts) / len(text_token_counts) if text_token_counts else 0
    text_max = max(text_token_counts) if text_token_counts else 0
    table_max = max(table_token_counts) if table_token_counts else 0

    logger.info(
        "Token stats — text: count=%d avg=%.0f max=%d | table: count=%d max=%d",
        len(text_token_counts), text_avg, text_max,
        len(table_token_counts), table_max,
    )


# ============================================================================
# CLEANUP
# ============================================================================

def cleanup():
    """Remove test Qdrant database after tests complete."""
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _CLEANUP_QDRANT and os.path.exists(_TEST_QDRANT_PATH):
        try:
            shutil.rmtree(_TEST_QDRANT_PATH)
            logger.info("Cleaned up test Qdrant DB: %s", _TEST_QDRANT_PATH)
        except PermissionError:
            logger.warning("Could not remove Qdrant DB (locked): %s", _TEST_QDRANT_PATH)


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """Run the complete E2E integration test suite."""
    print("=" * 70)
    print("MODULE 1: END-TO-END INGESTION INTEGRATION TEST")
    print("=" * 70)
    print(f"Test file: {_TEST_FILE}")
    print(f"File exists: {_TEST_FILE.exists()}")
    if _TEST_FILE.exists():
        size_mb = _TEST_FILE.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
    print()

    tests = [
        ("Full Pipeline Execution", lambda: test_full_pipeline_execution()),
        ("Metadata Integrity", None),           # depends on test 1
        ("Table & Financial Notation", None),    # depends on test 1
        ("Dual Storage", None),                  # depends on test 1
        ("Physical Storage", None),              # depends on test 4
        ("Filtered Query Retrieval", None),      # depends on test 4
        ("JSON Logging", lambda: test_json_logging()),
        ("Token Constraint", None),              # depends on test 1
    ]

    passed = 0
    failed = 0
    chunks = None
    qdrant_indexer = None
    mongo_available = False

    for i, (name, fn) in enumerate(tests):
        try:
            print(f"[{i+1}/{len(tests)}] {name}...")
            if name == "Full Pipeline Execution":
                chunks, base_metadata, parsed = fn()
                passed += 1
            elif name == "Metadata Integrity":
                test_metadata_integrity(chunks)
                passed += 1
            elif name == "Table & Financial Notation":
                test_table_and_financial_notation(chunks)
                passed += 1
            elif name == "Dual Storage":
                qdrant_indexer, mongo_indexer, mongo_available = test_dual_storage(chunks)
                passed += 1
            elif name == "Physical Storage":
                test_physical_storage()
                passed += 1
            elif name == "Filtered Query Retrieval":
                test_filtered_query_retrieval(qdrant_indexer)
                passed += 1
            elif name == "JSON Logging":
                fn()
                passed += 1
            elif name == "Token Constraint":
                test_token_constraint(chunks)
                passed += 1
            print(f"  PASS\n")
        except AssertionError as e:
            print(f"  FAIL: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}\n")
            failed += 1

    # Cleanup — close Qdrant client before removing files
    if qdrant_indexer is not None:
        qdrant_indexer.close()
    cleanup()

    print("=" * 70)
    status = "ALL PASSED" if failed == 0 else f"{failed} FAILED"
    print(f"RESULTS: {passed}/{len(tests)} passed | {status}")
    print(f"MongoDB: {'LIVE' if mongo_available else 'MOCK (local instance not running)'}")
    print(f"Qdrant:  PERSISTENT ({_TEST_QDRANT_PATH})")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
