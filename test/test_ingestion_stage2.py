"""
Verification tests for Ingestion Stage 2: Hybrid Chunking & Dual-Storage Indexing.

Tests cover:
  - Table atomic preservation (no table split into half chunks)
  - Token size constraints (strictly <= 768 tokens per chunk)
  - Section boundary detection and chunk metadata
  - Embedding engine dimension validation
  - Dual-storage insertion (in-memory Qdrant + mocked MongoDB)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from qdrant_client import QdrantClient

# Add project root and ingestion module path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_ingestion"))

from hybrid_chunker import (
    chunk_document,
    count_tokens,
    _split_on_sections,
    _extract_table_chunks,
    _recursive_token_split,
    _get_token_encoder,
)
from database_indexer import (
    EmbeddingEngine,
    DualStorageIndexer,
    MongoDBIndexer,
    QdrantIndexer,
)
from metadata_extractor import extract_metadata


# ============================================================================
# TEST DATA
# ============================================================================

SAMPLE_TABLE = """| Line Item | 2025 | 2024 | 2023 |
| --- | --- | --- | --- |
| Net Sales | $395,760 | $383,285 | $383,285 |
| Cost of Sales | (211,848) | (214,137) | (214,137) |
| Gross Margin | $183,912 | $169,148 | $169,148 |
| Operating Expenses | (54,847) | (51,345) | (51,345) |
| Operating Income | $129,065 | $117,803 | $117,803 |"""

SAMPLE_SECTION_TEXT = """Item 1A. Risk Factors

The Company faces various risks related to global economic conditions, including inflation,
supply chain disruptions, and geopolitical tensions. These risks could materially impact
the Company's operating results, financial condition, and cash flows. The Company maintains
a diversified product portfolio and geographic presence to mitigate concentration risk.

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations

Revenue for fiscal year 2025 increased by 3.3% to $395.8 billion compared to $383.3 billion
in fiscal year 2024. The increase was primarily driven by strong Services revenue growth
of 13.2% year over year, partially offset by a modest decline in iPhone revenue.

Product revenue decreased 1.2% to $290.2 billion, driven by lower iPhone and Mac sales,
partially offset by growth in iPad and Wearables. Services revenue reached $105.6 billion,
an increase of 13.2%, driven by higher App Store, advertising, and cloud services revenue.

The Company's gross margin was 46.5% compared to 44.1% in the prior year, reflecting
a favorable product mix shift toward higher-margin Services. Operating expenses were
$54.8 billion, an increase of 6.8% driven by higher R&D investment.

Item 8. Financial Statements and Supplementary Data

The consolidated financial statements include the statements of operations, balance sheets,
cash flows, and equity for the fiscal years ended September 27, 2025 and September 28, 2024."""

SAMPLE_LONG_TEXT = (
    "Apple Inc. designs, manufactures, and markets smartphones, personal computers, "
    "tablets, wearables, and accessories worldwide. The company offers iPhone, Mac, "
    "iPad, and wearables, home, and accessories products. It also provides AppleCare "
    "support and cloud services. The company was founded in 1976 and is headquartered "
    "in Cupertino, California. " * 50
)


# ============================================================================
# TEST FUNCTIONS — CHUNKER
# ============================================================================


def test_token_counting():
    """Verify exact token counting with tiktoken."""
    encoder = _get_token_encoder()
    text = "Apple Inc. reported revenue of $395.8 billion for fiscal year 2025."
    token_count = count_tokens(text, encoder)
    assert 10 < token_count < 30, f"Token count {token_count} seems wrong for this text"
    print("  [PASS] Token counting accuracy")


def test_table_atomic_preservation():
    """Verify tables are never split — each table becomes one atomic chunk."""
    metadata = extract_metadata(
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        chunk_text=SAMPLE_TABLE,
    )

    chunks = chunk_document(
        text="Some preamble text.\n\n" + SAMPLE_TABLE + "\n\nSome trailing text.",
        tables=[],
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        metadata_base=metadata,
    )

    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) >= 1, f"Expected at least 1 table chunk, got {len(table_chunks)}"

    for tc in table_chunks:
        assert "|" in tc["text"], f"Table chunk missing pipe chars: {tc['text'][:100]}"
        assert tc["metadata"]["contains_table"] is True, "Table chunk missing contains_table=True"
        # Table should NOT be truncated — verify all rows present
        assert "Net Sales" in tc["text"], "Table chunk missing data rows"
        assert "Operating Income" in tc["text"], "Table chunk truncated"

    print("  [PASS] Table atomic preservation")


def test_token_size_constraint():
    """Verify all text chunks are strictly <= 768 tokens."""
    metadata = extract_metadata(
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        chunk_text=SAMPLE_LONG_TEXT,
    )

    chunks = chunk_document(
        text=SAMPLE_LONG_TEXT,
        tables=[],
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        metadata_base=metadata,
    )

    encoder = _get_token_encoder()
    max_observed = 0
    for chunk in chunks:
        actual_tokens = count_tokens(chunk["text"], encoder)
        assert actual_tokens <= 768, (
            f"Chunk {chunk['chunk_id']} has {actual_tokens} tokens (max=768)"
        )
        max_observed = max(max_observed, actual_tokens)

    print(f"  [PASS] Token size constraint (max observed: {max_observed} tokens)")


def test_token_size_includes_metadata():
    """Verify chunk metadata includes correct token_count field."""
    metadata = extract_metadata(
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        chunk_text=SAMPLE_LONG_TEXT,
    )

    chunks = chunk_document(
        text=SAMPLE_LONG_TEXT,
        tables=[],
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        metadata_base=metadata,
    )

    encoder = _get_token_encoder()
    for chunk in chunks:
        expected = count_tokens(chunk["text"], encoder)
        actual = chunk["token_count"]
        assert actual == expected, (
            f"Chunk token_count mismatch: declared={actual}, actual={expected}"
        )

    print("  [PASS] Token count metadata consistency")


def test_section_boundary_detection():
    """Verify Tier 1 splits correctly on SEC Item boundaries."""
    sections = _split_on_sections(SAMPLE_SECTION_TEXT)
    section_names = [s["section_name"] for s in sections]

    assert any("Risk Factors" in name for name in section_names), (
        f"Expected 'Risk Factors' section. Got: {section_names}"
    )
    assert any("Management" in name for name in section_names), (
        f"Expected 'MD&A' section. Got: {section_names}"
    )
    assert any("Financial Statements" in name for name in section_names), (
        f"Expected 'Financial Statements' section. Got: {section_names}"
    )

    print(f"  [PASS] Section boundary detection ({len(sections)} sections found)")


def test_chunk_metadata_populated():
    """Verify all chunks have complete metadata dictionaries."""
    metadata = extract_metadata(
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        chunk_text=SAMPLE_SECTION_TEXT,
    )

    chunks = chunk_document(
        text=SAMPLE_SECTION_TEXT,
        tables=[SAMPLE_TABLE],
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        metadata_base=metadata,
    )

    required_fields = ["chunk_id", "ticker", "fiscal_year", "doc_type", "section", "contains_table"]
    for chunk in chunks:
        meta = chunk["metadata"]
        for field in required_fields:
            assert field in meta, f"Chunk {chunk['chunk_id']} missing metadata field: {field}"

    print("  [PASS] Chunk metadata completeness")


def test_external_tables_parameter():
    """Verify parser-extracted tables are included as atomic chunks."""
    metadata = extract_metadata(
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        chunk_text="Simple text.",
    )

    chunks = chunk_document(
        text="Simple text.",
        tables=[SAMPLE_TABLE],
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        metadata_base=metadata,
    )

    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) >= 1, "Parser-provided table not created as chunk"
    assert "Net Sales" in table_chunks[0]["text"], "Table content missing from chunk"

    print("  [PASS] External tables parameter handling")


# ============================================================================
# TEST FUNCTIONS — EMBEDDING ENGINE
# ============================================================================


def test_embedding_dimension():
    """Verify embedding engine produces 768-dim vectors."""
    engine = EmbeddingEngine()
    embedding = engine.embed_single("Test financial text about revenue.")

    assert len(embedding) == 768, f"Expected 768 dims, got {len(embedding)}"
    # Verify it's a valid vector (not all zeros)
    assert any(v != 0.0 for v in embedding), "Embedding is all zeros"
    print("  [PASS] Embedding dimension (768)")


def test_embedding_batch():
    """Verify batch embedding produces correct count and dimensions."""
    engine = EmbeddingEngine()
    texts = ["Revenue increased.", "Net income fell.", "Cash flow improved."]
    embeddings = engine.embed(texts)

    assert len(embeddings) == 3, f"Expected 3 embeddings, got {len(embeddings)}"
    for i, emb in enumerate(embeddings):
        assert len(emb) == 768, f"Embedding {i} has {len(emb)} dims, expected 768"

    print("  [PASS] Batch embedding correctness")


# ============================================================================
# TEST FUNCTIONS — DUAL STORAGE (In-Memory/Mocked)
# ============================================================================


def test_dual_storage_with_mock():
    """Verify dual-storage pipeline with mocked MongoDB and in-memory Qdrant."""
    # Create test chunks
    metadata = extract_metadata(
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        chunk_text="Test chunk for dual storage.",
        chunk_index=0,
    )
    chunks = chunk_document(
        text="Test chunk for dual storage verification. Revenue was $100 billion.",
        tables=[],
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        metadata_base=metadata,
    )

    # Mock MongoDB
    mock_mongo = MagicMock()
    mock_mongo.upsert_chunks.return_value = len(chunks)

    # Use real in-memory Qdrant
    qdrant_client = QdrantClient(":memory:")
    indexer = DualStorageIndexer()
    indexer.mongo_indexer = mock_mongo
    indexer.qdrant_indexer._client = qdrant_client
    indexer.qdrant_indexer.ensure_collection()

    # Use the full pipeline (index_chunks embeds + stores in both)
    result = indexer.index_chunks(chunks)

    # Verify MongoDB was called
    mock_mongo.upsert_chunks.assert_called_once_with(chunks)

    # Verify Qdrant has the points
    assert result["qdrant_count"] == len(chunks), f"Qdrant upsert count mismatch: {result['qdrant_count']}"

    # Verify vector search works
    query_embedding = indexer.embedding_engine.embed_single("Test query")
    results = indexer.qdrant_indexer.search(query_embedding, top_k=3)
    assert len(results) > 0, "Qdrant search returned no results"
    assert results[0]["score"] > 0, f"Search score should be > 0, got {results[0]['score']}"

    # Verify metadata payload in results
    payload = results[0]["payload"]
    assert "ticker" in payload, "Payload missing 'ticker'"
    assert "contains_table" in payload, "Payload missing 'contains_table'"

    print("  [PASS] Dual storage with mocked MongoDB + in-memory Qdrant")


def test_qdrant_metadata_filtering():
    """Verify Qdrant pre-filtering by ticker works correctly."""
    qdrant_client = QdrantClient(":memory:")
    indexer = QdrantIndexer()
    indexer._client = qdrant_client
    indexer.ensure_collection()

    # Create chunks for different tickers
    engine = EmbeddingEngine()
    chunks_aapl = []
    chunks_msft = []
    for i in range(3):
        meta = {"ticker": "AAPL", "fiscal_year": "2025", "section": "Item 7: MD&A",
                "doc_type": "10-K", "contains_table": False, "chunk_id": f"AAPL_{i:04d}"}
        chunks_aapl.append({"chunk_id": f"AAPL_{i:04d}", "text": f"AAPL text {i}",
                           "chunk_type": "text", "metadata": meta})
        meta2 = {"ticker": "MSFT", "fiscal_year": "2025", "section": "Item 1: Business",
                 "doc_type": "10-K", "contains_table": False, "chunk_id": f"MSFT_{i:04d}"}
        chunks_msft.append({"chunk_id": f"MSFT_{i:04d}", "text": f"MSFT text {i}",
                           "chunk_type": "text", "metadata": meta2})

    all_chunks = chunks_aapl + chunks_msft
    embeddings = engine.embed([c["text"] for c in all_chunks])
    indexer.upsert_vectors(all_chunks, embeddings)

    # Search without filter — should return results from both tickers
    query_emb = engine.embed_single("financial text")
    all_results = indexer.search(query_emb, top_k=10)
    tickers_in_results = {r["payload"]["ticker"] for r in all_results}
    assert "AAPL" in tickers_in_results and "MSFT" in tickers_in_results, (
        f"Expected both tickers in unfiltered results: {tickers_in_results}"
    )

    # Search with AAPL filter
    aapl_results = indexer.search(query_emb, top_k=10, ticker="AAPL")
    for r in aapl_results:
        assert r["payload"]["ticker"] == "AAPL", (
            f"Filter failed: got ticker={r['payload']['ticker']}"
        )

    print("  [PASS] Qdrant metadata pre-filtering")


def test_tokenizer_fallback():
    """Verify tokenizer fallback mechanism works."""
    encoder = _get_token_encoder()
    text = "Financial analysis of quarterly earnings report."
    tokens = encoder.encode(text)
    decoded = encoder.decode(tokens)
    # Round-trip should be lossless for simple text
    assert decoded.strip() == text.strip(), (
        f"Tokenizer round-trip failed: '{decoded}' != '{text}'"
    )
    print("  [PASS] Tokenizer fallback (tiktoken cl100k_base)")


# ============================================================================
# RUN ALL TESTS
# ============================================================================


def main():
    """Run all Stage 2 ingestion tests."""
    print("=" * 60)
    print("INGESTION STAGE 2 -- Verification Tests")
    print("=" * 60)

    tests = [
        test_token_counting,
        test_table_atomic_preservation,
        test_token_size_constraint,
        test_token_size_includes_metadata,
        test_section_boundary_detection,
        test_chunk_metadata_populated,
        test_external_tables_parameter,
        test_embedding_dimension,
        test_embedding_batch,
        test_dual_storage_with_mock,
        test_qdrant_metadata_filtering,
        test_tokenizer_fallback,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            print(f"\nRunning: {test_fn.__name__}")
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
