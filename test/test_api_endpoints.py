"""
Comprehensive test suite for FastAPI endpoints.

Uses TestClient with mocked pipeline to avoid external dependencies.
Tests cover:
  - Health check (with mocked service checks)
  - Chat endpoint (valid query, empty body, empty query)
  - Streaming endpoint (SSE format)
  - Document upload (valid file, invalid file type)
  - Background worker guardrail
  - Error handling (pipeline unavailable, unhandled exceptions)
"""

import io
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Import these at test level so patching works when main.py imports them at runtime
import pymongo
import qdrant_client
import redis

# Add paths so we can import app modules under test
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_src_root = _PROJECT_ROOT / "src"
for _p in (
    str(_PROJECT_ROOT),
    str(_src_root),
    str(_src_root / "1_ingestion"),
    str(_src_root / "2_generation"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.api.main import app
from app.api.schemas import ChatQueryResponse
from app.api.worker import run_guardrail_background, log_metrics_background


# ============================================================================
# Fixtures — patch the pipeline to return canned responses
# ============================================================================

@pytest.fixture(autouse=True)
def patch_pipeline():
    """Mock the pipeline and health-check dependencies for every test."""
    mock_pipe = MagicMock()
    mock_pipe.query.return_value = {
        "raw_output": json.dumps({
            "internal_thought": "Test reasoning $100B revenue.",
            "extracted_raw_data": "Revenue $100B.",
            "answer": "Apple reported $100B in revenue.",
            "sources": ["AAPL - 2025 - Business Overview - 10"],
        }),
        "parsed": MagicMock(
            answer="Apple reported $100B in revenue.",
            extracted_raw_data="Revenue $100B.",
            sources=["AAPL - 2025 - Business Overview - 10"],
            model_dump_json=lambda: '{"answer":"Apple reported $100B in revenue."}',
        ),
        "model_used": "llama-3.3-70b-versatile",
        "fallback_triggered": False,
        "ttft_ms": 234.5,
        "cache_hit": False,
        "pipeline_run_id": "test1234",
    }

    async def mock_stream(**kwargs):
        yield "Apple"
        yield " reported"
        yield " $100B"
        yield " in revenue."

    mock_pipe.query_stream = mock_stream
    mock_pipe.close = MagicMock()

    # Patch the pipeline construction in the lifespan
    with (
        patch("app.api.main.FinancialRAGPipeline", return_value=mock_pipe) as mock_cls,
        patch.object(pymongo, "MongoClient") as mock_mongo,
        patch.object(qdrant_client, "QdrantClient") as mock_qdrant,
        patch.object(redis, "from_url") as mock_redis,
    ):
        mock_mongo.return_value.admin.command.return_value = {"ok": 1}
        mock_qdrant.return_value.get_collections.return_value = MagicMock()
        mock_redis.return_value.ping.return_value = True

        yield {
            "pipeline": mock_pipe,
            "pipeline_cls": mock_cls,
            "pymongo": mock_mongo,
            "qdrant": mock_qdrant,
            "redis": mock_redis,
        }


@pytest.fixture
def client(patch_pipeline):
    """FastAPI TestClient with mocked pipeline."""
    with TestClient(app) as c:
        yield c


# ============================================================================
# Health Check
# ============================================================================

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] in ("ok", "degraded")

    def test_health_contains_services(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert "services" in body
        assert "mongodb" in body["services"]
        assert "qdrant" in body["services"]
        assert "redis" in body["services"]

    def test_health_mongodb_reports_ok(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert body["services"]["mongodb"] == "ok"


# ============================================================================
# Chat Endpoint
# ============================================================================

class TestChatEndpoint:

    def test_valid_query_returns_200(self, client):
        resp = client.post("/api/v1/chat", json={
            "user_query": "What is Apple's revenue?",
        })
        assert resp.status_code == 200

    def test_valid_query_returns_chat_response_schema(self, client):
        resp = client.post("/api/v1/chat", json={
            "user_query": "What is Apple's revenue?",
        })
        body = resp.json()
        assert "answer" in body
        assert "execution_time_ms" in body
        assert "model_used" in body
        assert "cache_hit" in body

    def test_query_with_ticker_filter(self, client):
        resp = client.post("/api/v1/chat", json={
            "user_query": "What is Apple's revenue?",
            "ticker": "AAPL",
            "fiscal_year": "2025",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Apple reported $100B in revenue."

    def test_query_with_session_id(self, client):
        resp = client.post("/api/v1/chat", json={
            "user_query": "What is Apple's revenue?",
            "session_id": "sess_001",
        })
        assert resp.status_code == 200

    def test_empty_body_returns_422(self, client):
        resp = client.post("/api/v1/chat", json={})
        assert resp.status_code == 422

    def test_short_query_returns_422(self, client):
        resp = client.post("/api/v1/chat", json={
            "user_query": "ab",
        })
        assert resp.status_code == 422

    def test_missing_user_query_returns_422(self, client):
        resp = client.post("/api/v1/chat", json={
            "ticker": "AAPL",
        })
        assert resp.status_code == 422

    def test_invalid_json_body_returns_422(self, client):
        resp = client.post("/api/v1/chat", data="not-json")
        assert resp.status_code == 422

    def test_pipeline_called_with_correct_args(self, client, patch_pipeline):
        client.post("/api/v1/chat", json={
            "user_query": "Revenue?",
            "ticker": "AAPL",
            "fiscal_year": "2025",
            "session_id": "sess_001",
        })
        pipe = patch_pipeline["pipeline"]
        pipe.query.assert_called_once_with(
            user_query="Revenue?",
            ticker="AAPL",
            fiscal_year="2025",
            session_id="sess_001",
        )


# ============================================================================
# Streaming Endpoint
# ============================================================================

class TestStreamingEndpoint:

    def test_stream_returns_200(self, client):
        resp = client.post("/api/v1/chat/stream", json={
            "user_query": "What is Apple's revenue?",
        })
        assert resp.status_code == 200

    def test_stream_returns_sse_content_type(self, client):
        resp = client.post("/api/v1/chat/stream", json={
            "user_query": "What is Apple's revenue?",
        })
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_yields_tokens(self, client):
        resp = client.post("/api/v1/chat/stream", json={
            "user_query": "What is Apple's revenue?",
        })
        assert resp.status_code == 200
        content = resp.text
        assert "data: Apple" in content or "data:Apple" in content

    def test_stream_ends_with_done_event(self, client):
        resp = client.post("/api/v1/chat/stream", json={
            "user_query": "What is Apple's revenue?",
        })
        assert resp.status_code == 200
        assert 'event: done' in resp.text

    def test_stream_with_ticker_filter(self, client):
        resp = client.post("/api/v1/chat/stream", json={
            "user_query": "Revenue?",
            "ticker": "MSFT",
        })
        assert resp.status_code == 200

    def test_stream_empty_query_returns_422(self, client):
        resp = client.post("/api/v1/chat/stream", json={
            "user_query": "ab",
        })
        assert resp.status_code == 422


# ============================================================================
# Document Upload
# ============================================================================

class TestDocumentUpload:

    def test_upload_valid_txt_file_returns_200(self, client):
        content = b"<DOCUMENT><TYPE>10-K\nTest SEC filing content."
        resp = client.post(
            "/api/v1/documents/upload?ticker=AAPL&fiscal_year=2025",
            files={"file": ("filing.txt", content, "text/plain")},
        )
        assert resp.status_code == 200

    def test_upload_returns_document_upload_schema(self, client):
        content = b"SEC test data."
        resp = client.post(
            "/api/v1/documents/upload?ticker=AAPL&fiscal_year=2025",
            files={"file": ("test.txt", content, "text/plain")},
        )
        body = resp.json()
        assert body["filename"] == "test.txt"
        assert body["ticker"] == "AAPL"
        assert body["fiscal_year"] == "2025"
        assert "task_id" in body

    def test_upload_invalid_file_extension_returns_400(self, client):
        content = b"some data"
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("filing.exe", content, "application/octet-stream")},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "Unsupported file type" in body["detail"]

    def test_upload_html_file_returns_200(self, client):
        content = b"<html><body>test</body></html>"
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("filing.html", content, "text/html")},
        )
        assert resp.status_code == 200

    def test_upload_saves_file_to_data_dir(self, client):
        content = b"SEC test data for saving."
        resp = client.post(
            "/api/v1/documents/upload?ticker=TEST&fiscal_year=2024",
            files={"file": ("save_test.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        from config.settings import DATA_DIR
        saved = DATA_DIR / "TEST" / "10-K" / "2024" / "save_test.txt"
        assert saved.exists()
        saved.unlink()  # cleanup
        # Clean up empty dirs
        saved.parent.rmdir()
        saved.parent.parent.rmdir()
        saved.parent.parent.parent.rmdir()


# ============================================================================
# Error Handling
# ============================================================================

class TestErrorHandling:

    def test_pipeline_unavailable_returns_503(self):
        """When pipeline is None in app state, /chat returns 503."""
        with TestClient(app) as c:
            # Clear app state
            app.state.pipeline = None
            resp = c.post("/api/v1/chat", json={"user_query": "Revenue?"})
            assert resp.status_code == 503
            assert "Pipeline not initialized" in resp.json()["detail"]

    def test_global_exception_handler_catches_errors(self):
        """An unhandled exception in an endpoint returns 500."""
        with TestClient(app) as c:
            app.state.pipeline = None
            resp = c.post("/api/v1/chat/stream", json={"user_query": "Revenue?"})
            assert resp.status_code in (500, 503)

    def test_invalid_http_method_returns_405(self, client):
        resp = client.get("/api/v1/chat")
        assert resp.status_code == 405

    def test_unknown_route_returns_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404


# ============================================================================
# Background Worker
# ============================================================================

class TestBackgroundWorker:

    @pytest.mark.asyncio
    async def test_guardrail_worker_returns_dict(self):
        result = await run_guardrail_background(
            query="Revenue?",
            answer="Test answer with $100B.",
            extracted_raw_data="$100B revenue.",
        )
        assert isinstance(result, dict)
        assert "passed" in result
        assert "verified_claims" in result
        assert "failed_claims" in result

    @pytest.mark.asyncio
    async def test_guardrail_worker_handles_empty_data(self):
        result = await run_guardrail_background(
            query="Revenue?",
            answer="",
            extracted_raw_data="",
        )
        assert result["passed"] is True

    def test_metrics_logging_does_not_crash(self):
        log_metrics_background({"test_metric": 42})
        log_metrics_background({"test_param": "hello"})
        log_metrics_background({})
