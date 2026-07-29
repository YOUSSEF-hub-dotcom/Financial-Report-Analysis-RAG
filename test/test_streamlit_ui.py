"""
Tests for Streamlit UI API client wrapper functions.

Mocks httpx at the transport layer to avoid needing a running FastAPI server.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.ui.streamlit_app import (
    DEFAULT_API_URL,
    SUPPORTED_EXTENSIONS,
    check_health,
    send_query,
    stream_query,
    upload_document,
)


# ===========================================================================
# Health Check
# ===========================================================================

class TestCheckHealth:

    def test_returns_status_and_services(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "ok",
            "services": {"mongodb": "ok", "qdrant": "ok", "redis": "ok"},
        }

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            result = check_health()

        assert result["status"] == "ok"
        assert result["services"]["mongodb"] == "ok"
        assert result["services"]["qdrant"] == "ok"
        assert result["services"]["redis"] == "ok"
        mock_get.assert_called_once_with(f"{DEFAULT_API_URL}/health", timeout=5)

    def test_raises_on_http_error(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 503
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service unavailable", request=MagicMock(), response=mock_resp
        )

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                check_health()

    def test_degraded_status_when_redis_down(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "degraded",
            "services": {"mongodb": "ok", "qdrant": "ok", "redis": "unavailable"},
        }

        with patch("httpx.get", return_value=mock_resp):
            result = check_health()

        assert result["status"] == "degraded"


# ===========================================================================
# Sync Query
# ===========================================================================

class TestSendQuery:

    def test_returns_full_response(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "answer": "Apple's revenue was $100B.",
            "sources": [{"ticker": "AAPL", "score": 0.95, "fiscal_year": "2025", "section": "Item 7: MD&A", "text_snippet": "Revenue $100B."}],
            "execution_time_ms": 1234.56,
            "model_used": "llama-3.3-70b-versatile",
            "cache_hit": False,
        }

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = send_query(
                query="What is Apple's revenue?",
                ticker="AAPL",
                fiscal_year="2025",
                session_id="sess_test",
            )

        assert result["answer"] == "Apple's revenue was $100B."
        assert result["model_used"] == "llama-3.3-70b-versatile"
        assert result["cache_hit"] is False
        assert len(result["sources"]) == 1

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"] == {
            "user_query": "What is Apple's revenue?",
            "ticker": "AAPL",
            "fiscal_year": "2025",
            "session_id": "sess_test",
        }

    def test_query_without_filters(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"answer": "Test", "execution_time_ms": 100, "model_used": "test", "cache_hit": False}

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = send_query(query="Revenue?")

        assert result["answer"] == "Test"
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"] == {"user_query": "Revenue?"}

    def test_raises_on_non_200(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 422
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unprocessable", request=MagicMock(), response=mock_resp
        )

        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                send_query(query="ab")


# ===========================================================================
# Streaming Query
# ===========================================================================

class TestStreamQuery:

    def _make_sse_lines(self, tokens: list[str]) -> list[str]:
        lines = []
        for t in tokens:
            lines.append(f"data: {t}")
        lines.append("event: done")
        lines.append("data: ")
        return lines

    def test_yields_tokens(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(self._make_sse_lines(["Apple", " reported", " $100B", " revenue."]))

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.stream.return_value.__enter__.return_value = mock_resp

        with patch("httpx.Client", return_value=mock_client):
            tokens = list(stream_query(query="Revenue?"))

        assert tokens == ["Apple", " reported", " $100B", " revenue."]

    def test_handles_done_event_early(self):
        """Stream stops when 'event: done' appears, even if more lines follow."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(self._make_sse_lines(["First"]))

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.stream.return_value.__enter__.return_value = mock_resp

        with patch("httpx.Client", return_value=mock_client):
            tokens = list(stream_query(query="Test?"))

        assert tokens == ["First"]

    def test_skips_empty_lines(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter([
            "data: Token1",
            "",
            "data: Token2",
            "event: done",
            "data: ",
        ])

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.stream.return_value.__enter__.return_value = mock_resp

        with patch("httpx.Client", return_value=mock_client):
            tokens = list(stream_query(query="Test?"))

        assert tokens == ["Token1", "Token2"]

    def test_raises_on_http_error(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad request", request=MagicMock(), response=mock_resp
        )

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.stream.return_value.__enter__.return_value = mock_resp

        with patch("httpx.Client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                list(stream_query(query="Bad?"))

    def test_passes_filters(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(self._make_sse_lines(["OK"]))

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__.return_value = mock_client
        mock_client.stream.return_value.__enter__.return_value = mock_resp

        with patch("httpx.Client", return_value=mock_client) as mock_cls:
            list(stream_query(query="Q", ticker="AAPL", fiscal_year="2025", session_id="s1"))

        call_json = mock_cls.return_value.__enter__.return_value.stream.call_args.kwargs["json"]
        assert call_json == {"user_query": "Q", "ticker": "AAPL", "fiscal_year": "2025", "session_id": "s1"}


# ===========================================================================
# Document Upload
# ===========================================================================

class TestUploadDocument:

    def test_upload_txt_returns_result(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "filename": "report.txt",
            "ticker": "AAPL",
            "fiscal_year": "2025",
            "chunks_created": 42,
            "mongo_count": 42,
            "qdrant_count": 42,
            "elapsed_seconds": 12.34,
            "task_id": "abc123",
        }

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = upload_document(
                file_bytes=b"test content",
                filename="report.txt",
                ticker="AAPL",
                fiscal_year="2025",
            )

        assert result["filename"] == "report.txt"
        assert result["chunks_created"] == 42
        assert result["task_id"] == "abc123"

        # Check correct params passed
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["params"] == {"ticker": "AAPL", "fiscal_year": "2025"}
        assert "file" in call_kwargs["files"]

    def test_upload_pdf(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"filename": "r.pdf", "chunks_created": 10}

        with patch("httpx.post", return_value=mock_resp):
            result = upload_document(b"%PDF-data", "r.pdf")
        assert result["chunks_created"] == 10

    def test_upload_docx(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"filename": "r.docx", "chunks_created": 15}

        with patch("httpx.post", return_value=mock_resp):
            result = upload_document(b"DOCX-data", "r.docx")
        assert result["chunks_created"] == 15

    def test_unsupported_extension_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            upload_document(b"data", "file.exe")

    def test_raises_on_http_error(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad request", request=MagicMock(), response=mock_resp
        )

        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                upload_document(b"data", "test.txt")


# ===========================================================================
# Constants
# ===========================================================================

class TestConstants:

    def test_supported_extensions_include_required_types(self):
        for ext in (".pdf", ".docx", ".html", ".htm", ".txt", ".sgml"):
            assert ext in SUPPORTED_EXTENSIONS
