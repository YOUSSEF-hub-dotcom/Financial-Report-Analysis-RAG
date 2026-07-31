"""
Tests for multi-format document parsing and upload integration.

Covers:
  - APIFileParser with HTML/TXT (real content)
  - APIFileParser with PDF/DOCX (LlamaParse mock + fallback paths)
  - Upload endpoint integration for .pdf and .docx files
"""

import io
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Stub llama_parse so mock.patch("llama_parse.LlamaParse") resolves even
# when the real package is not installed.
if "llama_parse" not in sys.modules:
    import types
    fake = types.ModuleType("llama_parse")
    fake.LlamaParse = MagicMock()
    sys.modules["llama_parse"] = fake

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

from app.api.parsers import APIFileParser


# ===========================================================================
# Helpers — generate minimal PDF and DOCX content for testing
# ===========================================================================

def _make_test_pdf(text: str = "Test financial data Revenue $100M") -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(w=0, text=text)
    from io import BytesIO
    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def _make_test_docx(text: str = "Test financial data Revenue $100M") -> bytes:
    import docx
    from io import BytesIO
    doc = docx.Document()
    doc.add_paragraph(text)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_test_html(table: bool = True) -> str:
    if table:
        return """<html><body><table>
<tr><th>Item</th><th>Amount</th></tr>
<tr><td>Revenue</td><td>$100M</td></tr>
</table></body></html>"""
    return "<html><body><p>Simple text content.</p></body></html>"


# ===========================================================================
# APIFileParser Unit Tests
# ===========================================================================

class TestHTMLParsing:
    """HTML files route through the existing SEC parser."""

    def test_html_with_table_returns_text_and_tables(self):
        parser = APIFileParser()
        html = _make_test_html(table=True).encode("utf-8")
        result = parser.parse_file(html, "filing.html")
        assert "text_content" in result
        assert "tables" in result
        assert len(result["tables"]) > 0
        assert "Revenue" in " ".join(result["tables"])

    def test_html_without_table_returns_text_only(self):
        parser = APIFileParser()
        html = _make_test_html(table=False).encode("utf-8")
        result = parser.parse_file(html, "filing.html")
        assert "text_content" in result
        assert isinstance(result["tables"], list)
        assert "Simple text content" in result["text_content"]

    def test_htm_extension_accepted(self):
        parser = APIFileParser()
        result = parser.parse_file(b"<html><p>Test</p></html>", "page.htm")
        assert "Test" in result["text_content"]


class TestTXTParsing:
    """Plain text files pass through cleanly."""

    def test_txt_returns_raw_text(self):
        parser = APIFileParser()
        content = b"This is a plain text financial document.\nRevenue: $100M."
        result = parser.parse_file(content, "report.txt")
        assert "text_content" in result
        assert "Revenue" in result["text_content"]
        assert "$100M" in result["text_content"]
        assert len(result["tables"]) == 0

    def test_sgml_extension_accepted(self):
        parser = APIFileParser()
        content = b"<DOCUMENT><TYPE>10-K\nTest SEC filing."
        result = parser.parse_file(content, "filing.sgml")
        assert "text_content" in result


class TestPDFLlamaParse:
    """PDF files route through LlamaParse when key is available."""

    def test_llamaparse_called_on_pdf(self):
        pdf_bytes = _make_test_pdf()
        parser = APIFileParser()
        parser._llama_key = "test-key-123"

        mock_parser = MagicMock()
        mock_doc = MagicMock()
        mock_doc.text = "Parsed markdown content.\n\n| Rev | $100M |\n| --- | --- |\n| Q1 | 50M |"
        mock_parser.load_data.return_value = [mock_doc]

        with patch("llama_parse.LlamaParse", return_value=mock_parser):
            result = parser.parse_file(pdf_bytes, "report.pdf")

        assert "text_content" in result
        assert "Parsed markdown content" in result["text_content"]
        assert len(result["tables"]) > 0

    def test_llamaparse_fallback_on_error(self):
        pdf_bytes = _make_test_pdf("Fallback content $50M.")
        parser = APIFileParser()
        parser._llama_key = "test-key-123"

        mock_parser = MagicMock()
        mock_parser.load_data.side_effect = RuntimeError("API error")

        with patch("llama_parse.LlamaParse", return_value=mock_parser):
            result = parser.parse_file(pdf_bytes, "report.pdf")

        assert "text_content" in result
        assert "Fallback content" in result["text_content"]
        assert "$50M" in result["text_content"]


class TestPDFFallback:
    """PDF files fall back to pypdf when no LlamaParse key."""

    def test_pypdf_extracts_text_when_no_key(self):
        parser = APIFileParser()
        parser._llama_key = ""
        pdf_bytes = _make_test_pdf("PyPDF extracted Revenue $200M.")
        result = parser.parse_file(pdf_bytes, "report.pdf")
        assert "text_content" in result
        assert "Revenue" in result["text_content"]
        assert "$200M" in result["text_content"]
        assert len(result["tables"]) == 0

    def test_pypdf_empty_pdf_returns_empty_text(self):
        parser = APIFileParser()
        parser._llama_key = ""
        pdf_bytes = _make_test_pdf("")
        result = parser.parse_file(pdf_bytes, "empty.pdf")
        assert "text_content" in result


class TestDOCXLlamaParse:
    """DOCX files route through LlamaParse when key is available."""

    def test_llamaparse_called_on_docx(self):
        docx_bytes = _make_test_docx()
        parser = APIFileParser()
        parser._llama_key = "test-key-456"

        mock_parser = MagicMock()
        mock_doc = MagicMock()
        mock_doc.text = "DOCX parsed content.\n\n| Metric | Value |\n| --- | --- |\n| Rev | $300M |"
        mock_parser.load_data.return_value = [mock_doc]

        with patch("llama_parse.LlamaParse", return_value=mock_parser):
            result = parser.parse_file(docx_bytes, "report.docx")

        assert "text_content" in result
        assert "DOCX parsed content" in result["text_content"]
        assert len(result["tables"]) >= 1
        assert "$300M" in " ".join(result["tables"])


class TestDOCXFallback:
    """DOCX files fall back to python-docx when no LlamaParse key."""

    def test_python_docx_extracts_text_when_no_key(self):
        parser = APIFileParser()
        parser._llama_key = ""
        docx_bytes = _make_test_docx("DOCX fallback Revenue $400M.")
        result = parser.parse_file(docx_bytes, "report.docx")
        assert "text_content" in result
        assert "Revenue" in result["text_content"]
        assert "$400M" in result["text_content"]

    def test_python_docx_handles_empty_document(self):
        parser = APIFileParser()
        parser._llama_key = ""
        empty_docx = _make_test_docx("")
        result = parser.parse_file(empty_docx, "empty.docx")
        assert "text_content" in result


class TestAPIFileParserEdgeCases:

    def test_unsupported_extension_raises_valueerror(self):
        parser = APIFileParser()
        with pytest.raises(ValueError, match="Unsupported file type"):
            parser.parse_file(b"data", "file.xyz")

    def test_supported_extensions_list(self):
        assert ".pdf" in APIFileParser.SUPPORTED_EXTENSIONS
        assert ".docx" in APIFileParser.SUPPORTED_EXTENSIONS
        assert ".html" in APIFileParser.SUPPORTED_EXTENSIONS
        assert ".txt" in APIFileParser.SUPPORTED_EXTENSIONS
        assert ".sgml" in APIFileParser.SUPPORTED_EXTENSIONS

    def test_returns_all_expected_keys(self):
        parser = APIFileParser()
        result = parser.parse_file(b"test content", "file.txt")
        assert "text_content" in result
        assert "tables" in result
        assert "raw_html" in result


# ===========================================================================
# API Upload Integration Tests
# ===========================================================================

@pytest.fixture(autouse=True)
def patch_pipeline_and_deps():
    """Mock the pipeline constructor for API upload tests only.
    
    EmbeddingEngine/MongoDBIndexer/QdrantIndexer are imported inside
    _ingest_document (background task), not at module level, so they
    don't need to be patched here — the background task is not awaited
    in TestClient tests.
    """
    mock_pipe = MagicMock()
    mock_pipe.close = MagicMock()

    with patch("app.api.main.FinancialRAGPipeline", return_value=mock_pipe):
        yield {"pipeline": mock_pipe, "pipeline_cls": mock_pipe}


@pytest.fixture
def client(patch_pipeline_and_deps):
    with TestClient(app) as c:
        yield c


# Need to import app after fixtures are defined
from app.api.main import app


class TestAPIUploadIntegration:

    def test_upload_pdf_returns_200(self, client):
        pdf_bytes = _make_test_pdf()
        resp = client.post(
            "/api/v1/documents/upload?ticker=AAPL&fiscal_year=2025",
            files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "report.pdf"

    def test_upload_docx_returns_200(self, client):
        docx_bytes = _make_test_docx()
        resp = client.post(
            "/api/v1/documents/upload?ticker=MSFT&fiscal_year=2024",
            files={"file": ("report.docx", docx_bytes, "application/docx")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "report.docx"

    def test_upload_html_returns_200(self, client):
        html_bytes = _make_test_html().encode("utf-8")
        resp = client.post(
            "/api/v1/documents/upload?ticker=AAPL&fiscal_year=2025",
            files={"file": ("filing.html", html_bytes, "text/html")},
        )
        assert resp.status_code == 200

    def test_upload_txt_returns_200(self, client):
        resp = client.post(
            "/api/v1/documents/upload?ticker=AAPL&fiscal_year=2025",
            files={"file": ("filing.txt", b"Test content.", "text/plain")},
        )
        assert resp.status_code == 200

    def test_upload_sgml_returns_200(self, client):
        content = b"<DOCUMENT><TYPE>10-K\nTest SEC filing content."
        resp = client.post(
            "/api/v1/documents/upload?ticker=AAPL&fiscal_year=2025",
            files={"file": ("filing.sgml", content, "text/plain")},
        )
        assert resp.status_code == 200

    def test_upload_unsupported_extension_returns_400(self, client):
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("file.exe", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_upload_pdf_saves_to_data_dir(self, client):
        pdf_bytes = _make_test_pdf()
        resp = client.post(
            "/api/v1/documents/upload?ticker=TEST&fiscal_year=2024",
            files={"file": ("save_test.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 200
        from config.settings import DATA_DIR
        saved = DATA_DIR / "TEST" / "10-K" / "2024" / "save_test.pdf"
        assert saved.exists()
        saved.unlink()
        saved.parent.rmdir()
        saved.parent.parent.rmdir()
        saved.parent.parent.parent.rmdir()

    def test_upload_docx_saves_to_data_dir(self, client):
        docx_bytes = _make_test_docx()
        resp = client.post(
            "/api/v1/documents/upload?ticker=TEST&fiscal_year=2024",
            files={"file": ("save_test.docx", docx_bytes, "application/docx")},
        )
        assert resp.status_code == 200
        from config.settings import DATA_DIR
        saved = DATA_DIR / "TEST" / "10-K" / "2024" / "save_test.docx"
        assert saved.exists()
        saved.unlink()
        saved.parent.rmdir()
        saved.parent.parent.rmdir()
        saved.parent.parent.parent.rmdir()
