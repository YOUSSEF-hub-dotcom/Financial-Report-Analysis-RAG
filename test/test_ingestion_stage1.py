"""
Verification tests for Ingestion Stage 1: Parsing, Cleaning, and Metadata Extraction.

Tests cover:
  - Negative sign parentheses preservation (150)
  - Currency/percentage/scale notation preservation
  - HTML table → Markdown table conversion
  - Metadata dictionary generation (ticker, year, section, etc.)
  - SEC SGML wrapper extraction
"""

import sys
from pathlib import Path

# Add project root and ingestion module path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
# Add the numbered directory directly (not a valid Python package name)
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_ingestion"))

from html_table_parser import parse_sec_filing, _extract_main_document
from cleaning import clean_financial_text
from metadata_extractor import (
    extract_ticker,
    extract_fiscal_year,
    extract_doc_type,
    identify_section,
    contains_table,
    extract_metadata,
)


# ============================================================================
# TEST DATA — Simulated SEC filing content
# ============================================================================

SAMPLE_HTML_WITH_TABLE = """<html><body>
<h1>CONSOLIDATED STATEMENTS OF OPERATIONS</h1>
<p>Apple Inc. reported the following financial results for the fiscal year ended September 27, 2025:</p>
<table>
<tr><th>Line Item</th><th>2025</th><th>2024</th><th>2023</th></tr>
<tr><td>Net Sales</td><td>$395,760</td><td>$383,285</td><td>$383,285</td></tr>
<tr><td>Cost of Sales</td><td>(211,848)</td><td>(214,137)</td><td>(214,137)</td></tr>
<tr><td>Gross Margin</td><td>$183,912</td><td>$169,148</td><td>$169,148</td></tr>
<tr><td>Operating Expenses</td><td>(54,847)</td><td>(51,345)</td><td>(51,345)</td></tr>
<tr><td>Operating Income</td><td>$129,065</td><td>$117,803</td><td>$117,803</td></tr>
</table>
<p>The Company's effective tax rate was 15.2% for 2025, down from 16.1% in 2024.</p>
</body></html>"""

SAMPLE_SEC_SGML = """<SEC-DOCUMENT>0000320193-25-000079.txt : 20251031
<SEC-HEADER>
ACCESSION NUMBER: 0000320193-25-000079
CONFORMED SUBMISSION TYPE: 10-K
COMPANY CONFORMED NAME: Apple Inc.
</SEC-HEADER>
<DOCUMENT>
<TYPE>10-K
<SEQUENCE>1
<TEXT>
<html><body>
<h1>Item 1A. Risk Factors</h1>
<p>The Company faces risks related to global economic conditions.</p>
<h1>Item 7. Management's Discussion and Analysis</h1>
<p>Revenue increased by 3.3% year over year to $395.8 billion.</p>
<table>
<tr><th>Segment</th><th>Revenue</th><th>Operating Income</th></tr>
<tr><td>Products</td><td>$290,159</td><td>$108,949</td></tr>
<tr><td>Services</td><td>$105,601</td><td>$45,141</td></tr>
</table>
<h1>Item 8. Financial Statements</h1>
<p>Please refer to the consolidated financial statements.</p>
</body></html>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>"""

SAMPLE_RAW_FINANCIAL_TEXT = """
The Company reported net income of $96,995 million for fiscal year 2025,
compared to $93,736 million in fiscal year 2024.  Total assets amounted
to $364,980 million while total liabilities were $311,609 million.
The net loss from discontinued operations was $(1,234) million.
Operating cash flow was $118,254 million (2024: $110,543 million).
Return on equity was 157.8% compared to 147.2% in the prior year.
Revenue from Greater China was $78.1B, representing 19.7% of total revenue.
The company repurchased $94,949M of its common stock during the period.
"""


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


def test_negative_parentheses_preservation():
    """Verify (150) and similar parenthesized negatives survive cleaning."""
    input_text = "Net loss was $(1,234) million and operating loss was (500.00)."
    cleaned = clean_financial_text(input_text)

    assert "$(1,234)" in cleaned, f"FAILED: $(1,234) not found in cleaned text. Got: {cleaned}"
    assert "(500.00)" in cleaned, f"FAILED: (500.00) not found in cleaned text. Got: {cleaned}"
    print("  [PASS] Negative sign parentheses preservation")


def test_currency_and_percentage_preservation():
    """Verify currency symbols, percentages, and scale notations are preserved."""
    input_text = "Revenue was $395,760M (15.2% growth). Loss was (€1,200B). Cost: £500K."
    cleaned = clean_financial_text(input_text)

    assert "$395,760M" in cleaned, f"FAILED: $395,760M not preserved. Got: {cleaned}"
    assert "15.2%" in cleaned, f"FAILED: 15.2% not preserved. Got: {cleaned}"
    assert "(€1,200B)" in cleaned, f"FAILED: (€1,200B) not preserved. Got: {cleaned}"
    assert "£500K" in cleaned, f"FAILED: £500K not preserved. Got: {cleaned}"
    print("  [PASS] Currency, percentage, and scale notation preservation")


def test_table_conversion_to_markdown():
    """Verify HTML table converts to valid Markdown table format."""
    from io import StringIO
    from bs4 import BeautifulSoup
    import pandas as pd

    html = """<table>
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Revenue</td><td>$395,760</td></tr>
    <tr><td>Cost</td><td>$(211,848)</td></tr>
    <tr><td>Margin</td><td>$183,912</td></tr>
    </table>"""

    soup = BeautifulSoup(html, "lxml")
    table_tag = soup.find("table")
    df = pd.read_html(StringIO(str(table_tag)), header=0)[0]
    md = df.to_markdown(index=False, tablefmt="pipe")

    # Verify Markdown table structure
    lines = md.strip().split("\n")
    assert len(lines) >= 3, f"FAILED: Markdown table has < 3 lines. Got:\n{md}"
    assert "|" in lines[0], f"FAILED: No pipe characters in header. Got: {lines[0]}"
    assert "---" in lines[1], f"FAILED: No separator line. Got: {lines[1]}"
    assert "$(211,848)" in md, f"FAILED: Parenthesized negative lost in table. Got:\n{md}"
    print("  [PASS] HTML table -> Markdown table conversion")


def test_full_parser_table_extraction():
    """Verify the full parser extracts tables as atomic Markdown blocks."""
    # Create a temporary file for parsing
    import tempfile
    import os

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tmp.write(SAMPLE_SEC_SGML)
    tmp.close()

    try:
        result = parse_sec_filing(tmp.name)
        assert result["table_count"] >= 1, f"FAILED: Expected >= 1 table, got {result['table_count']}"
        assert len(result["tables"]) >= 1, "FAILED: tables list is empty"

        # Check Markdown table structure in extracted tables
        first_table = result["tables"][0]
        assert "|" in first_table, f"FAILED: No pipe in extracted table. Got:\n{first_table}"
        assert "Products" in first_table or "Segment" in first_table, (
            f"FAILED: Expected table content missing. Got:\n{first_table}"
        )
        print("  [PASS] Full parser table extraction")
    finally:
        os.unlink(tmp.name)


def test_metadata_ticker_extraction():
    """Verify ticker extraction from file path and content."""
    # From path
    assert extract_ticker("data/AAPL/10-K/filing.txt") == "AAPL"
    assert extract_ticker("data/NVDA/10-K/filing.txt") == "NVDA"
    assert extract_ticker("data/MSFT/10-K/filing.txt") == "MSFT"

    # From content
    content = "APPLE INC. reported strong quarterly results."
    assert extract_ticker("unknown.txt", content) == "AAPL"

    print("  [PASS] Ticker extraction")


def test_metadata_fiscal_year_extraction():
    """Verify fiscal year extraction from paths and content."""
    # From path
    year = extract_fiscal_year("data/AAPL/10-K/0000320193-25-000079.txt")
    assert year == "2025", f"FAILED: Expected 2025, got {year}"

    # From content
    content = "For the fiscal year ended September 27, 2025, the Company reported..."
    year = extract_fiscal_year("unknown.txt", content)
    assert year == "2025", f"FAILED: Expected 2025 from content, got {year}"

    print("  [PASS] Fiscal year extraction")


def test_metadata_doc_type_extraction():
    """Verify document type extraction."""
    assert extract_doc_type("data/AAPL/10-K/filing.txt") == "10-K"
    assert extract_doc_type("data/MSFT/10-Q/filing.txt") == "10-Q"

    # From SGML content
    sgml = "<TYPE>10-K\n<SEQUENCE>1"
    assert extract_doc_type("unknown.txt", sgml) == "10-K"

    print("  [PASS] Document type extraction")


def test_section_identification():
    """Verify SEC section identification from text."""
    text_item1a = "Item 1A. Risk Factors The Company faces various risks..."
    assert identify_section(text_item1a) == "Item 1A: Risk Factors"

    text_item7 = "Item 7. Management's Discussion and Analysis of Financial Condition"
    assert identify_section(text_item7) == "Item 7: MD&A"

    text_item8 = "Item 8. Financial Statements and Supplementary Data"
    assert identify_section(text_item8) == "Item 8: Financial Statements"

    # Default case
    assert identify_section("Some random text without section headers") == "General"

    print("  [PASS] Section identification")


def test_contains_table_detection():
    """Verify Markdown table presence detection."""
    with_table = "Some text.\n| Col1 | Col2 |\n| --- | --- |\n| A | B |\nMore text."
    without_table = "Just plain text with no tables at all."

    assert contains_table(with_table) is True
    assert contains_table(without_table) is False

    print("  [PASS] Table presence detection")


def test_full_metadata_extraction():
    """Verify complete metadata dictionary generation."""
    meta = extract_metadata(
        file_path="data/AAPL/10-K/0000320193-25-000079/full-submission.txt",
        content=SAMPLE_SEC_SGML,
        chunk_text="Item 7. Management's Discussion and Analysis. Revenue increased.\n| A | B |\n| 1 | 2 |",
        chunk_index=0,
    )

    assert meta["ticker"] == "AAPL", f"FAILED: ticker={meta['ticker']}"
    assert meta["fiscal_year"] == "2025", f"FAILED: year={meta['fiscal_year']}"
    assert meta["doc_type"] == "10-K", f"FAILED: doc_type={meta['doc_type']}"
    assert meta["section"] == "Item 7: MD&A", f"FAILED: section={meta['section']}"
    assert meta["contains_table"] is True, "FAILED: contains_table should be True"
    assert meta["source_file"] == "full-submission.txt"
    assert meta["chunk_id"].startswith("AAPL_")
    assert meta["page_number"] is None

    print("  [PASS] Full metadata extraction")


def test_empty_input_handling():
    """Verify graceful handling of empty inputs."""
    assert clean_financial_text("") == ""
    assert clean_financial_text("   ") == ""
    assert identify_section("") == "General"
    assert contains_table("") is False
    print("  [PASS] Empty input handling")


def test_html_tag_stripping_in_cleaning():
    """Verify HTML tags are stripped while content is preserved."""
    input_text = '<p style="font-size:12px">Revenue was <b>$395,760</b></p>'
    cleaned = clean_financial_text(input_text)

    assert "<p" not in cleaned, f"FAILED: HTML tag not stripped. Got: {cleaned}"
    assert "<b>" not in cleaned, f"FAILED: HTML tag not stripped. Got: {cleaned}"
    assert "$395,760" in cleaned, f"FAILED: Content lost. Got: {cleaned}"
    assert "Revenue was" in cleaned, f"FAILED: Content lost. Got: {cleaned}"
    print("  [PASS] HTML tag stripping in cleaning")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def main():
    """Run all Stage 1 ingestion tests."""
    print("=" * 60)
    print("INGESTION STAGE 1 — Verification Tests")
    print("=" * 60)

    tests = [
        test_negative_parentheses_preservation,
        test_currency_and_percentage_preservation,
        test_table_conversion_to_markdown,
        test_full_parser_table_extraction,
        test_metadata_ticker_extraction,
        test_metadata_fiscal_year_extraction,
        test_metadata_doc_type_extraction,
        test_section_identification,
        test_contains_table_detection,
        test_full_metadata_extraction,
        test_empty_input_handling,
        test_html_tag_stripping_in_cleaning,
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
