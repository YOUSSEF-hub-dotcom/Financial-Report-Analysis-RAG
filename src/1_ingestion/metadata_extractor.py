"""
Metadata Extractor — extracts structured financial metadata from SEC filing content and paths.

Extracted fields:
  - ticker, fiscal_year, doc_type, source_file
  - section (SEC 10-K section identification)
  - contains_table (boolean flag for table presence)
  - chunk_id (unique identifier for downstream indexing)
"""

import hashlib
import re
import uuid
from pathlib import Path

from config.logging_config import get_logger

logger = get_logger("ingestion.metadata")

# --- SEC 10-K Section Patterns ---
# Maps regex patterns to canonical section names
_SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Item\s*1A[\.:]\s*Risk\s*Factors", re.IGNORECASE), "Item 1A: Risk Factors"),
    (re.compile(r"Item\s*1B[\.:]?\s*Risk\s*Factors", re.IGNORECASE), "Item 1B: Risk Factors"),
    (re.compile(r"Item\s*1[\.:]\s*(?:Business|Description\s*of\s*Business)", re.IGNORECASE), "Item 1: Business"),
    (re.compile(r"Item\s*2[\.:]\s*Properties", re.IGNORECASE), "Item 2: Properties"),
    (re.compile(r"Item\s*3[\.:]\s*Legal\s*Proceedings", re.IGNORECASE), "Item 3: Legal Proceedings"),
    (re.compile(r"Item\s*4[\.:]\s*Mine\s*Safety", re.IGNORECASE), "Item 4: Mine Safety Disclosures"),
    (re.compile(r"Item\s*5[\.:].*?(?:Market|Equity|Stock\s*Performance)", re.IGNORECASE), "Item 5: Market for Registrant's Common Equity"),
    (re.compile(r"Item\s*7[\.:]\s*(?:Management'?s?\s*Discussion|MD&A)", re.IGNORECASE), "Item 7: MD&A"),
    (re.compile(r"Item\s*7A[\.:].*?Quantitative", re.IGNORECASE), "Item 7A: Quantitative and Qualitative Disclosures"),
    (re.compile(r"Item\s*8[\.:]\s*(?:Financial\s*Statements|Financial\s*Data)", re.IGNORECASE), "Item 8: Financial Statements"),
    (re.compile(r"Item\s*9[\.:]\s*(?:Changes|Disagreements)", re.IGNORECASE), "Item 9: Changes in Accountants"),
    (re.compile(r"Item\s*9A[\.:]\s*Controls", re.IGNORECASE), "Item 9A: Controls and Procedures"),
    (re.compile(r"Item\s*9B[\.:]\s*Other", re.IGNORECASE), "Item 9B: Other Information"),
    (re.compile(r"Item\s*10[\.:]\s*Directors", re.IGNORECASE), "Item 10: Directors and Corporate Governance"),
    (re.compile(r"Item\s*11[\.:]\s*Executive\s*Compensation", re.IGNORECASE), "Item 11: Executive Compensation"),
    (re.compile(r"Item\s*12[\.:]\s*Security\s*Ownership", re.IGNORECASE), "Item 12: Security Ownership"),
    (re.compile(r"Item\s*13[\.:]\s*Certain\s*Relationships", re.IGNORECASE), "Item 13: Related Party Transactions"),
    (re.compile(r"Item\s*14[\.:]\s*Principal\s*Accounting", re.IGNORECASE), "Item 14: Accounting Fees and Services"),
    (re.compile(r"Part\s*IV", re.IGNORECASE), "Part IV"),
    (re.compile(r"Signatures", re.IGNORECASE), "Signatures"),
]

# Ticker extraction patterns from file paths
_TICKER_FROM_PATH = re.compile(r"(?:^|[/\\])(AAPL|MSFT|NVDA|GOOGL|AMZN|META|TSLA)(?:[/\\]|$)", re.IGNORECASE)

# Fiscal year patterns
# Match 10-K/10-Q directory then look for a plausible year (20xx or just the accession YY)
_FISCAL_YEAR_FROM_PATH = re.compile(r"(?:10-K|10-Q)[/\\](?:.*?[/\\])?(\d{4})")
_FISCAL_YEAR_FROM_ACCESSION = re.compile(r"-(\d{2})-\d{6}")
_FISCAL_YEAR_FROM_CONTENT = re.compile(
    r"(?:fiscal\s*year\s*(?:ended|ending)\s*\w+\s+\d{1,2},?\s*(\d{4})|"
    r"for\s*the\s*(?:fiscal\s*)?year\s*(?:ended|ending)\s*\w+\s+\d{1,2},?\s*(\d{4})|"
    r"period\s*of\s*report[:\s]*(\d{4})\s*$)",
    re.IGNORECASE | re.MULTILINE,
)

# Document type extraction
_DOC_TYPE_FROM_PATH = re.compile(r"(10-K|10-Q|8-K|S-1|DEF\s*14A)", re.IGNORECASE)


def _generate_chunk_id(ticker: str, source_file: str, index: int) -> str:
    """
    Generate a deterministic, unique chunk identifier.
    Format: {TICKER}_{short_hash}_{index}
    """
    hash_input = f"{ticker}:{source_file}:{index}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    return f"{ticker}_{short_hash}_{index:04d}"


def extract_ticker(file_path: str | Path, content: str = "") -> str:
    """
    Extract the company ticker symbol from file path or content.

    Priority: file path directory structure > file name > content scan.
    """
    path_str = str(file_path)

    # Try extracting from directory structure (e.g., data/AAPL/10-K/...)
    match = _TICKER_FROM_PATH.search(path_str)
    if match:
        ticker = match.group(1).upper()
        logger.debug("Extracted ticker from path: %s", ticker)
        return ticker

    # Try extracting from file name
    file_name = Path(path_str).stem.upper()
    for known in ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"):
        if known in file_name:
            logger.debug("Extracted ticker from filename: %s", known)
            return known

    # Try content-based extraction (look for company name patterns)
    content_upper = content[:5000].upper()  # Only scan first 5000 chars
    ticker_map = {
        "APPLE INC": "AAPL",
        "MICROSOFT CORP": "MSFT",
        "NVIDIA CORP": "NVDA",
        "ALPHABET INC": "GOOGL",
        "AMAZON.COM": "AMZN",
        "META PLATFORMS": "META",
        "TESLA INC": "TSLA",
    }
    for name, ticker in ticker_map.items():
        if name in content_upper:
            logger.debug("Extracted ticker from content: %s", ticker)
            return ticker

    logger.warning("Could not extract ticker from: %s", path_str)
    return "UNKNOWN"


def extract_fiscal_year(file_path: str | Path, content: str = "") -> str:
    """
    Extract the fiscal year from the file path or filing content.

    Priority: accession number heuristic > content pattern > path pattern.
    """
    path_str = str(file_path)

    # 1. Try SEC accession number (most reliable for SEC filings)
    # Format: XXXXXXXXXX-YY-ZZZZZZ where YY is 2-digit filing year
    acc_match = _FISCAL_YEAR_FROM_ACCESSION.search(path_str)
    if acc_match:
        short_year = int(acc_match.group(1))
        year = str(2000 + short_year) if short_year < 100 else str(short_year)
        logger.debug("Extracted fiscal year from accession number: %s", year)
        return year

    # 2. Try content-based extraction
    if content:
        match = _FISCAL_YEAR_FROM_CONTENT.search(content[:20000])
        if match:
            year = next(g for g in match.groups() if g)
            logger.debug("Extracted fiscal year from content: %s", year)
            return year

    # 3. Try path-based extraction (fallback)
    match = _FISCAL_YEAR_FROM_PATH.search(path_str)
    if match:
        year = match.group(1)
        logger.debug("Extracted fiscal year from path: %s", year)
        return year

    logger.warning("Could not extract fiscal year from: %s", path_str)
    return "UNKNOWN"


def extract_doc_type(file_path: str | Path, content: str = "") -> str:
    """Extract the SEC document type (10-K, 10-Q, etc.) from path or content."""
    path_str = str(file_path)

    # Path-based extraction
    match = _DOC_TYPE_FROM_PATH.search(path_str)
    if match:
        doc_type = match.group(1).upper().replace(" ", "-")
        logger.debug("Extracted doc type from path: %s", doc_type)
        return doc_type

    # Content-based extraction (look for <TYPE> tags in SEC SGML)
    if content:
        type_match = re.search(r"<TYPE>\s*(10-K|10-Q|8-K|S-1)", content[:5000], re.IGNORECASE)
        if type_match:
            doc_type = type_match.group(1).upper()
            logger.debug("Extracted doc type from content: %s", doc_type)
            return doc_type

    logger.warning("Could not extract document type from: %s", path_str)
    return "UNKNOWN"


def identify_section(text: str) -> str:
    """
    Identify the SEC 10-K section that the text belongs to.

    Scans for Item headers and returns the most recently matched section.
    If no section header is found, returns 'General'.
    """
    last_section = "General"
    for pattern, section_name in _SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            last_section = section_name
            break  # First match in text = current section

    logger.debug("Identified section: %s", last_section)
    return last_section


def contains_table(text: str) -> bool:
    """Check if the text contains a Markdown-formatted table."""
    has_table = bool(re.search(r"\|.*\|.*\|", text))
    return has_table


def extract_metadata(
    file_path: str | Path,
    content: str = "",
    chunk_text: str = "",
    chunk_index: int = 0,
) -> dict:
    """
    Extract the complete metadata dictionary for a filing chunk.

    Args:
        file_path: Path to the source SEC filing.
        content: Raw filing content (for content-based extraction).
        chunk_text: The specific text chunk (for section + table detection).
        chunk_index: Numeric index of this chunk within the document.

    Returns:
        Dictionary with all mandatory metadata fields:
            ticker, fiscal_year, doc_type, source_file,
            section, contains_table, chunk_id, page_number
    """
    ticker = extract_ticker(file_path, content)
    fiscal_year = extract_fiscal_year(file_path, content)
    doc_type = extract_doc_type(file_path, content)
    source_file = Path(file_path).name
    section = identify_section(chunk_text) if chunk_text else "General"
    has_table = contains_table(chunk_text) if chunk_text else False
    chunk_id = _generate_chunk_id(ticker, source_file, chunk_index)

    metadata = {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "doc_type": doc_type,
        "source_file": source_file,
        "section": section,
        "contains_table": has_table,
        "chunk_id": chunk_id,
        "page_number": None,  # Placeholder — SEC filings are not page-numbered
    }

    logger.info(
        "Metadata extracted: ticker=%s, year=%s, doc=%s, section=%s, tables=%s",
        ticker,
        fiscal_year,
        doc_type,
        section,
        has_table,
    )
    return metadata
