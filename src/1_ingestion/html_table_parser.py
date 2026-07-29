"""
HTML Table Parser — converts SEC filing HTML into structured Markdown.

Handles:
  - SGML-wrapped SEC filings (extracts the main 10-K document)
  - Complex HTML <table> elements → clean Markdown tables
  - Inline XBRL tags stripped from text content
  - Tables treated as atomic blocks to prevent fragmentation
"""

import re
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag

from config.logging_config import get_logger

logger = get_logger("ingestion.parser")

# --- Regex patterns for SEC SGML wrapper extraction ---
_SEC_DOC_START = re.compile(r"<DOCUMENT>\s*\n<TYPE>10-K\b", re.IGNORECASE)
_SEC_DOC_END = re.compile(r"</DOCUMENT>", re.IGNORECASE)

# XBRL inline tags to strip (preserve text content only)
_XBRL_TAG_PATTERN = re.compile(
    r"</?(?:ix:[A-Za-z]+|xbrli:[A-Za-z]+|link:[A-Za-z]+|xlink:[A-Za-z]+)[^>]*>"
)

# HTML entity replacements for common SEC encoding artifacts
_HTML_ENTITY_MAP = {
    "&#160;": " ",
    "&nbsp;": " ",
    "&#8217;": "'",
    "&#8220;": '"',
    "&#8221;": '"',
    "&#8211;": "–",
    "&#8212;": "—",
    "&#58;": ":",
    "&#8230;": "…",
}


def _extract_main_document(raw_text: str) -> str:
    """Extract the 10-K document body from an SEC SGML wrapper file."""
    match = _SEC_DOC_START.search(raw_text)
    if not match:
        logger.warning("No <DOCUMENT><TYPE>10-K block found; attempting raw HTML parse")
        return raw_text

    start = match.start()
    end_match = _SEC_DOC_END.search(raw_text, start + 1)
    if not end_match:
        logger.warning("No closing </DOCUMENT> tag found after 10-K start")
        return raw_text[start:]

    return raw_text[start:end_match.end()]


def _html_entities_to_text(text: str) -> str:
    """Replace numeric and named HTML entities with their Unicode equivalents."""
    for entity, char in _HTML_ENTITY_MAP.items():
        text = text.replace(entity, char)
    return text


def _table_to_markdown(table_tag: Tag) -> str:
    """
    Convert a single BeautifulSoup <table> Tag into a Markdown table string.
    Uses pandas.to_markdown(index=False) for clean column alignment.
    """
    try:
        df_list = pd.read_html(StringIO(str(table_tag)), header=0)
        if not df_list:
            return ""
        df = df_list[0]
        # Drop fully-empty rows that often result from decorative HTML rows
        df = df.dropna(how="all").reset_index(drop=True)
        if df.empty:
            return ""
        # Clean column names — strip whitespace and newlines
        df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
        md_table = df.to_markdown(index=False, tablefmt="pipe")
        logger.debug("Converted table with %d rows and %d columns", len(df), len(df.columns))
        return md_table
    except Exception as exc:
        logger.warning("Table conversion failed: %s", exc)
        return ""


def parse_sec_filing(file_path: str | Path) -> dict:
    """
    Parse an SEC filing file and extract structured content.

    Args:
        file_path: Path to the raw .txt or .html SEC filing.

    Returns:
        Dictionary with keys:
            - 'raw_html': The extracted HTML string (after SGML extraction).
            - 'clean_html': BeautifulSoup-parsed cleaned HTML.
            - 'tables': List of Markdown-formatted table strings.
            - 'text_content': Full text content with tables replaced by placeholders.
            - 'table_positions': List of (start, end) character offsets for each table placeholder.
            - 'table_count': Total number of tables found.
    """
    path = Path(file_path)
    logger.info("Parsing SEC filing: %s", path.name)

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    html_content = _extract_main_document(raw_text)

    # Parse with BeautifulSoup — lxml for speed + tolerance
    soup = BeautifulSoup(html_content, "lxml")

    # Strip XBRL inline tags that clutter text but keep their content
    for tag in soup.find_all(re.compile(r"^(ix|xbrli|link|xbrldi):")):
        tag.unwrap()

    # Remove <script>, <style>, <head> elements entirely
    for tag_name in ("script", "style", "head"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # --- Extract and convert tables to Markdown ---
    tables: list[str] = []
    for table_tag in soup.find_all("table"):
        md = _table_to_markdown(table_tag)
        if md.strip():
            tables.append(md)
            # Replace the HTML table with a placeholder in the soup
            table_tag.replace_with(f"\n%%TABLE_{len(tables) - 1}%%\n")

    # Get text content with table placeholders preserved
    text_content = soup.get_text(separator=" ", strip=True)
    text_content = _html_entities_to_text(text_content)

    # Collapse redundant whitespace (but not around table placeholders)
    text_content = re.sub(r"[ \t]+", " ", text_content)
    text_content = re.sub(r"\n{3,}", "\n\n", text_content)

    # Record table placeholder positions for downstream re-insertion
    table_positions: list[tuple[int, int]] = []
    for i in range(len(tables)):
        pattern = re.compile(rf"%%TABLE_{i}%%")
        m = pattern.search(text_content)
        if m:
            table_positions.append((m.start(), m.end()))

    result = {
        "raw_html": html_content,
        "clean_html": str(soup),
        "tables": tables,
        "text_content": text_content,
        "table_positions": table_positions,
        "table_count": len(tables),
    }

    logger.info(
        "Parsed %s — %d tables extracted, text length: %d chars",
        path.name,
        len(tables),
        len(text_content),
    )
    return result


def load_raw_file(file_path: str | Path) -> str:
    """
    Load a raw SEC filing as a string. Handles both .txt and .html extensions.
    Returns the full raw text content.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"SEC filing not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")
