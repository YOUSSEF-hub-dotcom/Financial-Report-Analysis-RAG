"""
Financial Text Cleaner — normalizes SEC filing text while preserving financial data integrity.

CRITICAL SAFEGUARDS:
  - Negative sign parentheses (150) are NEVER converted or dropped
  - Currency symbols ($, €), percentages (%), and scale notations (M, B, K) preserved
  - Column spacing inside Markdown tables preserved for numerical alignment
  - HTML artifacts stripped; redundant whitespace collapsed
"""

import re

from config.logging_config import get_logger

logger = get_logger("ingestion.cleaner")

# --- Financial notation patterns to PROTECT during cleaning ---

# Parenthesized negatives: (1,234), (150.00), $(500), etc.
_PAREN_NEGATIVE = re.compile(r"\([\d,.\s$€£%]+\)")

# Currency symbols with optional amounts: $1,234.56, €500M, £1.2B
_CURRENCY_AMOUNT = re.compile(r"[$€£¥]\s*[\d,.\s]+[MBKmbk]?")

# Scale notation: 1,234M, 5.6B, 100K, etc. (number followed by scale suffix)
_SCALE_NOTATION = re.compile(r"\b[\d,]+\.?\d*\s*[MBKmbk]\b")

# Percentage values: 12.5%, (3.2)%, -1.5%
_PERCENTAGE = re.compile(r"\(?[\d.,]+\s*%\)?")

# --- Patterns to REMOVE ---

# Residual HTML/XML tags (after parser has done its job)
_HTML_TAG = re.compile(r"<[^>]+>")

# CSS style attributes and inline styles
_STYLE_ATTR = re.compile(r'style\s*=\s*"[^"]*"', re.IGNORECASE)

# Multiple consecutive whitespace (but NOT newlines or table pipes)
_MULTI_SPACE = re.compile(r"[^\S\n|]+")

# Multiple blank lines
_MULTI_BLANK = re.compile(r"\n{3,}")

# SEC boilerplate footer/header markers
_SEC_BOILERPLATE = re.compile(
    r"(?:(?:Filing\s+Date|Filed\s+on|Date\s+of\s+Event|Document\s+Center|"
    r"EDGAR\s+Filing|Accession\s+Number|Submitter|Contact)[^\n]*)",
    re.IGNORECASE,
)


def _protect_financial_notation(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace financial notation patterns with unique placeholders
    to prevent accidental corruption during whitespace normalization.
    Uses re.sub with a lambda to avoid index shifting issues.
    Returns cleaned text and a restoration map.
    """
    restore_map: dict[str, str] = {}
    counter = 0

    # Order matters — protect compound patterns first
    for pattern in (_PAREN_NEGATIVE, _CURRENCY_AMOUNT, _PERCENTAGE, _SCALE_NOTATION):
        def _make_replacer(pat_counter: int) -> callable:
            def _replacer(match: re.Match) -> str:
                nonlocal counter
                token = f"§FIN{counter}§"
                restore_map[token] = match.group(0)
                counter += 1
                return token
            return _replacer

        text = pattern.sub(_make_replacer(counter), text)

    return text, restore_map


def _restore_financial_notation(text: str, restore_map: dict[str, str]) -> str:
    """Replace all §FINn§ placeholders with their original financial notation."""
    for token, original in restore_map.items():
        text = text.replace(token, original)
    return text


def _separate_table_regions(text: str) -> tuple[str, list[tuple[int, str]]]:
    """
    Identify Markdown table regions and separate them from regular text.
    This prevents whitespace normalization from collapsing table column alignment.

    Returns:
        - text with table regions replaced by placeholders
        - list of (placeholder_index, table_content) for restoration
    """
    tables: list[tuple[int, str]] = []
    counter = 0

    # Match sequences of lines that look like Markdown tables (lines with |)
    lines = text.split("\n")
    result_lines: list[str] = []
    in_table = False
    table_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        is_table_line = (
            stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2
        )

        if is_table_line:
            in_table = True
            table_buffer.append(line)
        else:
            if in_table and table_buffer:
                # End of table block — save it
                table_content = "\n".join(table_buffer)
                placeholder = f"§TABLE_{counter}§"
                tables.append((counter, table_content))
                result_lines.append(placeholder)
                counter += 1
                table_buffer = []
            in_table = False
            result_lines.append(line)

    # Handle table at end of text
    if table_buffer:
        table_content = "\n".join(table_buffer)
        tables.append((counter, table_content))
        result_lines.append(f"§TABLE_{counter}§")

    return "\n".join(result_lines), tables


def _restore_tables(text: str, tables: list[tuple[int, str]]) -> str:
    """Replace table placeholders with their original Markdown content."""
    for idx, table_content in tables:
        text = text.replace(f"§TABLE_{idx}§", table_content)
    return text


def clean_financial_text(text: str) -> str:
    """
    Clean raw SEC filing text while preserving all financial data integrity.

    Processing pipeline:
        1. Protect financial notation (parenthesized negatives, currencies, %)
        2. Separate and protect Markdown table regions
        3. Strip residual HTML tags and CSS
        4. Remove SEC boilerplate headers/footers
        5. Normalize whitespace (outside table regions)
        6. Restore protected regions

    Args:
        text: Raw or semi-cleaned text from HTML parser.

    Returns:
        Cleaned text safe for chunking with all financial notation intact.
    """
    if not text or not text.strip():
        logger.warning("Empty text passed to clean_financial_text")
        return ""

    logger.info("Cleaning text — length: %d chars", len(text))

    # Step 1: Protect financial notation
    protected_text, restore_map = _protect_financial_notation(text)
    logger.debug("Protected %d financial notation tokens", len(restore_map))

    # Step 2: Separate table regions
    table_text, table_regions = _separate_table_regions(protected_text)
    logger.debug("Separated %d Markdown table regions", len(table_regions))

    # Step 3: Strip HTML tags and style attributes
    cleaned = _STYLE_ATTR.sub("", table_text)
    cleaned = _HTML_TAG.sub(" ", cleaned)

    # Step 4: Decode common HTML entity artifacts
    entity_map = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&apos;": "'",
        "&#160;": " ",
        "&nbsp;": " ",
    }
    for entity, char in entity_map.items():
        cleaned = cleaned.replace(entity, char)

    # Step 5: Remove SEC boilerplate
    cleaned = _SEC_BOILERPLATE.sub("", cleaned)

    # Step 6: Normalize whitespace (outside table regions)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    cleaned = _MULTI_BLANK.sub("\n\n", cleaned)
    cleaned = cleaned.strip()

    # Step 7: Restore table regions
    cleaned = _restore_tables(cleaned, table_regions)

    # Step 8: Restore financial notation
    cleaned = _restore_financial_notation(cleaned, restore_map)

    logger.info("Cleaning complete — output length: %d chars", len(cleaned))
    return cleaned
