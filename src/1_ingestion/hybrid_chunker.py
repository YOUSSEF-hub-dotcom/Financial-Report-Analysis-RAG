"""
Hybrid Chunker — 3-Tier token-bounded chunking engine for SEC filings.

Tier 1: Section-level splitting on SEC Item boundaries.
Tier 2: Table & footnote isolation as atomic chunks (never split).
Tier 3: Token-bounded recursive text splitting (512-768 tokens, 10-15% overlap).
"""

import re
import uuid
from pathlib import Path

import tiktoken

from config.logging_config import get_logger
from config.settings import CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS, CHUNK_OVERLAP_RATIO

logger = get_logger("ingestion.chunker")

# SEC Item boundary patterns for Tier 1 section splitting
_SECTION_BOUNDARIES = re.compile(
    r"(?=(?:^|\n)\s*(?:Item\s+\d+[A-Z]?[\.\:]\s|Part\s+(?:I{1,3}V?|IV)\b))",
    re.IGNORECASE,
)

# Markdown table pattern — lines starting/ending with pipes
_TABLE_LINE = re.compile(r"^\s*\|.+\|\s*$")

# Footnote pattern — common SEC footnote markers
_FOOTNOTE_PATTERN = re.compile(
    r"(?:^\s*\(\d+\)|^\s*\[\d+\]|^\s*\*\s|^\s*\d+\.\s)",
    re.MULTILINE,
)

# Table placeholder pattern from parser
_TABLE_PLACEHOLDER = re.compile(r"%%TABLE_\d+%%")


def _get_token_encoder() -> tiktoken.Encoding:
    """Load tiktoken encoder for cl100k_base (used by most modern models)."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder: tiktoken.Encoding | None = None) -> int:
    """Count exact token count using tiktoken encoder."""
    if encoder is None:
        encoder = _get_token_encoder()
    return len(encoder.encode(text))


def _split_on_sections(text: str) -> list[dict]:
    """
    Tier 1: Split text on SEC Item boundaries.

    Returns list of dicts with 'section_name' and 'text' keys.
    """
    parts = _SECTION_BOUNDARIES.split(text)
    sections = []

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue

        # Extract section name from the first line
        first_line = stripped.split("\n", 1)[0].strip()
        section_match = re.match(
            r"(Item\s+\d+[A-Z]?[\.\:]\s*.+|Part\s+(?:I{1,3}V?|IV)\b.*)",
            first_line,
            re.IGNORECASE,
        )
        section_name = section_match.group(0) if section_match else "General"

        sections.append({
            "section_name": section_name,
            "text": stripped,
        })

    logger.debug("Tier 1: Split document into %d sections", len(sections))
    return sections


def _extract_table_chunks(text: str) -> tuple[list[str], str]:
    """
    Tier 2: Extract Markdown tables and their immediately following footnotes
    as atomic chunks. Returns (table_chunks, remaining_text).
    """
    lines = text.split("\n")
    table_chunks = []
    non_table_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        is_table_start = bool(_TABLE_LINE.match(line))

        if is_table_start:
            # Collect contiguous table lines
            table_buffer = [line]
            i += 1
            while i < len(lines) and bool(_TABLE_LINE.match(lines[i])):
                table_buffer.append(lines[i])
                i += 1

            # Collect immediately following footnote lines
            while i < len(lines) and bool(_FOOTNOTE_PATTERN.match(lines[i])):
                table_buffer.append(lines[i])
                i += 1

            table_text = "\n".join(table_buffer)
            table_chunks.append(table_text)
        else:
            non_table_lines.append(line)
            i += 1

    remaining_text = "\n".join(non_table_lines)
    logger.debug(
        "Tier 2: Extracted %d atomic table chunks, remaining text: %d chars",
        len(table_chunks),
        len(remaining_text),
    )
    return table_chunks, remaining_text


def _recursive_token_split(
    text: str,
    encoder: tiktoken.Encoding,
    max_tokens: int = CHUNK_MAX_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
    overlap_tokens: int | None = None,
) -> list[str]:
    """
    Tier 3: Recursively split text into token-bounded chunks.

    Split priority: paragraph break -> sentence boundary -> word boundary -> character.
    Enforces strict token count limits using tiktoken.
    """
    if overlap_tokens is None:
        overlap_tokens = int(max_tokens * CHUNK_OVERLAP_RATIO)

    if not text.strip():
        return []

    total_tokens = count_tokens(text, encoder)
    if total_tokens <= max_tokens:
        return [text.strip()]

    # Split hierarchy: paragraphs -> sentences -> words
    split_patterns = [
        re.compile(r"\n{2,}"),           # Paragraph breaks
        re.compile(r"(?<=[.!?])\s+"),    # Sentence boundaries
        re.compile(r"\s+"),              # Word boundaries
    ]

    chunks = []
    for pattern in split_patterns:
        segments = pattern.split(text)
        # Filter empty segments
        segments = [s.strip() for s in segments if s.strip()]

        if len(segments) <= 1:
            continue

        current_chunk_parts = []
        current_token_count = 0

        for segment in segments:
            segment_tokens = count_tokens(segment, encoder)

            if current_token_count + segment_tokens <= max_tokens:
                current_chunk_parts.append(segment)
                current_token_count += segment_tokens
            else:
                # Flush current chunk if it meets minimum size
                if current_chunk_parts and current_token_count >= min_tokens:
                    chunks.append(" ".join(current_chunk_parts))

                # Handle segments larger than max_tokens — hard split
                if segment_tokens > max_tokens:
                    hard_chunks = _hard_split_segment(segment, encoder, max_tokens)
                    chunks.extend(hard_chunks)
                    current_chunk_parts = []
                    current_token_count = 0
                else:
                    # Start new chunk with overlap from previous
                    overlap_parts = []
                    overlap_count = 0
                    for prev_part in reversed(current_chunk_parts):
                        prev_tokens = count_tokens(prev_part, encoder)
                        if overlap_count + prev_tokens <= overlap_tokens:
                            overlap_parts.insert(0, prev_part)
                            overlap_count += prev_tokens
                        else:
                            break

                    current_chunk_parts = overlap_parts + [segment]
                    current_token_count = overlap_count + segment_tokens

        # Flush remaining
        if current_chunk_parts and current_token_count >= min_tokens:
            chunks.append(" ".join(current_chunk_parts))
        elif current_chunk_parts and chunks:
            # Merge small final chunk with previous if within limits
            last = chunks[-1]
            merged = last + " " + " ".join(current_chunk_parts)
            if count_tokens(merged, encoder) <= max_tokens:
                chunks[-1] = merged
            else:
                chunks.append(" ".join(current_chunk_parts))

        # If this split level produced valid chunks, use it
        if len(chunks) > 1:
            break

    # Fallback: if no splits worked, return the text as a single chunk
    if not chunks and text.strip():
        chunks = [text.strip()]

    return chunks


def _hard_split_segment(
    text: str, encoder: tiktoken.Encoding, max_tokens: int
) -> list[str]:
    """Force-split a segment that exceeds max_tokens by encoding and slicing."""
    tokens = encoder.encode(text)
    chunks = []
    for start in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[start : start + max_tokens]
        chunks.append(encoder.decode(chunk_tokens).strip())
    return chunks


def chunk_document(
    text: str,
    tables: list[str] | None = None,
    file_path: str | Path = "",
    metadata_base: dict | None = None,
) -> list[dict]:
    """
    Execute the full 3-Tier chunking pipeline on parsed document content.

    Args:
        text: Cleaned text content from Stage 1 parser.
        tables: List of Markdown table strings extracted by parser.
        file_path: Source file path for metadata.
        metadata_base: Base metadata dict from metadata_extractor.

    Returns:
        List of chunk dicts, each containing:
            'chunk_id', 'text', 'chunk_type' ('table'|'text'),
            'token_count', 'metadata' (with section, contains_table, etc.)
    """
    encoder = _get_token_encoder()
    all_chunks: list[dict] = []

    if metadata_base is None:
        metadata_base = {}

    # --- Tier 2: Extract table atomic chunks from text ---
    table_chunks_from_text, remaining_text = _extract_table_chunks(text)

    # Combine parser-extracted tables with inline tables
    all_table_chunks = list(tables or []) + table_chunks_from_text

    # Create atomic table chunks
    for idx, table_text in enumerate(all_table_chunks):
        if not table_text.strip():
            continue
        chunk_id = f"{metadata_base.get('ticker', 'UNK')}_tbl_{uuid.uuid4().hex[:12]}_{idx:04d}"
        chunk = {
            "chunk_id": chunk_id,
            "text": table_text.strip(),
            "chunk_type": "table",
            "token_count": count_tokens(table_text, encoder),
            "metadata": {
                **metadata_base,
                "contains_table": True,
                "chunk_type": "table",
            },
        }
        all_chunks.append(chunk)

    logger.info(
        "Tier 2: Created %d atomic table chunks",
        len([c for c in all_chunks if c["chunk_type"] == "table"]),
    )

    # --- Tier 1: Split remaining text on section boundaries ---
    sections = _split_on_sections(remaining_text)

    # --- Tier 3: Token-bounded splitting within each section ---
    text_chunk_idx = 0
    for section in sections:
        section_text = section["text"]
        section_name = section["section_name"]

        # Tier 3: Recursive token splitting
        sub_chunks = _recursive_token_split(section_text, encoder)

        for sub_text in sub_chunks:
            chunk_id = f"{metadata_base.get('ticker', 'UNK')}_txt_{uuid.uuid4().hex[:12]}_{text_chunk_idx:04d}"
            chunk = {
                "chunk_id": chunk_id,
                "text": sub_text,
                "chunk_type": "text",
                "token_count": count_tokens(sub_text, encoder),
                "metadata": {
                    **metadata_base,
                    "section": section_name,
                    "contains_table": False,
                    "chunk_type": "text",
                },
            }
            all_chunks.append(chunk)
            text_chunk_idx += 1

    logger.info(
        "Chunking complete: %d total chunks (%d text, %d table)",
        len(all_chunks),
        len([c for c in all_chunks if c["chunk_type"] == "text"]),
        len([c for c in all_chunks if c["chunk_type"] == "table"]),
    )
    return all_chunks
