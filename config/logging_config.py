"""
Unified Structured JSON Logger for the Financial RAG system.

Provides a reusable get_logger() function that emits JSON-formatted log events
compatible with ELK/Datadog ingestion pipelines.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredJSONFormatter(logging.Formatter):
    """Formats log records as structured JSON strings for observability pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data
        return json.dumps(log_entry, default=str)


def get_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    """
    Returns a named logger that writes structured JSON to both stdout and a log file.

    Args:
        name: Logger namespace (e.g., 'ingestion.parser').
        log_dir: Directory for log file output. Defaults to <project_root>/logs/.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = StructuredJSONFormatter()

    # Console handler — INFO level for operational visibility
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — DEBUG level for full trace logging
    if log_dir is None:
        log_dir = str(Path(__file__).resolve().parent.parent / "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "rag_events.log")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
