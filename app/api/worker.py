"""
Background Task Worker for post-response processing.

Uses FastAPI BackgroundTasks to run guardrail verification, cache updates,
and metric logging off the main HTTP thread so API responses remain fast.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

from config.logging_config import get_logger

# Ensure src subdirectories are importable
_src_root = Path(__file__).resolve().parent.parent.parent / "src"
for _subdir in ("1_ingestion", "2_generation"):
    _p = str(_src_root / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)
_src_root_str = str(_src_root)
_project_root_str = str(_src_root.parent)
for _p in (_src_root_str, _project_root_str):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = get_logger("api.worker")


# ---------------------------------------------------------------------------
# Background guardrail runner
# ---------------------------------------------------------------------------

async def run_guardrail_background(
    query: str,
    answer: str,
    extracted_raw_data: str,
    parsed_output: Any = None,
) -> dict[str, Any]:
    """
    Run deep numerical guardrail verification as a background task.

    Args:
        query: Original user query.
        answer: Generated answer text.
        extracted_raw_data: Verbatim extracted facts from context.
        parsed_output: Optional ConsolidatedFinancialAnswer for caching.

    Returns:
        Dict with guardrail outcome (passed, verified_claims, failed_claims, etc.).
    """
    from async_guardrail import AsyncGuardrail

    t0 = time.time()
    guardrail = AsyncGuardrail()

    try:
        result = await guardrail.check(
            query=query,
            answer=answer,
            extracted_raw_data=extracted_raw_data,
            parsed_output=parsed_output,
        )
        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            "Background guardrail: passed=%s verified=%d failed=%d elapsed=%.1fms",
            result.get("passed"),
            len(result.get("verified_claims", [])),
            len(result.get("failed_claims", [])),
            elapsed_ms,
        )
        return result
    except Exception as exc:
        logger.warning("Background guardrail failed: %s", exc)
        return {
            "passed": False,
            "verified_claims": [],
            "failed_claims": [],
            "final_output": answer,
            "cache_written": False,
            "detail": f"Guardrail error: {exc}",
        }
    finally:
        guardrail.close()


# ---------------------------------------------------------------------------
# Background metric logging
# ---------------------------------------------------------------------------

def log_metrics_background(metrics: dict[str, Any]) -> None:
    """
    Log pipeline metrics as a fire-and-forget background task.

    Args:
        metrics: Dict of metric names to values (int/float/str).
    """
    try:
        import mlflow

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            elif isinstance(value, str):
                mlflow.log_param(key, value)
        logger.debug("Background metrics logged: %s", metrics)
    except Exception as exc:
        logger.warning("Background metric logging failed: %s", exc)
