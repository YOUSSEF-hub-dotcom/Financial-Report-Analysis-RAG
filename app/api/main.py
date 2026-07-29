"""
FastAPI Production Backend for Financial RAG System.

Endpoints:
  - POST /api/v1/chat         → Main query (sync)
  - POST /api/v1/chat/stream  → SSE streaming
  - POST /api/v1/documents/upload → File upload + ingestion
  - GET  /health              → Service health check

Lifespan manages FinancialRAGPipeline init/close.
Structured JSON logging on every request via middleware.
"""

import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from config.logging_config import get_logger
from config.settings import DATA_DIR, SUPPORTED_TICKERS

# Ensure src subdirectories are importable (numbered names not valid packages)
_src_root = Path(__file__).resolve().parent.parent.parent / "src"
for _subdir in ("1_ingestion", "2_generation"):
    _p = str(_src_root / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Also add src/ itself and project root
_src_root_str = str(_src_root)
_project_root_str = str(_src_root.parent)
for _p in (_src_root_str, _project_root_str):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.api.parsers import APIFileParser
from app.api.schemas import (
    CacheFlushResponse,
    ChatQueryRequest,
    ChatQueryResponse,
    DocumentUploadResponse,
    ErrorResponse,
    GuardrailStatus,
    HealthCheckResponse,
    HealthServiceStatus,
    SourceCitation,
)
from app.api.worker import run_guardrail_background
from pipeline import FinancialRAGPipeline

logger = get_logger("api.main")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: init pipeline + warm-up. Shutdown: close connections."""
    logger.info("API starting up — initializing FinancialRAGPipeline")
    pipeline = FinancialRAGPipeline()
    app.state.pipeline = pipeline
    app.state.warmup_completed = False

    # Warm-up phase — pre-load models & establish connections
    try:
        await _run_warmup(pipeline)
        app.state.warmup_completed = True
    except Exception as exc:
        logger.warning("Warm-up failed (non-blocking): %s", exc)

    yield

    logger.info("API shutting down — closing pipeline")
    pipeline.close()


app = FastAPI(
    title="Financial RAG API",
    version="1.0.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


async def _run_warmup(pipeline: "FinancialRAGPipeline") -> None:
    """Pre-load embedding model and pre-warm DB connections.

    Runs a lightweight embedding call to load nomic-embed-text-v1.5 into
    GPU memory (the #1 cold-start bottleneck), then pings Qdrant, MongoDB
    and Redis so the first user query sees sub-second response times.
    All errors are caught and logged — a warm-up failure never crashes
    the server.
    """
    # 1. Embedding model — the main cold-start cost (~2–3 min first load)
    logger.info("Warm-up: loading embedding model...")
    pipeline._embedding_engine.embed_single("warmup")
    logger.info("Warm-up: embedding model loaded")

    # 2. Qdrant persistent client connection
    logger.info("Warm-up: connecting to Qdrant...")
    pipeline._qdrant_indexer.count_points()
    logger.info("Warm-up: Qdrant ready")

    # 3. MongoDB connection
    logger.info("Warm-up: connecting to MongoDB...")
    pipeline._mongo_indexer.count_documents()
    logger.info("Warm-up: MongoDB ready")

    # 4. Redis cache connection (lazy-init)
    logger.info("Warm-up: connecting to Redis...")
    if pipeline._cache is not None:
        pipeline._cache._get_client()
    logger.info("Warm-up: Redis ready")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTP %d on %s: %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=exc.detail, error_code=f"HTTP_{exc.status_code}").model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error. Please try again later.",
            error_code="INTERNAL_ERROR",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Middleware: structured request logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        "api_request method=%s path=%s status=%d elapsed_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pipeline(request: Request) -> FinancialRAGPipeline:
    """Get pipeline from app state (raises 503 if unavailable)."""
    pipe: FinancialRAGPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipe


def _build_sources(result: dict[str, Any]) -> list[SourceCitation]:
    """Extract source citations from pipeline result."""
    sources: list[SourceCitation] = []
    parsed = result.get("parsed")
    if parsed is not None and hasattr(parsed, "sources"):
        for src_str in parsed.sources:
            parts = src_str.split(" - ")
            sources.append(SourceCitation(
                chunk_id=src_str,
                score=1.0,
                ticker=parts[0] if len(parts) > 0 else "UNKNOWN",
                fiscal_year=parts[1] if len(parts) > 1 else "UNKNOWN",
                section=parts[2] if len(parts) > 2 else "General",
                text_snippet=src_str[:300],
            ))
    return sources


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/chat", response_model=ChatQueryResponse)
async def chat(request: ChatQueryRequest, fastapi_request: Request):
    """Main RAG query endpoint."""
    t0 = time.time()
    pipe = _get_pipeline(fastapi_request)

    logger.info(
        "chat_request query=%.80s ticker=%s year=%s session=%s",
        request.user_query, request.ticker, request.fiscal_year, request.session_id,
    )

    result = pipe.query(
        user_query=request.user_query,
        ticker=request.ticker,
        fiscal_year=request.fiscal_year,
        session_id=request.session_id,
    )

    elapsed = (time.time() - t0) * 1000
    answer = result.get("raw_output", "")
    parsed = result.get("parsed")
    if parsed is not None and hasattr(parsed, "answer"):
        answer = parsed.answer

    sources = _build_sources(result)

    return ChatQueryResponse(
        answer=answer,
        sources=sources,
        execution_time_ms=round(elapsed, 2),
        guardrail_status=GuardrailStatus(passed=True),
        model_used=result.get("model_used", "unknown"),
        cache_hit=result.get("cache_hit", False),
    )


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatQueryRequest, fastapi_request: Request):
    """SSE streaming endpoint for real-time token generation."""
    pipe = _get_pipeline(fastapi_request)

    from sse_starlette.sse import EventSourceResponse

    async def event_generator() -> AsyncIterator[dict]:
        async for token in pipe.query_stream(
            user_query=request.user_query,
            ticker=request.ticker,
            fiscal_year=request.fiscal_year,
            session_id=request.session_id,
        ):
            yield {"event": "token", "data": token}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    fastapi_request: Request,
    file: UploadFile = File(...),
):
    """Upload an SEC filing document and trigger the ingestion pipeline."""
    ticker = fastapi_request.query_params.get("ticker", "UNKNOWN")
    fiscal_year = fastapi_request.query_params.get("fiscal_year", "UNKNOWN")

    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in APIFileParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(APIFileParser.SUPPORTED_EXTENSIONS))}",
        )

    raw_bytes = await file.read()

    # Save to data dir
    save_dir = DATA_DIR / ticker / "10-K" / fiscal_year
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file.filename
    save_path.write_bytes(raw_bytes)

    # Run ingestion in background
    task_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(
        _ingest_document,
        save_path,
        ticker,
        fiscal_year,
        task_id,
    )

    logger.info(
        "Document upload queued: file=%s ticker=%s year=%s task=%s",
        file.filename, ticker, fiscal_year, task_id,
    )

    return DocumentUploadResponse(
        filename=file.filename,
        ticker=ticker,
        fiscal_year=fiscal_year,
        task_id=task_id,
    )


async def _ingest_document(file_path: Path, ticker: str, fiscal_year: str, task_id: str):
    """Run the full ingestion pipeline on an uploaded document (background)."""
    from cleaning import clean_financial_text
    from metadata_extractor import extract_metadata
    from hybrid_chunker import chunk_document
    from database_indexer import EmbeddingEngine, MongoDBIndexer, QdrantIndexer

    logger.info("Ingestion task %s started: %s", task_id, file_path.name)

    try:
        file_bytes = file_path.read_bytes()
        filename = file_path.name
        parser = APIFileParser()
        parsed = parser.parse_file(file_bytes, filename)

        cleaned = clean_financial_text(parsed["text_content"])
        base_metadata = extract_metadata(
            file_path=file_path,
            content=parsed["raw_html"],
            chunk_text=cleaned[:2000],
            chunk_index=0,
        )
        chunks = chunk_document(
            text=cleaned,
            tables=parsed["tables"],
            file_path=file_path,
            metadata_base=base_metadata,
        )

        if not chunks:
            logger.warning("Ingestion task %s produced zero chunks", task_id)
            return

        embedder = EmbeddingEngine()
        mongo = MongoDBIndexer()
        qdrant = QdrantIndexer()

        all_embeddings = embedder.embed([c["text"] for c in chunks])
        qdrant_count = qdrant.upsert_vectors(chunks, all_embeddings)
        mongo_count = mongo.upsert_chunks(chunks)

        qdrant.close()
        mongo.close()

        logger.info(
            "Ingestion task %s done: %d chunks, %d mongo, %d qdrant",
            task_id, len(chunks), mongo_count, qdrant_count,
        )
    except Exception as exc:
        logger.error("Ingestion task %s failed: %s", task_id, exc, exc_info=True)


@app.delete("/api/v1/cache", response_model=CacheFlushResponse)
async def flush_cache(fastapi_request: Request):
    """Flush all cached RAG responses from Redis."""
    pipe = _get_pipeline(fastapi_request)
    if pipe._cache is None:
        return CacheFlushResponse(status="cache_disabled", keys_removed=0)
    count = pipe._cache.flush_all()
    if count < 0:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return CacheFlushResponse(status="ok", keys_removed=count)


@app.get("/health", response_model=HealthCheckResponse)
async def health(fastapi_request: Request):
    """Health check endpoint — probes MongoDB, Qdrant, and Redis."""
    services = HealthServiceStatus()

    # MongoDB check
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        client.close()
        services.mongodb = "ok"
    except Exception as exc:
        services.mongodb = f"unavailable ({exc})"

    # Qdrant check
    try:
        pipe = _get_pipeline(fastapi_request)
        pipe._qdrant_indexer.count_points()
        services.qdrant = "ok"
    except Exception as exc:
        services.qdrant = f"unavailable ({exc})"

    # Redis check
    try:
        import redis
        from config.settings import REDIS_URL
        r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
        r.ping()
        r.close()
        services.redis = "ok"
    except Exception:
        services.redis = "unavailable (not critical)"

    overall = "ok" if services.mongodb == "ok" else "degraded"
    warmup = getattr(fastapi_request.app.state, "warmup_completed", False)
    return HealthCheckResponse(status=overall, services=services, warmup_completed=warmup)
