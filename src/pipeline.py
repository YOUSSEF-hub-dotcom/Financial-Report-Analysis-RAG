"""
Financial RAG Pipeline Orchestrator.

Master class that wires together Module 1 (Ingestion/Retrieval) and
Module 2 (Generation/Guardrails) into a single queryable interface.

Flow:
    1. Check Redis semantic cache for identical query
    2. Vector search Qdrant with optional metadata pre-filtering
    3. Fetch full text from MongoDB (Qdrant payload has no raw_text)
    4. Format retrieved docs into XML <CONTEXT> tags
    5. Generate answer via Groq LLM (primary → fallback)
    6. Background async guardrail verification + cache write
    7. Log end-to-end metrics to MLflow
"""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import mlflow

# Ensure src subdirectories (numbered names, not valid Python packages) are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_src_root = Path(__file__).resolve().parent
for _subdir in ("1_ingestion", "2_generation"):
    _p = str(_src_root / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.logging_config import get_logger
from config.settings import (
    GROQ_FALLBACK_MODEL,
    GROQ_PRIMARY_MODEL,
    LLM_MAX_TOKENS,
    LLM_SEED,
    LLM_TEMPERATURE,
    MONGODB_COLLECTION,
    MONGODB_DB,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    SUPPORTED_TICKERS,
)

from database_indexer import EmbeddingEngine, MongoDBIndexer, QdrantIndexer
from async_guardrail import AsyncGuardrail, SemanticCache
from generator import FinancialRAGGenerator, format_context_xml

logger = get_logger("pipeline.orchestrator")


class FinancialRAGPipeline:
    """
    Master orchestrator that connects vector retrieval, MongoDB enrichment,
    LLM generation, guardrail verification, semantic caching, and MLflow logging
    into a single unified query interface.
    """

    def __init__(
        self,
        qdrant_path: str = QDRANT_PATH,
        mongo_db: str = MONGODB_DB,
        mongo_collection: str = MONGODB_COLLECTION,
        qdrant_collection: str = QDRANT_COLLECTION,
        primary_model: str = GROQ_PRIMARY_MODEL,
        fallback_model: str = GROQ_FALLBACK_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        seed: int = LLM_SEED,
        top_k: int = 3,
        enable_cache: bool = True,
        enable_guardrail: bool = True,
    ):
        """
        Initialize all sub-components of the RAG pipeline.

        Args:
            qdrant_path: Persistent path for Qdrant vector storage.
            mongo_db: MongoDB database name.
            mongo_collection: MongoDB collection name.
            qdrant_collection: Qdrant collection name.
            primary_model: Groq primary LLM model name.
            fallback_model: Groq fallback LLM model name.
            temperature: LLM temperature for generation.
            max_tokens: Maximum tokens in LLM response.
            seed: Random seed for deterministic generation.
            top_k: Default number of documents to retrieve.
            enable_cache: Whether to use Redis semantic cache.
            enable_guardrail: Whether to run async guardrail checks.
        """
        # Retrieval components
        self._embedding_engine = EmbeddingEngine()
        self._qdrant_indexer = QdrantIndexer(
            path=qdrant_path,
            collection_name=qdrant_collection,
        )
        self._mongo_indexer = MongoDBIndexer(
            db_name=mongo_db,
            collection_name=mongo_collection,
        )

        # Generation components
        self._generator = FinancialRAGGenerator(
            primary_model=primary_model,
            fallback_model=fallback_model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

        # Guardrail and caching
        self._enable_guardrail = enable_guardrail
        self._enable_cache = enable_cache
        self._guardrail = AsyncGuardrail() if enable_guardrail else None
        self._cache = SemanticCache() if enable_cache else None

        # Config
        self._top_k = top_k

        logger.info(
            "FinancialRAGPipeline initialized: primary=%s fallback=%s "
            "top_k=%d cache=%s guardrail=%s",
            primary_model,
            fallback_model,
            top_k,
            enable_cache,
            enable_guardrail,
        )

    def _retrieve_documents(
        self,
        query: str,
        ticker: str | None = None,
        fiscal_year: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Two-phase retrieval: Qdrant vector search → MongoDB text enrichment.

        Phase 1: Embed query, search Qdrant with optional metadata pre-filtering.
        Phase 2: Fetch full raw_text from MongoDB for each retrieved chunk_id.

        Args:
            query: User query string.
            ticker: Optional ticker filter (e.g. 'AAPL').
            fiscal_year: Optional fiscal year filter (e.g. '2025').
            top_k: Number of results to retrieve (default: self._top_k).

        Returns:
            List of enriched document dicts ready for LLM context formatting.
        """
        k = top_k or self._top_k

        # Phase 1: Qdrant vector search
        query_embedding = self._embedding_engine.embed_single(query)
        qdrant_results = self._qdrant_indexer.search(
            query_embedding, top_k=k, ticker=ticker, fiscal_year=fiscal_year
        )

        if not qdrant_results:
            logger.warning("Vector search returned 0 results for query: %s", query[:80])
            return []

        # Phase 2: MongoDB text enrichment
        chunk_ids = [r["chunk_id"] for r in qdrant_results]
        mongo_docs = self._mongo_indexer.get_chunks_by_ids(chunk_ids)

        documents = []
        for r in qdrant_results:
            cid = r["chunk_id"]
            payload = r.get("payload", {})
            mongo_doc = mongo_docs.get(cid, {})
            raw_text = mongo_doc.get("raw_text", "")

            if not raw_text:
                logger.debug("Chunk %s missing raw_text in MongoDB, skipping", cid)
                continue

            documents.append({
                "text": raw_text,
                "metadata": {
                    "ticker": payload.get("ticker", "UNKNOWN"),
                    "fiscal_year": payload.get("fiscal_year", "UNKNOWN"),
                    "section": payload.get("section", "General"),
                    "contains_table": payload.get("contains_table", False),
                    "page_number": payload.get("page_number", "N/A"),
                },
                "chunk_id": cid,
                "score": r["score"],
            })

        logger.info(
            "Retrieved %d/%d documents (Qdrant: %d, MongoDB enriched: %d)",
            len(documents),
            k,
            len(qdrant_results),
            len(mongo_docs),
        )
        return documents

    def query(
        self,
        user_query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Execute a full RAG query end-to-end.

        Steps:
            1. Cache check — return instantly on Redis hit
            2. Vector retrieval from Qdrant + MongoDB enrichment
            3. XML context formatting
            4. LLM generation via primary/fallback Groq models
            5. Background async guardrail verification + cache write
            6. MLflow logging of end-to-end metrics

        Args:
            user_query: Natural language financial question.
            ticker: Optional ticker filter (e.g. 'AAPL').
            fiscal_year: Optional fiscal year filter (e.g. '2025').
            session_id: Optional session identifier for conversation memory.
            top_k: Number of documents to retrieve.

        Returns:
            ConsolidatedFinancialAnswer dict with validated JSON response.
        """
        pipeline_run_id = str(uuid.uuid4())[:8]
        t_start = time.time()

        logger.info(
            "Pipeline query started: id=%s ticker=%s year=%s",
            pipeline_run_id,
            ticker,
            fiscal_year,
        )

        # Start MLflow run
        mlflow_active = False
        try:
            mlflow.set_experiment("financial_rag_pipeline")
            mlflow.start_run(run_name=f"pipeline_{pipeline_run_id}", nested=True)
            mlflow.log_param("query", user_query[:200])
            mlflow.log_param("ticker", ticker or "ANY")
            mlflow.log_param("fiscal_year", fiscal_year or "ANY")
            mlflow.log_param("top_k", top_k)
            mlflow_active = True
        except Exception as exc:
            logger.warning("MLflow pipeline run start failed (non-blocking): %s", exc)

        result: dict[str, Any] | None = None
        started_guardrail = False
        try:
            # Step 1: Cache check
            if self._enable_cache and self._cache is not None:
                cached = self._cache.get(user_query)
                if cached is not None:
                    logger.info("Cache HIT for pipeline query (id=%s)", pipeline_run_id)
                    if mlflow_active:
                        try:
                            mlflow.log_metric("cache_hit", 1)
                            mlflow.log_metric("e2e_latency_ms", round((time.time() - t_start) * 1000, 2))
                        except Exception:
                            pass
                    result = {
                        "raw_output": cached.get("answer_json", ""),
                        "parsed": None,
                        "model_used": "cache",
                        "fallback_triggered": False,
                        "ttft_ms": 0,
                        "cache_hit": True,
                        "pipeline_run_id": pipeline_run_id,
                    }
                    return result

            # Step 2: Retrieve documents
            t_retrieve = time.time()
            documents = self._retrieve_documents(
                user_query, ticker=ticker, fiscal_year=fiscal_year, top_k=top_k
            )
            retrieval_ms = (time.time() - t_retrieve) * 1000

            if not documents:
                logger.warning("No documents retrieved for query (id=%s)", pipeline_run_id)
                if mlflow_active:
                    try:
                        mlflow.log_metric("documents_retrieved", 0)
                        mlflow.log_metric("retrieval_ms", round(retrieval_ms, 2))
                        mlflow.log_metric("cache_hit", 0)
                        mlflow.log_metric("e2e_latency_ms", round((time.time() - t_start) * 1000, 2))
                    except Exception:
                        pass
                result = {
                    "raw_output": "The requested financial information is not available in the provided reports.",
                    "parsed": None,
                    "model_used": "none",
                    "fallback_triggered": False,
                    "ttft_ms": 0,
                    "cache_hit": False,
                    "pipeline_run_id": pipeline_run_id,
                }
                return result

            # Step 3: Format XML context
            context_xml = format_context_xml(documents)

            # Step 4: Generate answer
            gen_result = self._generator.generate(
                query=user_query,
                retrieved_docs=documents,
                stream=False,
            )

            total_ms = (time.time() - t_start) * 1000

            # Step 5: Background guardrail + cache write (skip cache for fallback/invalid answers)
            parsed_answer = gen_result.get("parsed")
            is_valid_answer = (
                parsed_answer is not None
                and not gen_result.get("fallback_triggered", False)
                and "not available" not in parsed_answer.answer.lower()
            )
            skip_cache = not is_valid_answer

            if parsed_answer is not None and self._enable_guardrail and self._guardrail is not None:
                started_guardrail = True
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(
                            self._guardrail.check(
                                query=user_query,
                                answer=parsed_answer.answer,
                                extracted_raw_data=parsed_answer.extracted_raw_data,
                                parsed_output=parsed_answer,
                                skip_cache=skip_cache,
                            )
                        )
                    else:
                        loop.run_until_complete(
                            self._guardrail.check(
                                query=user_query,
                                answer=parsed_answer.answer,
                                extracted_raw_data=parsed_answer.extracted_raw_data,
                                parsed_output=parsed_answer,
                                skip_cache=skip_cache,
                            )
                        )
                except RuntimeError:
                    try:
                        asyncio.run(
                            self._guardrail.check(
                                query=user_query,
                                answer=parsed_answer.answer,
                                extracted_raw_data=parsed_answer.extracted_raw_data,
                                parsed_output=parsed_answer,
                                skip_cache=skip_cache,
                            )
                        )
                    except Exception as exc:
                        logger.warning("Guardrail check failed (non-blocking): %s", exc)
                except Exception as exc:
                    logger.warning("Guardrail check failed (non-blocking): %s", exc)

            # Step 6: MLflow logging
            if mlflow_active:
                try:
                    mlflow.log_metric("documents_retrieved", len(documents))
                    mlflow.log_metric("retrieval_ms", round(retrieval_ms, 2))
                    mlflow.log_metric("generation_ttft_ms", gen_result.get("ttft_ms", 0))
                    mlflow.log_metric("cache_hit", 0)
                    mlflow.log_metric("fallback_triggered", int(gen_result.get("fallback_triggered", False)))
                    mlflow.log_metric("e2e_latency_ms", round(total_ms, 2))
                    mlflow.log_param("model_used", gen_result.get("model_used", "unknown"))
                except Exception as exc:
                    logger.warning("MLflow pipeline metrics failed (non-blocking): %s", exc)

            result = {
                **gen_result,
                "cache_hit": False,
                "pipeline_run_id": pipeline_run_id,
            }
            return result

        finally:
            if mlflow_active:
                try:
                    mlflow.end_run()
                except Exception:
                    pass

    async def query_stream(
        self,
        user_query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 3,
    ) -> AsyncIterator[str]:
        """
        Streaming RAG query — yields LLM tokens as they arrive.

        Executes retrieval synchronously, then streams tokens from the LLM.
        Guardrail verification runs in the background after streaming completes.

        Args:
            user_query: Natural language financial question.
            ticker: Optional ticker filter.
            fiscal_year: Optional fiscal year filter.
            session_id: Optional session identifier.
            top_k: Number of documents to retrieve.

        Yields:
            Individual token strings from the LLM response.
        """
        # Step 1: Retrieve documents
        documents = self._retrieve_documents(
            user_query, ticker=ticker, fiscal_year=fiscal_year, top_k=top_k
        )

        if not documents:
            yield "The requested financial information is not available in the provided reports."
            return

        # Step 2: Stream tokens from generator
        full_response_parts: list[str] = []
        async for token in self._generator.stream_tokens(
            query=user_query,
            retrieved_docs=documents,
        ):
            full_response_parts.append(token)
            yield token

        # Step 3: Background guardrail after streaming completes
        full_response = "".join(full_response_parts)
        if self._enable_guardrail and self._guardrail is not None:
            try:
                parsed = self._generator._parse_json_output(full_response)
                if parsed is not None:
                    skip_cache = "not available" in parsed.answer.lower()
                    asyncio.ensure_future(
                        self._guardrail.check(
                            query=user_query,
                            answer=parsed.answer,
                            extracted_raw_data=parsed.extracted_raw_data,
                            parsed_output=parsed,
                            skip_cache=skip_cache,
                        )
                    )
            except Exception as exc:
                logger.warning("Stream guardrail check failed (non-blocking): %s", exc)

        logger.info("Stream complete: %d tokens", len(full_response_parts))

    def clear_cache(self) -> None:
        """Clear the semantic cache."""
        if self._cache is not None:
            self._cache.close()
            logger.info("Cache cleared")

    def reset_memory(self) -> None:
        """Reset the generator's conversation memory."""
        self._generator.reset_memory()

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics (Qdrant/MongoDB counts)."""
        try:
            return {
                "qdrant_total": self._qdrant_indexer.count_points(),
                "mongo_total": self._mongo_indexer.count_documents(),
            }
        except Exception as exc:
            logger.warning("Failed to get stats: %s", exc)
            return {"error": str(exc)}

    def close(self) -> None:
        """Close all database connections and release resources."""
        try:
            self._qdrant_indexer.close()
        except Exception:
            pass
        try:
            self._mongo_indexer.close()
        except Exception:
            pass
        if self._guardrail is not None:
            try:
                self._guardrail.close()
            except Exception:
                pass
        if self._cache is not None:
            try:
                self._cache.close()
            except Exception:
                pass
        logger.info("Pipeline closed")
