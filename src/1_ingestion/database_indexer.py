"""
Database Indexer — dual-storage pipeline for MongoDB (raw) + Qdrant (vectors).

Embedding Engine: nomic-ai/nomic-embed-text-v1.5 (768 dims, CUDA-accelerated).
Primary Storage: MongoDB — raw text, markdown, full metadata (chunk_id as PK).
Vector Storage: Qdrant — 768-dim HNSW vectors with lightweight metadata payload.
"""

import time
import uuid as _uuid
from typing import Any

import mlflow
import pymongo
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from config.logging_config import get_logger
from config.settings import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_RATIO,
    EMBEDDING_DIM,
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PATH,
    QDRANT_PORT,
)

logger = get_logger("ingestion.indexer")

# HNSW index configuration per spec
_HNSW_M = 16
_HNSW_EF_CONSTRUCT = 128

# Qdrant payload fields for metadata pre-filtering
_QDRANT_PAYLOAD_FIELDS = [
    "chunk_id",
    "ticker",
    "fiscal_year",
    "section",
    "doc_type",
    "contains_table",
]

# Maximum retry attempts for database connections
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2


# ============================================================================
# MLflow Ingestion Tracking
# ============================================================================

class IngestionTracker:
    """Logs ingestion hyper-parameters and run metrics to MLflow."""

    def __init__(self, experiment_name: str = "financial_rag_ingestion"):
        self._experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> Any:
        """Begin a tracked ingestion run."""
        return mlflow.start_run(run_name=run_name)

    def log_params(
        self,
        embedding_model: str,
        device: str,
        dim: int,
    ):
        """Log ingestion hyper-parameters to MLflow."""
        mlflow.log_param("chunk_min_tokens", CHUNK_MIN_TOKENS)
        mlflow.log_param("chunk_max_tokens", CHUNK_MAX_TOKENS)
        mlflow.log_param("chunk_overlap_ratio", CHUNK_OVERLAP_RATIO)
        mlflow.log_param("embedding_model", embedding_model)
        mlflow.log_param("embedding_dim", dim)
        mlflow.log_param("device", device)
        mlflow.log_param("hnsw_m", _HNSW_M)
        mlflow.log_param("hnsw_ef_construct", _HNSW_EF_CONSTRUCT)

    def log_metrics(
        self,
        total_chunks: int,
        text_chunks: int,
        table_chunks: int,
        elapsed_seconds: float,
    ):
        """Log ingestion run metrics to MLflow."""
        mlflow.log_metric("total_chunks_indexed", total_chunks)
        mlflow.log_metric("text_chunks", text_chunks)
        mlflow.log_metric("table_chunks", table_chunks)
        mlflow.log_metric("ingestion_elapsed_seconds", round(elapsed_seconds, 2))


class EmbeddingEngine:
    """
    Wraps sentence-transformers with nomic-ai/nomic-embed-text-v1.5.
    Forces CUDA device — asserts GPU availability at startup.
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first use. Asserts CUDA availability."""
        if self._model is None:
            assert torch.cuda.is_available(), (
                "CUDA is required for EmbeddingEngine but torch.cuda.is_available() is False. "
                "Install PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
            device = "cuda"
            logger.info(
                "Loading embedding model %s on device=%s (GPU: %s)",
                self._model_name,
                device,
                torch.cuda.get_device_name(0),
            )
            self._model = SentenceTransformer(self._model_name, device=device)
            # Support both old and new API names
            dim_method = getattr(self._model, "get_embedding_dimension", None)
            if dim_method is None:
                dim_method = getattr(self._model, "get_sentence_embedding_dimension")
            logger.info(
                "Model loaded -- dim=%d, device=%s, GPU=%s",
                dim_method(),
                device,
                torch.cuda.get_device_name(0),
            )
        return self._model

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed a list of texts into 768-dim vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding (default 32 for 6GB VRAM GPUs).

        Returns:
            List of 768-dim float vectors.
        """
        model = self._load_model()
        logger.info("Embedding %d texts (batch_size=%d)", len(texts), batch_size)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text])[0]


class MongoDBIndexer:
    """
    Stores raw chunk text and full metadata in MongoDB.
    Uses chunk_id as the primary key with upsert semantics.
    """

    def __init__(
        self,
        uri: str = MONGODB_URI,
        db_name: str = MONGODB_DB,
        collection_name: str = MONGODB_COLLECTION,
    ):
        self._uri = uri
        self._db_name = db_name
        self._collection_name = collection_name
        self._client: pymongo.MongoClient | None = None
        self._collection: pymongo.collection.Collection | None = None

    def _connect(self) -> pymongo.collection.Collection:
        """Establish MongoDB connection with retry logic."""
        if self._collection is not None:
            return self._collection

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                self._client = pymongo.MongoClient(
                    self._uri,
                    serverSelectionTimeoutMS=5000,
                )
                # Verify connection
                self._client.admin.command("ping")
                db = self._client[self._db_name]
                self._collection = db[self._collection_name]

                # Create unique index on chunk_id for upsert semantics
                self._collection.create_index("chunk_id", unique=True)

                logger.info(
                    "Connected to MongoDB: %s/%s.%s",
                    self._uri,
                    self._db_name,
                    self._collection_name,
                )
                return self._collection
            except Exception as exc:
                logger.warning(
                    "MongoDB connection attempt %d/%d failed: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY_SECONDS)
                else:
                    raise ConnectionError(
                        f"Failed to connect to MongoDB after {_MAX_RETRIES} attempts"
                    ) from exc

        raise ConnectionError("MongoDB connection failed unexpectedly")

    def upsert_chunks(self, chunks: list[dict]) -> int:
        """
        Upsert chunk records into MongoDB.

        Args:
            chunks: List of chunk dicts from hybrid_chunker.

        Returns:
            Number of documents successfully upserted.
        """
        collection = self._connect()
        success_count = 0

        for chunk in chunks:
            try:
                document = {
                    "raw_text": chunk["text"],
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "token_count": chunk.get("token_count", 0),
                    **chunk.get("metadata", {}),
                    "chunk_id": chunk["chunk_id"],
                }
                collection.update_one(
                    {"chunk_id": chunk["chunk_id"]},
                    {"$set": document},
                    upsert=True,
                )
                success_count += 1
            except Exception as exc:
                logger.error(
                    "Failed to upsert chunk %s: %s",
                    chunk.get("chunk_id", "UNKNOWN"),
                    exc,
                )

        logger.info(
            "MongoDB upsert: %d/%d chunks stored",
            success_count,
            len(chunks),
        )
        return success_count

    def get_chunk(self, chunk_id: str) -> dict | None:
        """Retrieve a chunk by its ID."""
        collection = self._connect()
        doc = collection.find_one({"chunk_id": chunk_id}, {"_id": 0})
        return doc

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict]:
        """
        Batch-retrieve chunks by their IDs.

        Args:
            chunk_ids: List of chunk_id strings to fetch.

        Returns:
            Dict mapping chunk_id -> document dict.
        """
        collection = self._connect()
        results = collection.find(
            {"chunk_id": {"$in": chunk_ids}}, {"_id": 0}
        )
        return {doc["chunk_id"]: doc for doc in results}

    def count_documents(self, ticker: str | None = None) -> int:
        """Count total documents, optionally filtered by ticker."""
        collection = self._connect()
        query = {"ticker": ticker} if ticker else {}
        return collection.count_documents(query)

    def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._collection = None

    def drop_collection(self):
        """Drop the entire collection (for test cleanup)."""
        collection = self._connect()
        collection.drop()


class QdrantIndexer:
    """
    Stores 768-dim vectors with HNSW indexing in Qdrant.
    Includes lightweight metadata payload for pre-filtering.
    """

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection_name: str = QDRANT_COLLECTION,
        embedding_dim: int = EMBEDDING_DIM,
        path: str | None = QDRANT_PATH,
    ):
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._path = path
        self._client: QdrantClient | None = None

    def _connect(self) -> QdrantClient:
        """Establish Qdrant connection with retry logic. Uses persistent path if provided."""
        if self._client is not None:
            return self._client

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                if self._path:
                    # Persistent local storage mode
                    import os
                    os.makedirs(self._path, exist_ok=True)
                    self._client = QdrantClient(path=self._path)
                    logger.info("Connected to Qdrant (persistent): %s", self._path)
                else:
                    # Remote server mode
                    self._client = QdrantClient(host=self._host, port=self._port)
                    self._client.get_collections()
                    logger.info("Connected to Qdrant: %s:%d", self._host, self._port)
                return self._client
            except Exception as exc:
                logger.warning(
                    "Qdrant connection attempt %d/%d failed: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY_SECONDS)
                else:
                    raise ConnectionError(
                        f"Failed to connect to Qdrant after {_MAX_RETRIES} attempts"
                    ) from exc

        raise ConnectionError("Qdrant connection failed unexpectedly")

    def ensure_collection(self):
        """Create the collection with HNSW config if it doesn't exist."""
        client = self._connect()
        collections = client.get_collections().collections
        existing_names = [c.name for c in collections]

        if self._collection_name not in existing_names:
            client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection: %s (dim=%d, cosine)",
                self._collection_name,
                self._embedding_dim,
            )
        else:
            logger.debug("Qdrant collection already exists: %s", self._collection_name)

    def upsert_vectors(
        self, chunks: list[dict], embeddings: list[list[float]]
    ) -> int:
        """
        Upsert vectors with metadata payload into Qdrant.

        Args:
            chunks: List of chunk dicts (must align with embeddings).
            embeddings: List of 768-dim float vectors.

        Returns:
            Number of points successfully upserted.
        """
        client = self._connect()
        self.ensure_collection()

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = chunk.get("metadata", {})
            # Build lightweight payload for pre-filtering
            payload = {
                "chunk_id": chunk["chunk_id"],
                "ticker": metadata.get("ticker", "UNKNOWN"),
                "fiscal_year": metadata.get("fiscal_year", "UNKNOWN"),
                "section": metadata.get("section", "General"),
                "doc_type": metadata.get("doc_type", "UNKNOWN"),
                "contains_table": metadata.get("contains_table", False),
            }

            # Convert string chunk_id to deterministic UUID for Qdrant
            point_id = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, chunk["chunk_id"]))

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert in batches of 100 for efficiency
        batch_size = 100
        success_count = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            try:
                client.upsert(
                    collection_name=self._collection_name,
                    points=batch,
                )
                success_count += len(batch)
            except Exception as exc:
                logger.error(
                    "Failed to upsert Qdrant batch at offset %d: %s", i, exc
                )

        logger.info(
            "Qdrant upsert: %d/%d vectors stored",
            success_count,
            len(points),
        )
        return success_count

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        ticker: str | None = None,
        fiscal_year: str | None = None,
    ) -> list[dict]:
        """
        Perform vector similarity search with optional metadata pre-filtering.

        Args:
            query_embedding: 768-dim query vector.
            top_k: Number of results to return.
            ticker: Filter by ticker symbol.
            fiscal_year: Filter by fiscal year.

        Returns:
            List of search results with scores and payloads.
        """
        client = self._connect()

        # Build metadata filter conditions
        must_conditions = []
        if ticker:
            must_conditions.append(
                FieldCondition(key="ticker", match=MatchValue(value=ticker))
            )
        if fiscal_year:
            must_conditions.append(
                FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        response = client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            {
                "chunk_id": hit.payload.get("chunk_id", str(hit.id)),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in response.points
        ]

    def count_points(self, ticker: str | None = None) -> int:
        """Count total points, optionally filtered by ticker."""
        client = self._connect()
        if ticker:
            result = client.count(
                collection_name=self._collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))]
                ),
            )
        else:
            result = client.count(collection_name=self._collection_name)
        return result.count

    def close(self):
        """Close Qdrant connection."""
        if self._client:
            self._client.close()
            self._client = None


class DualStorageIndexer:
    """
    Orchestrates the dual-storage pipeline:
    1. Embed chunks using nomic-embed-text-v1.5 (CUDA)
    2. Store raw text + metadata in MongoDB
    3. Store vectors + lightweight payload in Qdrant
    4. Log ingestion run to MLflow
    """

    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.mongo_indexer = MongoDBIndexer()
        self.qdrant_indexer = QdrantIndexer()
        self._tracker = IngestionTracker()

    def index_chunks(self, chunks: list[dict]) -> dict[str, int]:
        """
        Execute the full dual-storage indexing pipeline.

        Args:
            chunks: List of chunk dicts from hybrid_chunker.

        Returns:
            Summary dict with 'mongo_count' and 'qdrant_count'.
        """
        if not chunks:
            logger.warning("No chunks to index")
            return {"mongo_count": 0, "qdrant_count": 0}

        logger.info("Starting dual-storage indexing for %d chunks", len(chunks))
        t0 = time.time()

        # Step 1: Embed all chunk texts
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_engine.embed(texts)

        # Step 2: Store in MongoDB (raw text + full metadata)
        mongo_count = self.mongo_indexer.upsert_chunks(chunks)

        # Step 3: Store in Qdrant (vectors + lightweight payload)
        qdrant_count = self.qdrant_indexer.upsert_vectors(chunks, embeddings)

        elapsed = time.time() - t0

        # Step 4: Log to MLflow
        text_chunks = sum(1 for c in chunks if c.get("chunk_type") == "text")
        table_chunks = len(chunks) - text_chunks
        try:
            self._tracker.log_params(
                embedding_model=self.embedding_engine._model_name,
                device="cuda",
                dim=EMBEDDING_DIM,
            )
            self._tracker.log_metrics(
                total_chunks=len(chunks),
                text_chunks=text_chunks,
                table_chunks=table_chunks,
                elapsed_seconds=elapsed,
            )
            logger.info("MLflow ingestion run logged successfully")
        except Exception as exc:
            logger.warning("MLflow logging failed (non-blocking): %s", exc)

        result = {
            "mongo_count": mongo_count,
            "qdrant_count": qdrant_count,
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info("Dual-storage indexing complete: %s", result)
        return result

    def search_similar(
        self,
        query_text: str,
        top_k: int = 3,
        ticker: str | None = None,
        fiscal_year: str | None = None,
    ) -> list[dict]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_text: User query string.
            top_k: Number of results.
            ticker: Optional ticker filter.
            fiscal_year: Optional fiscal year filter.

        Returns:
            List of search results with scores and metadata.
        """
        query_embedding = self.embedding_engine.embed_single(query_text)
        return self.qdrant_indexer.search(
            query_embedding, top_k, ticker, fiscal_year
        )

    def get_stats(self) -> dict[str, Any]:
        """Get current storage statistics."""
        return {
            "mongo_total": self.mongo_indexer.count_documents(),
            "qdrant_total": self.qdrant_indexer.count_points(),
            "mongo_by_ticker": {
                t: self.mongo_indexer.count_documents(t)
                for t in ["AAPL", "MSFT", "NVDA"]
            },
        }

    def close(self):
        """Close all database connections."""
        self.mongo_indexer.close()
        self.qdrant_indexer.close()
