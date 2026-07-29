"""Verify FastAPI lifespan warm-up routine."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymongo
import pytest
import qdrant_client
import redis
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_ingestion"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "2_generation"))

from app.api.main import app, _run_warmup


@pytest.fixture(scope="module")
def fresh_app_instance():
    """Yield a mocked TestClient once per module.

    The lifespan runs once when TestClient enters, invoking warm-up on the
    mocked FinancialRAGPipeline instance.
    """
    pipe = MagicMock()
    pipe._embedding_engine = MagicMock()
    pipe._qdrant_indexer = MagicMock()
    pipe._mongo_indexer = MagicMock()
    pipe._cache = MagicMock()
    pipe._cache._get_client = MagicMock()
    pipe.close = MagicMock()

    with (
        patch("app.api.main.FinancialRAGPipeline", return_value=pipe) as mock_cls,
        patch.object(pymongo, "MongoClient") as mock_mongo,
        patch.object(qdrant_client, "QdrantClient") as mock_qdrant,
        patch.object(redis, "from_url") as mock_redis,
    ):
        mock_mongo.return_value.admin.command.return_value = {"ok": 1}
        mock_qdrant.return_value.get_collections.return_value = MagicMock()
        mock_redis.return_value.ping.return_value = True

        with TestClient(app) as client:
            yield client, pipe


class TestWarmUp:

    def test_warmup_runs_embedding(self, fresh_app_instance):
        _, pipe = fresh_app_instance
        pipe._embedding_engine.embed_single.assert_called_with("warmup")

    def test_warmup_runs_qdrant(self, fresh_app_instance):
        _, pipe = fresh_app_instance
        pipe._qdrant_indexer.count_points.assert_called_once()

    def test_warmup_runs_mongo(self, fresh_app_instance):
        _, pipe = fresh_app_instance
        pipe._mongo_indexer.count_documents.assert_called_once()

    def test_warmup_runs_redis(self, fresh_app_instance):
        _, pipe = fresh_app_instance
        pipe._cache._get_client.assert_called_once()

    def test_health_includes_warmup_field(self, fresh_app_instance):
        client, _ = fresh_app_instance
        resp = client.get("/health")
        body = resp.json()
        assert "warmup_completed" in body
        assert body["warmup_completed"] is True

    def test_main_status_badge_shows_ok(self, fresh_app_instance):
        client, _ = fresh_app_instance
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "ok"


class TestWarmUpErrorHandling:

    def test_warmup_error_propagates(self):
        """_run_warmup raises — the lifespan's try/except is responsible for catching."""
        pipe = MagicMock()
        pipe._embedding_engine = MagicMock()
        pipe._embedding_engine.embed_single.side_effect = RuntimeError("GPU busy")
        pipe._qdrant_indexer = MagicMock()
        pipe._mongo_indexer = MagicMock()
        pipe._cache = MagicMock()
        pipe._cache._get_client = MagicMock()

        import asyncio
        with pytest.raises(RuntimeError, match="GPU busy"):
            asyncio.run(_run_warmup(pipe))
