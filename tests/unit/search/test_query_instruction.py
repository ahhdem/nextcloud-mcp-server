"""Unit tests for EMBEDDING_QUERY_INSTRUCTION (query-side instruction prefix)."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nextcloud_mcp_server.config import Settings, _reload_config, get_settings
from nextcloud_mcp_server.search.bm25_hybrid import BM25HybridSearchAlgorithm
from nextcloud_mcp_server.search.query_instruction import (
    format_query_for_embedding,
    query_embedding_text,
)
from nextcloud_mcp_server.search.semantic import SemanticSearchAlgorithm

INSTRUCTION = "Given a search query, retrieve relevant passages that answer it"


@pytest.mark.unit
class TestFormat:
    def test_unset_returns_query_unchanged(self):
        assert format_query_for_embedding("hello", None) == "hello"
        assert format_query_for_embedding("hello", "") == "hello"
        assert format_query_for_embedding("hello", "   ") == "hello"

    def test_set_uses_qwen3_embedding_format(self):
        assert (
            format_query_for_embedding("hello", INSTRUCTION)
            == f"Instruct: {INSTRUCTION}\nQuery: hello"
        )

    def test_instruction_is_stripped(self):
        assert (
            format_query_for_embedding("hello", f"  {INSTRUCTION}\n")
            == f"Instruct: {INSTRUCTION}\nQuery: hello"
        )

    def test_non_string_instruction_is_ignored(self):
        """An unconfigured MagicMock settings object must not leak into the text."""
        assert query_embedding_text("hello", MagicMock()) == "hello"


@pytest.mark.unit
class TestConfig:
    def test_default_is_empty(self):
        assert Settings().embedding_query_instruction == ""

    @patch.dict(os.environ, {"EMBEDDING_QUERY_INSTRUCTION": INSTRUCTION}, clear=True)
    def test_read_from_env(self):
        _reload_config()
        try:
            assert get_settings().embedding_query_instruction == INSTRUCTION
        finally:
            with patch.dict(os.environ, {}, clear=True):
                _reload_config()


def _settings(instruction: str) -> MagicMock:
    settings = MagicMock()
    settings.get_collection_name.return_value = "test_collection"
    settings.get_embedding_provider_family.return_value = "openai"
    settings.vector_search_rrf_k = 60
    settings.embedding_query_instruction = instruction
    return settings


def _patch_hybrid(monkeypatch, settings):
    embed = AsyncMock(return_value=([0.1, 0.2, 0.3], 7))
    svc = MagicMock()
    svc.embed_with_usage = embed
    monkeypatch.setattr(
        "nextcloud_mcp_server.search.bm25_hybrid.get_provider", lambda: svc
    )
    bm25 = MagicMock()
    bm25.encode_async = AsyncMock(return_value={"indices": [1], "values": [0.5]})
    monkeypatch.setattr(
        "nextcloud_mcp_server.search.bm25_hybrid.get_bm25_service",
        AsyncMock(return_value=bm25),
    )
    qdrant = MagicMock()
    empty = MagicMock()
    empty.points = []
    qdrant.query_points = AsyncMock(return_value=empty)
    monkeypatch.setattr(
        "nextcloud_mcp_server.search.bm25_hybrid.get_qdrant_client",
        AsyncMock(return_value=qdrant),
    )
    monkeypatch.setattr(
        "nextcloud_mcp_server.search.bm25_hybrid.get_settings", lambda: settings
    )
    monkeypatch.setattr(
        "nextcloud_mcp_server.search.bm25_hybrid.build_base_filter_conditions",
        lambda **kwargs: [],
    )
    return embed, bm25


@pytest.mark.unit
class TestHybridQueryPath:
    async def test_dense_query_is_prefixed_sparse_is_raw(self, monkeypatch):
        embed, bm25 = _patch_hybrid(monkeypatch, _settings(INSTRUCTION))

        await BM25HybridSearchAlgorithm().search(query="hello", user_id="alice")

        embed.assert_awaited_once_with(f"Instruct: {INSTRUCTION}\nQuery: hello")
        # BM25 keyword matching must see the raw query, not the instruction.
        bm25.encode_async.assert_awaited_once_with("hello")

    async def test_unset_embeds_raw_query(self, monkeypatch):
        embed, bm25 = _patch_hybrid(monkeypatch, _settings(""))

        await BM25HybridSearchAlgorithm().search(query="hello", user_id="alice")

        embed.assert_awaited_once_with("hello")
        bm25.encode_async.assert_awaited_once_with("hello")

    async def test_cache_still_hits_with_instruction(self, monkeypatch):
        embed, _ = _patch_hybrid(monkeypatch, _settings(INSTRUCTION))
        algo = BM25HybridSearchAlgorithm()

        for dtype in ("note", "file"):
            await algo.search(query="hello", user_id="alice", doc_type=dtype)

        assert embed.await_count == 1


@pytest.mark.unit
class TestDenseQueryPath:
    async def test_semantic_query_is_prefixed(self, monkeypatch):
        embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        svc = MagicMock()
        svc.embed = embed
        monkeypatch.setattr(
            "nextcloud_mcp_server.search.semantic.get_provider", lambda: svc
        )
        monkeypatch.setattr(
            "nextcloud_mcp_server.search.semantic.get_settings",
            lambda: _settings(INSTRUCTION),
        )
        # Stop right after the embedding: the Qdrant leg is out of scope here.
        monkeypatch.setattr(
            "nextcloud_mcp_server.search.semantic.build_base_filter_conditions",
            MagicMock(side_effect=RuntimeError("stop")),
        )

        with pytest.raises(RuntimeError, match="stop"):
            await SemanticSearchAlgorithm().search(query="hello", user_id="alice")

        embed.assert_awaited_once_with(f"Instruct: {INSTRUCTION}\nQuery: hello")


@pytest.mark.unit
def test_indexing_path_never_applies_the_instruction():
    """Documents must be embedded without the instruction, or turning the
    setting on would silently mismatch every stored vector. Guard that nothing
    outside the query-side modules reaches for it."""
    pkg = Path(__file__).resolve().parents[3] / "nextcloud_mcp_server"
    allowed = {
        pkg / "config.py",
        pkg / "search" / "query_instruction.py",
        pkg / "search" / "bm25_hybrid.py",
        pkg / "search" / "semantic.py",
        pkg / "api" / "visualization.py",
    }
    offenders = [
        str(p.relative_to(pkg))
        for p in pkg.rglob("*.py")
        if p not in allowed
        and (
            "query_embedding_text" in (text := p.read_text())
            or "format_query_for_embedding" in text
            or "embedding_query_instruction" in text
        )
    ]
    assert offenders == []
