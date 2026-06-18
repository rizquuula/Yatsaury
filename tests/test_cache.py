"""Tests for DiskCache — content-hash based disk cache."""
from __future__ import annotations


def test_cache_miss_returns_none(tmp_path):
    from yatsaury.cache import DiskCache
    cache = DiskCache(tmp_path / "cache")
    assert cache.get("file.txt", "fp1") is None


def test_cache_set_and_get(tmp_path):
    from yatsaury.cache import DiskCache
    cache = DiskCache(tmp_path / "cache")
    cache.set("file.txt", "fp1", [{"q": "Q", "a": "A"}])
    result = cache.get("file.txt", "fp1")
    assert result == [{"q": "Q", "a": "A"}]


def test_cache_different_fingerprint_is_miss(tmp_path):
    from yatsaury.cache import DiskCache
    cache = DiskCache(tmp_path / "cache")
    cache.set("file.txt", "fp1", [{"q": "Q"}])
    assert cache.get("file.txt", "fp2") is None


def test_cache_invalidate(tmp_path):
    from yatsaury.cache import DiskCache
    cache = DiskCache(tmp_path / "cache")
    cache.set("file.txt", "fp1", [{"q": "Q"}])
    assert cache.invalidate("file.txt", "fp1") is True
    assert cache.get("file.txt", "fp1") is None


def test_cache_invalidate_nonexistent(tmp_path):
    from yatsaury.cache import DiskCache
    cache = DiskCache(tmp_path / "cache")
    assert cache.invalidate("file.txt", "fp1") is False


def test_orchestrator_second_run_is_cache_hit(tmp_path):
    """Second Orchestrator.run() on same URI + config uses cache, skips LLM generate."""
    from unittest.mock import MagicMock, patch

    from yatsaury.pipeline import Orchestrator, OrchestratorConfig

    config = OrchestratorConfig(
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        llm_api_key="test",
    )

    mock_gen = MagicMock()
    mock_gen.generate.return_value = []

    with patch("yatsaury.pipeline.get_generator", return_value=mock_gen):
        orch = Orchestrator(config)
        orch.run(["hello world"])  # first run
        call_count_after_first = mock_gen.generate.call_count

        orch2 = Orchestrator(config)
        orch2.run(["hello world"])  # second run — should hit cache
        call_count_after_second = mock_gen.generate.call_count

    # Generator called on first run, not on second
    assert call_count_after_first > 0
    assert call_count_after_second == call_count_after_first  # no new calls
