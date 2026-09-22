from llm_preclassifier.cache import ClassificationCache


def test_cache_evicts_least_recently_used_entry():
    cache = ClassificationCache(max_entries=2, ttl_seconds=30)
    cache.put("a", {"id": "a"})
    cache.put("b", {"id": "b"})
    assert cache.get("a") == {"id": "a"}
    cache.put("c", {"id": "c"})

    assert cache.get("b") is None
    assert cache.get("a") == {"id": "a"}
    assert cache.get("c") == {"id": "c"}


def test_cache_expires_entries_with_injected_clock():
    now = [0.0]
    cache = ClassificationCache(max_entries=2, ttl_seconds=5, clock=lambda: now[0])
    cache.put("a", {"id": "a"})
    now[0] = 5.0

    assert cache.get("a") is None
