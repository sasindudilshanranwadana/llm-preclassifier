from llm_preclassifier.cache import ClassificationCache, RedisClassificationCache


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


class FakeRedis:
    """Minimal stand-in for redis.Redis, enough to exercise get/set with TTL."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def get(self, name):
        return self._store.get(name)

    def set(self, name, value, ex=None):
        self._store[name] = value


def test_redis_cache_round_trips_through_serialize_and_deserialize():
    cache = RedisClassificationCache(
        FakeRedis(), ttl_seconds=30,
        serialize=lambda value: value.encode(), deserialize=lambda raw: raw.decode(),
    )

    cache.put("a", "hello")

    assert cache.get("a") == "hello"


def test_redis_cache_misses_are_none():
    cache = RedisClassificationCache(
        FakeRedis(), ttl_seconds=30,
        serialize=lambda value: value.encode(), deserialize=lambda raw: raw.decode(),
    )

    assert cache.get("missing") is None


def test_redis_cache_uses_a_key_prefix_to_avoid_collisions():
    redis = FakeRedis()
    cache = RedisClassificationCache(
        redis, ttl_seconds=30,
        serialize=lambda value: value.encode(), deserialize=lambda raw: raw.decode(),
        key_prefix="myapp:",
    )

    cache.put("a", "hello")

    assert redis.get("myapp:a") == b"hello"
