from llm_preclassifier.ratelimit import RateLimiter


def test_disabled_when_limit_is_zero():
    limiter = RateLimiter(0)

    assert all(limiter.allow("key") for _ in range(1000))


def test_allows_up_to_the_limit_within_a_window():
    limiter = RateLimiter(3, clock=lambda: 0.0)

    assert limiter.allow("key")
    assert limiter.allow("key")
    assert limiter.allow("key")
    assert not limiter.allow("key")


def test_keys_are_tracked_independently():
    limiter = RateLimiter(1, clock=lambda: 0.0)

    assert limiter.allow("a")
    assert limiter.allow("b")
    assert not limiter.allow("a")


def test_window_resets_after_a_minute():
    now = [0.0]
    limiter = RateLimiter(1, clock=lambda: now[0])

    assert limiter.allow("key")
    assert not limiter.allow("key")
    now[0] = 61.0
    assert limiter.allow("key")
