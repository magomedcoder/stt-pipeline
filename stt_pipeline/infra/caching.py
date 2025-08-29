from functools import lru_cache

@lru_cache(maxsize=8)
def load_cached(key: tuple, factory):
    return factory()