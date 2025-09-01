import time
from functools import lru_cache

class TimedCache:
    def __init__(self, ttl=300):
        self.ttl = ttl
        self.store = {}

    def get(self, k):
        v = self.store.get(k)
        if not v: return None
        data, ts = v
        if time.time() - ts > self.ttl:
            self.store.pop(k, None)
            return None
        return data

    def set(self, k, v):
        self.store[k] = (v, time.time())

cache = TimedCache(ttl=300)
