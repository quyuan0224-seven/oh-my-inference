"""KV cache 生命周期管理。

mini_vllm 的简化：
- 一个请求一段连续的 K/V 列表；
- 只有一层、一个 attention head；
- 没有 paged block、block table、prefix sharing。

职责边界：
- KVCacheManager 在 engine 侧，负责为请求分配/查找/释放 cache；
- KVCache 在 model 侧，负责实际存储；
- ModelRunner/Model 负责往里写 K/V、从里面读 K/V。
"""

from __future__ import annotations

from ..model.cache import KVCache
from .types import Request


class KVCacheManager:
    """管理每个请求的 KV cache 生命周期。"""

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self._caches: dict[str, KVCache] = {}

    def allocate(self, request: Request) -> KVCache:
        if request.request_id in self._caches:
            raise ValueError(f"cache already allocated: {request.request_id}")
        cache = KVCache(hidden_size=self.hidden_size)
        self._caches[request.request_id] = cache
        return cache

    def get(self, request_id: str) -> KVCache:
        try:
            return self._caches[request_id]
        except KeyError as exc:
            raise KeyError(f"no KV cache for request: {request_id}") from exc

    def free(self, request_id: str) -> None:
        self._caches.pop(request_id, None)

    @property
    def active_requests(self) -> int:
        return len(self._caches)

    @property
    def cached_tokens(self) -> int:
        return sum(cache.num_tokens for cache in self._caches.values())
