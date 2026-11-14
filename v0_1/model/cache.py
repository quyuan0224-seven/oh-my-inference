"""一个请求的连续 KV cache 存储。

V0.1 只有一层、一个 head，所以直接用一个 K/V 列表表示。
它属于 model 侧的数据结构；Engine 侧的 KVCacheManager 负责生命周期，
这样 model 不需要反向依赖 engine。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KVCache:
    hidden_size: int
    keys: list[list[float]] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return len(self.keys)

    def append(self, key: list[float], value: list[float]) -> None:
        if len(key) != self.hidden_size or len(value) != self.hidden_size:
            raise ValueError("key/value size does not match hidden_size")
        self.keys.append(key)
        self.values.append(value)
