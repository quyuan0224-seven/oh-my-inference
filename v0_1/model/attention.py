"""V0.1 的 toy model：单层、单 head 的 attention 语言模型。

它不是完整 Transformer（没有多层、没有 MLP、没有 LayerNorm），
但保留了最关键的 inference 结构：

    token embedding + position embedding
    -> Q/K/V
    -> 把 K/V 写入 cache
    -> 对 cache 中 0..current 的 K/V 做 attention
    -> 输出下一个 token 的 logits

这样既能练习 prefill/decode 的 KV cache 读写，又不会让模型代码淹没主流程。
输出是随机权重产生的乱码，这是预期行为。
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from .cache import KVCache


def _matvec(matrix: list[list[float]], vector: Sequence[float]) -> list[float]:
    return [sum(w * x for w, x in zip(row, vector)) for row in matrix]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exps = [math.exp(x - max_score) for x in scores]
    total = sum(exps)
    return [x / total for x in exps]


class TinyAttentionModel:
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 16,
        max_model_len: int = 32,
        seed: int = 0,
    ) -> None:
        if hidden_size <= 0 or max_model_len <= 0:
            raise ValueError("hidden_size and max_model_len must be positive")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_model_len = max_model_len
        self.scale = 1.0 / math.sqrt(hidden_size)

        rng = random.Random(seed)
        scale = 0.2

        def matrix(rows: int, cols: int) -> list[list[float]]:
            return [
                [rng.uniform(-scale, scale) for _ in range(cols)]
                for _ in range(rows)
            ]

        self.token_emb = matrix(vocab_size, hidden_size)
        self.pos_emb = matrix(max_model_len, hidden_size)
        self.wq = matrix(hidden_size, hidden_size)
        self.wk = matrix(hidden_size, hidden_size)
        self.wv = matrix(hidden_size, hidden_size)
        self.wo = matrix(hidden_size, hidden_size)

    def _step(
        self,
        token_id: int,
        position: int,
        cache: KVCache,
    ) -> list[float]:
        """计算一个 token 的 K/V，写入 cache，并返回下一个 token 的 logits。"""
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"token_id out of range: {token_id}")
        if not 0 <= position < self.max_model_len:
            raise ValueError(f"position out of range: {position}")
        if position != cache.num_tokens:
            raise ValueError(
                f"cache position mismatch: position={position}, "
                f"cache.num_tokens={cache.num_tokens}"
            )

        x = [
            self.token_emb[token_id][i] + self.pos_emb[position][i]
            for i in range(self.hidden_size)
        ]

        q = _matvec(self.wq, x)
        k = _matvec(self.wk, x)
        v = _matvec(self.wv, x)

        # 关键：K/V 一旦写入，就留给后续 decode 复用。
        cache.append(k, v)

        # 对 cache 中 0..position 的所有 K/V 做单 head attention。
        scores = [_dot(q, past_k) * self.scale for past_k in cache.keys]
        probs = _softmax(scores)

        attn = [
            sum(probs[j] * cache.values[j][d] for j in range(len(cache.values)))
            for d in range(self.hidden_size)
        ]
        projected = _matvec(self.wo, attn)
        hidden = [x[d] + projected[d] for d in range(self.hidden_size)]

        # tied lm head：直接和 token embedding 做点积。
        return [
            _dot(hidden, self.token_emb[token])
            for token in range(self.vocab_size)
        ]

    def prefill(self, token_ids: Sequence[int], cache: KVCache) -> list[float]:
        """把 prompt 的所有 token 写入 cache，返回最后一个位置的 logits。"""
        if not token_ids:
            raise ValueError("prompt must not be empty")
        if cache.num_tokens != 0:
            raise ValueError("prefill expects an empty cache")

        logits: list[float] = []
        for position, token_id in enumerate(token_ids):
            logits = self._step(token_id, position, cache)
        return logits

    def decode(
        self,
        token_id: int,
        position: int,
        cache: KVCache,
    ) -> list[float]:
        """计算一个新 token 的 K/V，并基于历史 cache 返回下一个 logits。"""
        return self._step(token_id, position, cache)

    def forward_full(self, token_ids: Sequence[int]) -> list[float]:
        """不使用外部 cache 的全量前向，用作 prefill/decode 的 correctness oracle。"""
        cache = KVCache(hidden_size=self.hidden_size)
        logits: list[float] = []
        for position, token_id in enumerate(token_ids):
            logits = self._step(token_id, position, cache)
        return logits
