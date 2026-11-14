"""把 logits 变成 token id。"""

from __future__ import annotations

import math
import random
from typing import Sequence


def _softmax(logits: Sequence[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


class Sampler:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def sample(self, logits: Sequence[float], temperature: float) -> int:
        if temperature <= 1e-8:
            return max(range(len(logits)), key=logits.__getitem__)

        probs = _softmax([x / temperature for x in logits])
        return self.rng.choices(range(len(probs)), weights=probs, k=1)[0]
