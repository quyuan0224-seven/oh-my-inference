"""ModelRunner：执行 Scheduler 给出的这一步工作。

职责与 vLLM 的 GPUModelRunner 对应，但实现是极简版：
- prefill：把 prompt 的 K/V 写入 cache；
- decode：只计算新 token 的 K/V；
- 调用 Sampler 得到下一个 token；
- 返回 ModelRunnerOutput。

V0.1 没有 tensor batch、没有 attention backend 抽象、没有 CUDA graph。
"""

from __future__ import annotations

from ..model import Sampler, TinyAttentionModel
from .scheduler import SchedulerOutput
from .types import ModelRunnerOutput


class ModelRunner:
    def __init__(self, model: TinyAttentionModel, sampler: Sampler) -> None:
        self.model = model
        self.sampler = sampler

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        request = scheduler_output.request
        cache = scheduler_output.cache

        if scheduler_output.phase == "prefill":
            logits = self.model.prefill(scheduler_output.token_ids, cache)
        elif scheduler_output.phase == "decode":
            if len(scheduler_output.token_ids) != 1:
                raise ValueError("decode step must contain exactly one token")
            logits = self.model.decode(
                token_id=scheduler_output.token_ids[0],
                position=scheduler_output.start_pos,
                cache=cache,
            )
        else:
            raise ValueError(f"unknown phase: {scheduler_output.phase}")

        token_id = self.sampler.sample(
            logits,
            temperature=request.sampling_params.temperature,
        )
        return ModelRunnerOutput(
            request_id=request.request_id,
            token_id=token_id,
        )
