"""ModelRunner：执行 Scheduler 给出的这一步工作。

职责与 vLLM 的 GPUModelRunner 对应，但实现是极简版：
- prefill：把 prompt 的 K/V 写入 cache；
- decode：只计算新 token 的 K/V；
- 调用 Sampler 得到下一个 token；
- 返回 ModelRunnerOutput。

mini_vllm 没有 tensor batch、没有 attention backend 抽象、没有 CUDA graph。
"""

from __future__ import annotations

from ..model import Sampler, TinyAttentionModel
from .types import ModelRunnerOutput, SchedulerOutput


class ModelRunner:
    def __init__(self, model: TinyAttentionModel, sampler: Sampler) -> None:
        self.model = model
        self.sampler = sampler

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        sampled_token_ids: dict[str, list[int]] = {}
        num_forwarded_tokens = 0

        for scheduled in scheduler_output.scheduled:
            request = scheduled.request
            cache = scheduled.cache

            if scheduled.phase == "prefill":
                logits = self.model.prefill(scheduled.token_ids, cache)
            elif scheduled.phase == "decode":
                if len(scheduled.token_ids) != 1:
                    raise ValueError("decode step must contain exactly one token")
                logits = self.model.decode(
                    token_id=scheduled.token_ids[0],
                    position=scheduled.start_pos,
                    cache=cache,
                )
            else:
                raise ValueError(f"unknown phase: {scheduled.phase}")

            num_forwarded_tokens += scheduled.num_new_tokens

            if scheduled.needs_logits:
                token_id = self.sampler.sample(
                    logits,
                    temperature=request.sampling_params.temperature,
                )
                sampled_token_ids[request.request_id] = [token_id]

        return ModelRunnerOutput(
            sampled_token_ids=sampled_token_ids,
            num_forwarded_tokens=num_forwarded_tokens,
        )
