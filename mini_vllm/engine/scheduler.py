"""mini_vllm 的极简 Scheduler。

mini_vllm 的调度策略：
- FCFS：waiting 队列头部先来先服务；
- 同一时刻只有一个 running 请求；
- 先做一次 prefill，把 prompt 的 K/V 写满 cache；
- 之后每一步 decode 一个 token，直到 stop / max_tokens / max_model_len。

它和 V1 的差距是“没有 continuous batching、没有 token budget 混合调度”，
但 Scheduler 的职责边界已经存在：
  - 决定谁运行；
  - 决定这一步是 prefill 还是 decode；
  - 向 KVCacheManager 申请/释放 cache；
  - 根据 ModelRunnerOutput 更新 Request 状态。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..model.cache import KVCache
from .kv_cache import KVCacheManager
from .types import (
    ModelRunnerOutput,
    Request,
    RequestStatus,
    SchedulerStep,
)


@dataclass(frozen=True)
class SchedulerOutput:
    """Scheduler 交给 ModelRunner 的一步工作。"""

    request: Request
    phase: str  # "prefill" or "decode"
    token_ids: list[int]
    start_pos: int
    cache: KVCache

    @property
    def num_new_tokens(self) -> int:
        return len(self.token_ids)


class Scheduler:
    def __init__(self, kv_cache_manager: KVCacheManager, max_model_len: int) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.max_model_len = max_model_len
        self.waiting: deque[Request] = deque()
        self.running: Request | None = None

    def add_request(self, request: Request) -> None:
        if len(request.prompt_token_ids) >= self.max_model_len:
            raise ValueError("prompt is too long for max_model_len")
        self.waiting.append(request)

    def has_requests(self) -> bool:
        return self.running is not None or bool(self.waiting)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return 1 if self.running is not None else 0

    def schedule(self) -> SchedulerOutput | None:
        """返回下一步工作；没有请求时返回 None。"""
        if self.running is None:
            if not self.waiting:
                return None

            request = self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            cache = self.kv_cache_manager.allocate(request)
            self.running = request

            # 第一个 step：prefill 整个 prompt。
            return SchedulerOutput(
                request=request,
                phase="prefill",
                token_ids=list(request.prompt_token_ids),
                start_pos=0,
                cache=cache,
            )

        request = self.running
        if not request.output_token_ids:
            raise RuntimeError("running request has no output token to decode")

        # 后续 step：只计算最近采样出的那个 token。
        # 它的 KV 还没有写入 cache，写完后再采样下一个 token。
        return SchedulerOutput(
            request=request,
            phase="decode",
            token_ids=[request.output_token_ids[-1]],
            start_pos=request.num_computed_tokens,
            cache=self.kv_cache_manager.get(request.request_id),
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> SchedulerStep:
        """把采样结果写回 Request，并判断是否结束。"""
        request = scheduler_output.request
        request.num_computed_tokens += scheduler_output.num_new_tokens
        request.output_token_ids.append(model_output.token_id)

        finished = False
        finish_reason: str | None = None

        if model_output.token_id in request.stop_token_ids:
            finished = True
            finish_reason = "stop"
        elif len(request.output_token_ids) >= request.sampling_params.max_tokens:
            finished = True
            finish_reason = "length"
        elif request.num_tokens >= self.max_model_len:
            finished = True
            finish_reason = "length"

        if finished:
            request.finish_reason = finish_reason
            if finish_reason == "stop":
                request.status = RequestStatus.FINISHED_STOPPED
            else:
                request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            self.kv_cache_manager.free(request.request_id)
            self.running = None

        return SchedulerStep(
            request=request,
            phase=scheduler_output.phase,
            token_id=model_output.token_id,
            finished=finished,
            finish_reason=finish_reason,
        )
