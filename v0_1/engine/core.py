"""EngineCore：把 Scheduler / KVCacheManager / ModelRunner 串成 step loop。

这是 V0.1 的“引擎核心”，对应 vLLM 的 EngineCore：
- 接收请求；
- 每一步调用 scheduler.schedule()；
- 把 SchedulerOutput 交给 model runner；
- 把 ModelRunnerOutput 交回 scheduler 更新状态；
- 请求结束后组装 RequestOutput。
"""

from __future__ import annotations

from .kv_cache import KVCacheManager
from ..model import Sampler, TinyAttentionModel
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .tokenizer import CharTokenizer
from .types import (
    Request,
    RequestOutput,
    SamplingParams,
    StepResult,
)


class EngineCore:
    def __init__(
        self,
        tokenizer: CharTokenizer,
        model: TinyAttentionModel,
        seed: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.max_model_len = model.max_model_len

        self.kv_cache_manager = KVCacheManager(hidden_size=model.hidden_size)
        self.scheduler = Scheduler(
            kv_cache_manager=self.kv_cache_manager,
            max_model_len=self.max_model_len,
        )
        self.model_runner = ModelRunner(
            model=model,
            sampler=Sampler(seed=seed),
        )

        self._next_request_id = 0
        self._requests: dict[str, Request] = {}
        self._finished: dict[str, RequestOutput] = {}

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> str:
        params = sampling_params or SamplingParams()
        prompt_token_ids = self.tokenizer.encode(prompt)

        if len(prompt_token_ids) >= self.max_model_len:
            raise ValueError(
                f"prompt length {len(prompt_token_ids)} >= "
                f"max_model_len {self.max_model_len}"
            )

        request_id = f"req-{self._next_request_id:03d}"
        self._next_request_id += 1

        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=params,
            stop_token_ids=self.tokenizer.stop_token_ids(params.stop_chars),
        )
        self._requests[request_id] = request
        self.scheduler.add_request(request)
        return request_id

    def step(self) -> StepResult | None:
        """执行一个调度步；没有请求时返回 None。"""
        scheduler_output = self.scheduler.schedule()
        if scheduler_output is None:
            return None

        model_output = self.model_runner.execute(scheduler_output)
        scheduler_step = self.scheduler.update_from_output(
            scheduler_output,
            model_output,
        )

        request = scheduler_step.request
        output_token_ids = list(request.output_token_ids)
        output_text = self.tokenizer.decode(output_token_ids)
        result = StepResult(
            request_id=request.request_id,
            phase=scheduler_step.phase,
            token_id=scheduler_step.token_id,
            token_char=self.tokenizer.decode([scheduler_step.token_id]),
            output_text=output_text,
            output_token_ids=output_token_ids,
            num_computed_tokens=request.num_computed_tokens,
            finished=scheduler_step.finished,
            finish_reason=scheduler_step.finish_reason,
        )

        if scheduler_step.finished:
            self._finished[request.request_id] = self._make_request_output(
                request_id=request.request_id,
                prompt=request.prompt,
                output_text=output_text,
                output_token_ids=output_token_ids,
                finish_reason=scheduler_step.finish_reason,
            )

        return result

    def has_requests(self) -> bool:
        return self.scheduler.has_requests()

    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> RequestOutput:
        """同步生成一个请求；要求 engine 空闲，由 step() 的返回值判断完成。"""
        if self.has_requests():
            raise RuntimeError(
                "generate() requires an idle engine; "
                "use add_request() + run_until_idle() for queued requests"
            )

        request_id = self.add_request(prompt, sampling_params)

        while True:
            result = self.step()
            if result is None:
                raise RuntimeError("engine has no work but request is unfinished")
            if result.request_id != request_id:
                raise RuntimeError("scheduler returned an unexpected request")

            if result.finished:
                return self._make_request_output(
                    request_id=request_id,
                    prompt=prompt,
                    output_text=result.output_text,
                    output_token_ids=result.output_token_ids,
                    finish_reason=result.finish_reason,
                )

    @staticmethod
    def _make_request_output(
        request_id: str,
        prompt: str,
        output_text: str,
        output_token_ids: list[int],
        finish_reason: str | None,
    ) -> RequestOutput:
        return RequestOutput(
            request_id=request_id,
            prompt=prompt,
            output_text=output_text,
            output_token_ids=list(output_token_ids),
            finish_reason=finish_reason,
        )

    def run_until_idle(self) -> list[RequestOutput]:
        """把所有已提交请求跑完，返回按完成顺序排列的结果。"""
        while self.has_requests():
            self.step()
        return list(self._finished.values())

    @property
    def finished_outputs(self) -> list[RequestOutput]:
        return list(self._finished.values())
