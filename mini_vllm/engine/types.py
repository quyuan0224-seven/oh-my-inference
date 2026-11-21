"""mini_vllm 的共享数据类型。

这些 dataclass 是模块之间的契约，尽量保持小而明确：

    SchedulerOutput   Scheduler -> ModelRunner
    ModelRunnerOutput ModelRunner -> Scheduler
    SchedulerStep     Scheduler -> EngineCore
    StepResult        EngineCore -> 调用方（demo/server）
    RequestOutput     最终结果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..model.cache import KVCache


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH_CAPPED = "finished_length_capped"

    @property
    def is_finished(self) -> bool:
        return self in {
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        }


@dataclass(frozen=True)
class SamplingParams:
    """mini_vllm 只支持最基本的采样参数。"""

    max_tokens: int = 20
    temperature: float = 0.0
    stop_chars: str = "."

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")


@dataclass
class Request:
    """一个生成请求的完整状态。"""

    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    stop_token_ids: set[int] = field(default_factory=set)

    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    finish_reason: str | None = None

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status.is_finished


@dataclass(frozen=True)
class ScheduledRequest:
    """Scheduler 交给 ModelRunner 的一个请求分片。"""

    request: Request
    phase: str  # "prefill" or "decode"
    token_ids: list[int]
    start_pos: int
    cache: KVCache
    needs_logits: bool

    @property
    def num_new_tokens(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class SchedulerOutput:
    """Scheduler 交给 ModelRunner 的一步工作。"""

    scheduled: list[ScheduledRequest]

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(item.num_new_tokens for item in self.scheduled)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


@dataclass(frozen=True)
class ModelRunnerOutput:
    """ModelRunner 执行一步后的批量结果。"""

    sampled_token_ids: dict[str, list[int]]
    num_forwarded_tokens: int


@dataclass(frozen=True)
class SchedulerStep:
    """Scheduler 更新完请求状态后的结果（还不含文本解码）。"""

    request: Request
    phase: str
    token_id: int | None
    finished: bool
    finish_reason: str | None


@dataclass(frozen=True)
class StepResult:
    """EngineCore 返回给 demo/server 的单步结果。"""

    request_id: str
    phase: str
    token_id: int | None
    token_char: str | None
    output_text: str
    output_token_ids: list[int]
    num_computed_tokens: int
    finished: bool
    finish_reason: str | None


@dataclass(frozen=True)
class RequestOutput:
    """一个请求的最终结果。"""

    request_id: str
    prompt: str
    output_text: str
    output_token_ids: list[int]
    finish_reason: str | None
