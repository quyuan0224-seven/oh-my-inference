# 从 vLLM 到最小 Inference Engine

> 调研基准：2026-11，vLLM 0.33.x / V1 主线。
> 当前进度：**V0.1 已实现**；V1（paged KV、continuous batching、prefix caching 等）仍是设计稿。

## 先看什么

### 1. V0.1：模块齐全、实现极简

[V0.1 快速开始](v0_1/README.md) · [V0.1 设计与实现](docs/03-v0.1-design-and-implementation.md)

V0.1 保留了所有核心高层模块，只是把内部实现压缩到最小：

```text
Server
  -> EngineCore
       -> Scheduler
       -> KVCacheManager
       -> ModelRunner -> TinyAttentionModel + Sampler
```

代码按功能模块分目录：

```text
v0_1/
├── serving/        # HTTP server
├── engine/         # EngineCore / Scheduler / KVCacheManager / ModelRunner / types
├── model/          # toy model / KV storage / sampler
├── examples/
│   └── prefill_decode_kv_cache_trace.py
└── tests/
```

它支持：

- 单请求串行、FCFS 排队；
- prefill 一次写入 prompt 的 KV；
- decode 每步只算一个新 token；
- 每请求一段连续 KV cache；
- 最小 HTTP `/generate`；
- prefill/decode 与 full-forward 的 KV cache 正确性测试。

快速运行（当前环境只有 `python3`，没有 `python` 别名）：

```bash
python3 -m v0_1.examples.prefill_decode_kv_cache_trace --max-tokens 6 --temperature 0.8
python3 -m unittest v0_1.tests.test_v0_1 -v
python3 -m v0_1.serving.server --port 8000
```

V0.1 刻意不做的：continuous batching、paged KV cache、prefix caching、preemption、真实模型、TP/DP、流式输出。这些是 V1 的内容，不是 V0.1 的遗漏。

### 2. vLLM 调研与 V1 设计（后续蓝图）

1. [vLLM 架构与核心模块调研](docs/01-vllm-architecture-research.md)
   - vLLM 的分层与进程模型
   - Scheduler、KV Cache Manager、Worker/ModelRunner、Sampler 等模块职责
   - 请求生命周期、continuous batching、paged attention、chunked prefill、prefix caching、preemption
   - 哪些是核心，哪些是工程化能力

2. [最小 Custom Inference 设计方案（V1 讨论稿）](docs/02-mini-inference-design-proposal.md)
   - paged KV cache / block table
   - continuous batching 与 token-budget scheduler
   - 正确性验证、里程碑、技术取舍

### 3. 多主机协作与记账

后续可能在不同主机上开发。开始前读：

- [AGENTS.md](AGENTS.md)：硬性开发约定；
- [项目状态与任务认领](docs/00-status.md)：当前进度、任务认领、决策和变更日志。

设计方向和里程碑以 [V1 设计方案](docs/02-mini-inference-design-proposal.md) 为准；状态页只记录执行状态，不重复设计内容。

## 版本关系

| 版本 | 已实现 | 目的 |
|---|---|---|
| V0.1 | ✅ | 建立 server → engine core → scheduler / kv cache / model runner 的最基础认知 |
| V1 | 设计稿 | 加入 paged KV、continuous batching、prefix caching、preemption 等 |

V0.1 的代码在 [`v0_1/`](v0_1/)，V1 的设计在 [`docs/`](docs/)。
