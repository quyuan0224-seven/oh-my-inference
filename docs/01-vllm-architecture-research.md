# vLLM 架构与核心模块调研

> 调研时间：2026-11  
> 代码基准：vLLM 0.33.x / V1 主线（`main` 分支）  
> 说明：V1 仍在快速演进，具体文件位置可能变化；本文按“职责”而不是行号来组织。早期 V1 的 `gpu_model_runner.py` 现在位于 `vllm/v1/worker/gpu/model_runner.py`，阅读旧文章时需要注意路径差异。

---

## 0. 先给结论

如果把 vLLM 当成一个黑盒，它做的事情可以压缩成一句话：

> **不断从请求队列里挑出一批 token，为它们分配 KV cache 空间，调用模型计算这些 token 的 KV 和 logits，采样出下一个 token，更新请求状态，再释放已经不需要的资源。**

这句话对应五个核心抽象：

| 抽象 | 解决的问题 | vLLM 中的主要承载者 |
|---|---|---|
| Request / Sequence | 一个请求当前算到哪、生成了什么、是否结束 | `Request`、`RequestStatus` |
| Scheduler | 这一步让哪些请求各算多少 token | `Scheduler.schedule()`、`SchedulerOutput` |
| KV Cache Manager | 每个请求的 KV 存在哪些物理块里，何时分配/释放/复用 | `KVCacheManager`、`BlockPool`、block table |
| Model Runner | 把调度结果变成模型输入，执行 forward，产出 logits/token | `GPUModelRunner`、attention backend、worker |
| Engine Loop | 把上面几件事按 step 串起来，并处理输入/输出 | `EngineCore.step()`、`LLMEngine`、`OutputProcessor` |

理解 vLLM，最重要的不是记住文件，而是记住三个“契约”：

1. **Scheduler → Worker**：`SchedulerOutput`，核心是 `{request_id: num_scheduled_tokens}` 加上每个请求的 block table / 新请求元数据。
2. **Worker → Scheduler**：`ModelRunnerOutput`，核心是每个请求这一步采样出的 token（以及 logprobs 等可选信息）。
3. **Scheduler/KV Manager → Request**：`num_computed_tokens`、`block_table`、`status` 这些状态在每一步被精确更新。

把这三个契约吃透，再去看 continuous batching、chunked prefill、prefix caching、preemption，都会变得顺理成章。

---

## 1. 版本背景：为什么以 V1 为主线

vLLM V1 在 2025 年初发布 alpha，之后逐步成为默认架构。它相对 V0 的关键变化：

- **统一调度循环**：不再有独立的 prefill 阶段和 decode 阶段。调度决策表示为 `{request_id: num_tokens}`，prefill、decode、chunked prefill 都由同一个 token 预算循环产生。
- **EngineCore 独立**：把调度和模型执行放进单独的 EngineCore 进程/循环，与 API server、tokenization、detokenization 解耦，降低 CPU 开销、方便重叠。
- **KV cache 管理重构**：`KVCacheManager` + `BlockPool` 成为更清晰的一层，prefix caching 默认开启且开销很低。
- **Persistent Batch**：模型输入 tensor 持久化，每步只更新 diff，减少 CPU 侧重建 tensor 的开销。
- **Tensor Parallel 更对称**：worker 侧缓存请求状态，只传增量，调度器和 Worker 0 不必强行同进程。
- **torch.compile + piecewise CUDA graphs**：在保持动态 batch 的同时降低 kernel launch / CPU 开销。

因此，今天学习 vLLM，应该直接以 V1 为主。网上很多“prefill 阶段 / decode 阶段分开调度”的文章描述的是 V0，容易造成认知偏差。

参考：

- [vLLM V1: A Major Upgrade to vLLM’s Core Architecture](https://vllm-project.github.io/2025/01/27/v1-alpha-release.html)
- [Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [vLLM V1 User Guide](https://docs.vllm.ai/en/latest/usage/v1_guide/)

---

## 2. 全局架构

### 2.1 进程视角

在线 serving 的典型部署不是“一个 Python 进程跑完所有事”，而是：

| 进程 | 数量 | 主要职责 |
|---|---:|---|
| API Server | 默认等于 DP size，可配置 | HTTP/OpenAI 接口、输入预处理、tokenization、流式返回 |
| EngineCore | 每个 DP rank 一个 | 调度、KV cache 管理、协调模型执行 |
| GPU Worker | `DP × PP × TP`，每 GPU 一个 | 加载模型、持有 KV cache 显存、执行 forward |
| DP Coordinator | DP > 1 时 1 个 | DP rank 间负载均衡、同步 |

单卡、TP=1、DP=1 时，典型是：

```text
API Server 进程
  └── ZMQ ──> EngineCore 进程
                 └── Executor ──> 1 个 GPU Worker 进程
                                      └── GPUModelRunner
                                            ├── Model
                                            ├── Attention Backend
                                            └── Sampler
```

离线 `LLM` 也通过 `EngineCoreClient` 连接 EngineCore；默认常用 `SyncMPClient`（ZMQ + 后台 EngineCore 进程），特殊场景可用 `InprocClient`。所以“EngineCore 是调度与执行的核心循环”这个抽象，在离线和在线都成立。

参考：[Architecture Overview - V1 Process Architecture](https://docs.vllm.ai/en/latest/design/arch_overview/)

### 2.2 代码分层

```text
┌──────────────────────────────────────────────────────────────┐
│ Entrypoints                                                  │
│  LLM / AsyncLLM / OpenAI API Server                          │
│  - 接收请求、tokenize、multimodal 预处理、流式输出           │
└──────────────────────────────┬───────────────────────────────┘
                               │ EngineCoreRequest / Output
┌──────────────────────────────▼───────────────────────────────┐
│ LLMEngine / AsyncLLM                                         │
│  - InputProcessor：校验、tokenize、构造 Request              │
│  - OutputProcessor：detokenize、组装 RequestOutput、流式     │
│  - Detokenizer：增量解码，处理 UTF-8 边界                    │
└──────────────────────────────┬───────────────────────────────┘
                               │ EngineCoreClient (ZMQ / inproc)
┌──────────────────────────────▼───────────────────────────────┐
│ EngineCore（调度 + 执行循环）                                 │
│                                                              │
│  ┌────────────────┐        ┌──────────────────────────────┐  │
│  │ Scheduler      │<------>│ KVCacheManager / BlockPool   │  │
│  │ - waiting      │        │ - block table                │  │
│  │ - running      │        │ - allocate / free / cache    │  │
│  │ - token budget │        │ - prefix cache               │  │
│  └───────┬────────┘        └──────────────────────────────┘  │
│          │ SchedulerOutput                                   │
│  ┌───────▼────────────────────────────────────────────────┐  │
│  │ Executor                                               │  │
│  │  - 单卡：直接调用 worker                                │  │
│  │  - 多卡：MultiprocExecutor / Ray / 自定义后端           │  │
│  └───────┬────────────────────────────────────────────────┘  │
└──────────┼───────────────────────────────────────────────────┘
           │ execute_model / sample_tokens
┌──────────▼───────────────────────────────────────────────────┐
│ Worker                                                       │
│  - 设备初始化、模型加载、KV cache 显存分配                   │
│  - GPUModelRunner                                            │
│      - update request states                                 │
│      - prepare input tensors                                 │
│      - prepare attention metadata / block tables             │
│      - model forward                                        │
│      - sampler                                               │
└──────────┬───────────────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────────┐
│ Model + Attention Backend                                    │
│  - Decoder-only Transformer / MoE / VLM ...                  │
│  - FlashAttention / FlashInfer / Triton / ...                │
│  - PagedAttention：按 block table 读取 KV                    │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 一张表看懂主要模块

| 层 | 模块 | 关键代码 | 核心职责 | 关键输入/输出 |
|---|---|---|---|---|
| 入口 | `LLM` / `AsyncLLM` | `vllm/entrypoints/llm.py`, `vllm/v1/engine/async_llm.py` | 离线/在线 API；请求生命周期；流式输出 | prompt → `RequestOutput` |
| 输入处理 | `InputProcessor` | `vllm/v1/engine/input_processor.py` | 校验参数、tokenize、多模态预处理、构造 `EngineCoreRequest` | 原始输入 → `Request` |
| 引擎前端 | `LLMEngine` | `vllm/v1/engine/llm_engine.py` | 组织 input/output processor，调用 EngineCore | `add_request` / `step` |
| 核心循环 | `EngineCore` | `vllm/v1/engine/core.py` | 调度 → 执行 → 更新，busy loop | `SchedulerOutput` → `EngineCoreOutputs` |
| 调度 | `Scheduler` | `vllm/v1/core/sched/scheduler.py` | token 预算、running/waiting、抢占、构造调度结果 | `SchedulerOutput` |
| KV 管理 | `KVCacheManager` | `vllm/v1/core/kv_cache_manager.py` | 每个请求的 block table、分配/释放、prefix cache | `KVCacheBlocks` |
| 块池 | `BlockPool` | `vllm/v1/core/block_pool.py` | 物理 block 的 free queue、引用计数、LRU 驱逐 | free/used blocks |
| 执行分发 | `Executor` | `vllm/v1/executor/*` | 单卡/多卡/多进程执行，collective RPC | `execute_model` |
| Worker | `Worker` | `vllm/v1/worker/gpu_worker.py` | 设备、模型、KV 显存、worker 生命周期 | 执行模型 |
| 模型执行 | `GPUModelRunner` | `vllm/v1/worker/gpu/model_runner.py` | 准备输入/attention metadata、forward、采样 | `ModelRunnerOutput` |
| 注意力 | `AttentionBackend` | `vllm/v1/attention/backend.py`, `selector.py`, `backends/*` | 根据 backend 执行 paged attention | KV cache + query |
| 采样 | `Sampler` | `vllm/v1/worker/gpu/sample/sampler.py`, `vllm/v1/sample/ops/topk_topp_sampler.py` | logits 处理、temperature/top-p/top-k、采样 | token ids |
| 输出 | `OutputProcessor` / `Detokenizer` | `vllm/v1/engine/output_processor.py`, `detokenizer.py` | 增量 detokenize、stop string、流式输出、finish reason | `RequestOutput` |
| 模型 | `Model` | `vllm/model_executor/models/*` | 模型结构、权重加载、TP 切分、量化 | hidden states / logits |

补充：vLLM 还有多个横切模块，例如 structured output、speculative decoding、LoRA、multimodal、KV connector / disaggregated serving、metrics、sleep mode、fault tolerance。它们很重要，但不是理解“最小执行流”的第一优先级。

---

## 3. 请求生命周期

### 3.1 状态机

V1 的 `RequestStatus` 可以简化为：

```text
                 add_request
                     │
                     ▼
                 ┌────────┐
                 │ WAITING│
                 └───┬────┘
       schedule 成功  │
                     ▼
                 ┌────────┐   finish(stop/length/abort/error)
                 │ RUNNING│ ──────────────────────────────> FINISHED_*
                 └───┬────┘
                     │ KV 不足，被抢占
                     ▼
                 ┌──────────┐
                 │ PREEMPTED│
                 └────┬─────┘
                      │ 重新入 waiting，清空 KV，num_computed=0
                      └──────────────> WAITING
```

几个容易混淆的点：

- `PREEMPTED` 不是“失败”，而是“暂时被拿走资源，稍后重算”。V1 默认采用 **recompute**，不是把 KV swap 到 CPU。
- 被抢占后，已经生成的 output token 通常保留；恢复时把 prompt + 已生成 token 一起重新 prefill。
- `num_computed_tokens` 表示“这个请求已经有多少个 token 的 KV 被算过”，不是“已经生成了多少 token”。
- `num_tokens = len(prompt) + len(output)`，而 `num_computed_tokens` 通常小于或等于它；只有“追上”时才会采样下一个 token。

### 3.2 从 `add_request` 到输出

```text
Client
  │ prompt
  ▼
LLMEngine / AsyncLLM
  │ 1. InputProcessor: 校验、tokenize、多模态处理
  │ 2. 创建 Request（prompt_token_ids, sampling_params, status=WAITING）
  │ 3. OutputProcessor 登记请求（用于后续 detokenize/stream）
  ▼
EngineCoreClient ──ZMQ/inproc──> EngineCore
  │
  │ 4. Scheduler.add_request(request) -> waiting queue
  ▼
┌───────────────────────────── EngineCore.step() ─────────────────────────────┐
│                                                                              │
│  5. Scheduler.schedule()                                                     │
│       - token_budget = max_num_batched_tokens                                │
│       - 先调度 running，再调度 waiting                                       │
│       - KVCacheManager.allocate_slots() 确保 block table 覆盖新 token        │
│       - 构造 SchedulerOutput                                                 │
│                                                                              │
│  6. Executor.execute_model(SchedulerOutput)                                  │
│       - Worker / GPUModelRunner 更新请求状态                                 │
│       - 准备 input_ids / positions / seq_lens / block_tables / slot_mapping  │
│       - Model forward（attention backend 读 paged KV）                       │
│       - Sampler 采样                                                         │
│       - 返回 ModelRunnerOutput（sampled token ids）                          │
│                                                                              │
│  7. Scheduler.update_from_output()                                           │
│       - 追加 sampled token                                                   │
│       - 判断 stop / length / abort                                           │
│       - 释放完成请求的 KV blocks                                             │
│       - 生成 EngineCoreOutput                                                │
│                                                                              │
│  注：vLLM 在 schedule() 之后的 _update_after_schedule() 里乐观推进           │
│      num_computed_tokens，以便下一步立即继续调度；spec token 被拒绝时        │
│      再在 update_from_output() 中回滚。Mini 版可以选择执行成功后再推进。     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
LLMEngine / OutputProcessor
  │ 8. detokenize、组装 RequestOutput、流式返回
  ▼
Client
```

### 3.3 一次 step 的职责边界

`EngineCore.step()` 的伪代码非常短，值得记住：

```python
def step():
    if not scheduler.has_requests():
        return empty

    scheduler_output = scheduler.schedule()

    future = model_executor.execute_model(scheduler_output, non_block=True)
    model_output = wait(future)
    # 如果 execute_model 只做了 forward，这里再 sample_tokens

    engine_outputs = scheduler.update_from_output(scheduler_output, model_output)
    return engine_outputs
```

真正的复杂度都在三个地方：

- `scheduler.schedule()`：谁上、上多少 token、KV 够不够；
- `model_runner.execute_model()`：如何把逻辑请求变成正确的 tensor 和 attention metadata；
- `scheduler.update_from_output()`：如何根据采样结果推进状态、判断结束、释放资源。

---

## 4. 核心模块深挖

### 4.1 Scheduler：token 预算，而不是 prefill/decode 阶段

V1 scheduler 的核心注释说得很直接：

> There’s no “decoding phase” nor “prefill phase” in the scheduler. Each request just has `num_computed_tokens` and `num_tokens_with_spec`. At each step, the scheduler tries to assign tokens to the requests so that each request’s `num_computed_tokens` can catch up its `num_tokens_with_spec`.

也就是说，调度器的输入不是“这个请求是 prefill 还是 decode”，而是：

- 这个请求还有多少 token 没算：`num_tokens - num_computed_tokens`；
- 这一步总共有多少 token 预算：`max_num_batched_tokens`；
- KV cache 还能不能放下新 token 所需的 block；
- 当前 running 数量是否超过 `max_num_seqs`。

调度循环大致是：

```text
token_budget = max_num_batched_tokens

# 1. running 优先：让已经在跑的请求继续推进
for request in running:
    n = min(request.num_uncomputed_tokens, token_budget)
    if kv_cache_manager.allocate_slots(request, n):
        schedule(request, n)
        token_budget -= n
    else:
        preempt(victim)
        retry

# 2. waiting：在预算和 max_num_seqs 允许时接纳新请求
while token_budget > 0 and len(running) < max_num_seqs:
    request = waiting.peek()
    n = min(request.num_uncomputed_tokens, token_budget)
    if not kv_cache_manager.allocate_slots(request, n):
        break
    waiting.pop()
    running.append(request)
    schedule(request, n)
    token_budget -= n
```

几个关键设计：

- **running 优先**：已经在跑的请求先获得预算，保证 decode 延迟；waiting 请求只有在预算/槽位剩余时才被接纳。
- **FCFS / priority**：V1 支持两种策略。FCFS 下 KV 不足时抢占最后一个 running；priority 下按优先级选择 victim。
- **抢占 = 释放 + 回队列 + 重算**：被抢占请求的 block 被释放，`num_computed_tokens` 归零，放回 waiting。恢复时重新 prefill。prefix caching 或 KV connector 可以让一部分前缀不用重算，但这是优化，不是语义前提。
- **`num_computed_tokens` 的推进时机**：V1 在 `schedule()` 之后调用 `_update_after_schedule()`，先乐观地把 `num_computed_tokens` 加上本步调度量，这样下一步可以立即继续调度；如果 speculative token 被拒绝，再在 `update_from_output()` 里回滚。理解这个“先推进、后修正”对读 V1 源码很重要。
- **`max_num_seqs` 是并发上限**，`max_num_batched_tokens` 是每步计算量上限，KV blocks 是内存上限。三个约束共同决定这一步的 batch。

> 术语说明：V1 代码里调度器实际使用的 token 预算变量是 `max_num_scheduled_tokens`，它通常由配置的 `max_num_batched_tokens` 决定/派生。本文为了讲解统一写成 `max_num_batched_tokens`。
- **chunked prefill 是自然结果**：如果 prompt 有 1000 token，预算只有 256，那第一步就只算 256，请求仍然是 running；下一步继续算剩余。调度器不需要“chunked prefill 模式”分支。

`SchedulerOutput` 的核心内容（简化）：

```text
SchedulerOutput
├── scheduled_new_reqs: 新请求的元数据（prompt、block ids、sampling params ...）
├── scheduled_cached_reqs: 已在 running 的请求的增量信息
├── num_scheduled_tokens: {request_id: n_tokens_this_step}
├── total_num_scheduled_tokens
├── finished_req_ids
├── preempted_req_ids
├── scheduled_spec_decode_tokens（可选）
└── ...
```

最核心的 contract 就是 `num_scheduled_tokens`。Worker 只需要知道“每个请求这一步要算多少 token”，而不需要知道这些 token 是 prompt 还是 output，是 prefill 还是 decode。

### 4.2 KV Cache Manager / BlockPool：把 KV 当作页式内存管理

PagedAttention 的核心思想，来自操作系统虚拟内存：

- 把每个序列的 KV cache 切成长度为 `block_size` 的块；
- 每个请求维护一张 **block table**，逻辑块号 → 物理块号；
- 物理块不需要连续；
- 块可以共享（prefix caching），用引用计数保护；
- 块用完后回到 free pool，按 LRU 等策略复用。

一个请求的 block table 可能长这样：

```text
请求 A 的逻辑 token 位置:  0  1  2  3 | 4  5  6  7 | 8  9
逻辑 block:                 0          1          2
物理 block:                 7          1          5
block_table = [7, 1, 5]

token position p 的 KV 地址:
  physical_block = block_table[p // block_size]
  slot_in_block  = p % block_size
```

`BlockPool` 负责：

- 维护所有物理 block；
- free queue：分配时从头部取，释放时放回；
- `ref_cnt`：有多少请求/多少共享前缀正在使用这个 block；
- prefix cache 的 hash → block 映射；
- 驱逐：当需要新 block 时，从 free queue 尾部找可驱逐的缓存块。

`KVCacheManager` 负责：

- 每个请求的 block table；
- `allocate_slots(request, num_new_tokens)`：确保 block table 能覆盖从 `num_computed_tokens` 到 `num_computed_tokens + num_new_tokens` 的位置；
- `free(request)`：请求结束/被抢占时释放；
- prefix cache 查找、缓存 block、COW（copy-on-write）等。

几个重要细节：

- **block 是分配单位，slot 是写入单位**。模型 forward 时，`slot_mapping` 把每个新 token 映射到 `(physical_block, slot_in_block)`，attention kernel 按这个地址写 K/V。
- **block table 是读路径**。attention 计算时按 block table 找到历史 token 的 K/V，而不是假设它们连续存放。
- **引用计数**是共享的前提。一个 block 可能被多个请求的相同前缀共享；只有 `ref_cnt == 0` 才能回到 free pool。
- **prefix caching 不是“缓存整个请求”**，而是按 block 的 hash 前缀复用。相同前缀的请求可以跳过已经算过的 block；最后一个不完整 block 通常需要特殊处理（COW 或 partial block caching）。
- **多种 KV cache group**：当前 vLLM 支持 full attention、sliding window、Mamba 等不同 cache 类型，因此有 `KVCacheCoordinator` / `SingleTypeKVCacheManager` 之类的抽象。最小实现只需要一种 full attention group。

### 4.3 Worker / GPUModelRunner：把调度结果变成模型输入

`GPUModelRunner` 是“逻辑调度”与“物理 tensor”之间的转换层。它主要做：

1. **更新请求状态**
   - 新请求加入：初始化 token ids、block table、sampling 状态；
   - 已有请求：更新 `num_computed_tokens`、block table、要处理的 token；
   - 完成/抢占：释放 runner 侧状态。

2. **准备输入**
   - `input_ids`：这一步所有请求的新 token 拼成一条扁平序列；
   - `positions`：每个 token 的绝对位置（RoPE 需要）；
   - `seq_lens`：每个请求当前的总长度；
   - `query_start_loc`：扁平序列中每个请求的起始偏移；
   - `block_tables`：每个请求的 block table；
   - `slot_mapping`：每个新 token 写入 KV cache 的物理 slot；
   - `logits_indices`：哪些位置的 logits 需要采样。

3. **准备 attention metadata**
   - 每个 attention backend 有自己的 metadata 格式；
   - 包括 block table、seq len、query len、slot mapping、是否 causal 等；
   - 这一步把“调度语义”翻译成“kernel 参数”。

4. **执行 forward + sample**
   - 模型输出 hidden states；
   - 取需要采样的位置，过 lm_head 得到 logits；
   - Sampler 根据 sampling params 采样；
   - 返回 `ModelRunnerOutput`，其中核心是 `sampled_token_ids`。

一个容易忽略的点：**V1 的 batch 是“扁平 token batch”**。多个请求的 token 被拼成一个一维序列，用 `query_start_loc` / `seq_lens` 区分，而不是每个请求一个独立 tensor。这正是 continuous batching 和 chunked prefill 能高效混合的原因。

`GPUModelRunner` 里的 persistent batch、CUDA graph、torch.compile 都是性能优化；它们不改变上面的语义。初学时可以完全忽略，先理解 `input_ids / positions / block_tables / slot_mapping / logits_indices` 这几个东西。

### 4.4 Attention Backend：PagedAttention 现在在哪里

很多旧文章会指向 `csrc/attention/attention_kernels.cu` 里的 `paged_attention_kernel`。这个文件已经不存在了。当前 vLLM 的 paged attention 实现在各个 attention backend 中，例如：

- FlashAttention backend
- FlashInfer backend
- Triton backend
- 其他硬件/框架后端

`vllm/v1/attention/backend.py` 定义抽象接口，`selector.py` 负责根据模型、硬件、dtype、特性选择 backend。`vllm/v1/attention/ops/paged_attn.py` 主要提供 KV cache layout 辅助函数（如 `split_kv_cache`、`write_to_paged_cache`），不再负责完整的 attention 计算。

对最小实现来说，不需要写 kernel。只需要：

- 用一个 `[num_layers][num_blocks][block_size][num_heads][head_dim]` 的 KV 存储；
- 写一个 `write_kv(layer, block, slot, head, k, v)`；
- 写一个 `paged_attention(query, block_table, seq_len)`，按 block table 逐 token gather K/V，做 softmax 加权。

这会让代码看起来“很慢”，但能把 paged attention 的地址计算和因果 mask 讲清楚。后面替换成 PyTorch/Triton kernel 时，调度器和 KV manager 不需要改。

### 4.5 Sampler：logits → token

vLLM 的 sampler 不只是“argmax 或按概率抽样”，它还处理：

- temperature、top-p、top-k；
- repetition/presence/frequency penalty；
- logit bias；
- structured output 的 grammar mask；
- seeded RNG；
- logprobs 返回。

在 V1 中，sampler 在 worker/model runner 侧执行，输出 `sampled_token_ids`；Scheduler 拿到 token 后再判断 stop/length。这个分工很重要：**采样是模型执行的一部分，停止判断是请求状态管理的一部分**。

最小实现只需要 greedy + temperature + top-p，stop token 由 scheduler 判断。

### 4.6 OutputProcessor / Detokenizer：模型 token 到用户文本

`OutputProcessor` 维护每个请求的前端状态：

- 已经产出的 token / text；
- incremental detokenizer；
- finish reason；
- 流式输出的增量；
- stop string 处理（可能需要回退已输出的 token）；
- abort 请求；
- 统计信息（TTFT、TPOT 等）。

`Detokenizer` 负责增量解码，尤其是 UTF-8 多字节字符边界：一个 token 可能只对应半个汉字/emoji，不能简单地对每个 token 单独 decode。

最小实现可以用 char-level tokenizer，直接 `decode(output_token_ids)`，牺牲真实性和流式细节，换取理解主流程。

### 4.7 其他横切能力

| 能力 | 作用 | 是否首版需要 |
|---|---|---|
| Prefix Caching | 复用相同前缀的 KV block，降低 TTFT | 否，第二阶段 |
| Chunked Prefill | 长 prompt 分块，和 decode 混合 | 是，V1 核心认知 |
| Preemption | KV 不足时抢占 running，释放后重算 | 建议第二阶段 |
| Speculative Decoding | draft model 提议、target model 验证 | 否 |
| Structured Output | grammar/JSON 约束采样 | 否 |
| LoRA | 多适配器动态加载 | 否 |
| Multimodal | 图像/音频编码、encoder cache | 否 |
| KV Connector / Disaggregation | prefill/decode 分离、KV 传输 | 否 |
| Tensor/Pipeline/Data Parallel | 多卡扩展 | 否 |
| CUDA Graph / torch.compile | 降低 launch/CPU 开销 | 否 |
| Metrics / Tracing | 可观测性 | 简版计数即可 |
| Sleep Mode / Fault Tolerance | 生产运维 | 否 |

---

## 5. 关键机制

### 5.1 Continuous Batching

静态 batching 是“凑齐一批请求，一起跑到全部结束”。问题是：

- 短请求结束后，它的 batch slot 空着，浪费算力；
- 长请求会拖住整批；
- 新请求必须等下一批。

Continuous batching（iteration-level scheduling）是“每一个 forward step 都重新组 batch”：

- 每一步都可以有请求完成、离开；
- 每一步都可以有新请求被接纳、加入；
- batch 的大小和组成是动态的。

在 V1 里，这不是一个单独模块，而是 scheduler 每步重新计算 `num_scheduled_tokens` 的自然结果。只要 Scheduler 每步都能接纳 waiting 请求，并且 Worker 能处理“变化的 batch”，continuous batching 就成立了。

### 5.2 Paged Attention

传统实现通常为每个请求预留 `max_model_len` 的连续 KV 空间，造成严重内部碎片。PagedAttention 把 KV 切成固定大小的 block，按需分配：

```text
传统：每个请求 reserve 一段连续显存
  A: [##############################]  (大部分没用)
  B: [##############################]

Paged：
  A block table: [7, 1, 5]
  B block table: [3, 9]
  物理 block 池: [0][1][2][3][4][5][6][7][8][9]...
```

好处：

- 只有最后一个 block 有内部碎片；
- 物理 block 不需要连续，分配灵活；
- 相同前缀的 block 可以共享（prefix caching）；
- 可以为不同请求动态分配，支持 continuous batching。

代价：

- attention 需要按 block table gather K/V，地址计算更复杂；
- kernel 需要支持非连续 KV；
- block 管理、引用计数、驱逐策略变复杂。

### 5.3 Chunked Prefill

长 prompt 的 prefill 可能占用大量 token 预算。如果一次算完，会阻塞其他请求的 decode，导致 TPOT 抖动。Chunked prefill 把 prompt 拆成多个 chunk，每个 step 只算一部分：

```text
prompt 1000 tokens, budget 256

step 1: compute [0:256)     -> num_computed=256
step 2: compute [256:512)   -> num_computed=512
step 3: compute [512:768)   -> num_computed=768
step 4: compute [768:1000)  -> num_computed=1000, sample first token
```

V1 不需要专门的 chunked prefill 分支；调度器只是给这个请求分配 `min(remaining, budget)` 个 token。它可以把 decode 请求和 prefill chunk 放在同一个 batch 里，因为统一表示都是“要算多少 token”。

### 5.4 Prefix Caching

如果两个请求有相同前缀，例如同一个 system prompt，那么这段前缀的 KV 可以复用。V1 的做法：

- 按 block 计算 hash（通常结合前缀 hash，形成 hash chain）；
- 用 hash 在 `BlockPool` 中查找已缓存的 block；
- 命中则增加 `ref_cnt`，把 block 加入请求的 block table；
- 未命中则正常分配、计算，完成后按 hash 缓存；
- 内存不足时按 LRU 驱逐 `ref_cnt == 0` 的缓存 block。

关键难点：

- 共享 block 不能被某个请求随意写，需要 COW 或只共享完整 block；
- 最后一个不完整 block 的处理；
- hash 碰撞/正确性；
- 多 KV group、多模态 hash、LoRA 等因素。

首版不做 prefix caching 完全合理；但设计 block table 时最好预留“一个 block 可以被多个请求引用”的引用计数语义，避免以后大改。

### 5.5 Preemption

当 KV cache 不足时，scheduler 需要腾出空间。V1 默认是 recompute：

1. 选择 victim（FCFS 下通常是最后一个 running 请求）；
2. 释放它的 KV blocks；
3. `num_computed_tokens = 0`，放回 waiting；
4. 下次被调度时重新 prefill（prompt + 已生成 output）。

另一种方案是 swap 到 CPU，但 V1 主线更倾向 recompute；prefix caching / KV connector 可以降低重算成本。

Preemption 是“KV 内存压力”与“请求进度”之间的 trade-off。最小实现可以先通过配置保证不出现 KV 不足（例如 block 总数足够容纳 `max_num_seqs × max_model_len`），从而暂时不实现抢占；等主流程稳定后再加。

### 5.6 Async Scheduling 与 CPU Overhead

V1 的一个重要目标是降低 CPU 开销：

- EngineCore 独立进程；
- persistent batch：输入 tensor 持久化，只更新 diff；
- async scheduling：GPU 执行 step N 时，CPU 准备 step N+1（用 placeholder 表示尚未采样的 token）；
- CUDA graph / torch.compile：减少 kernel launch 和 Python 开销；
- 输出处理与调度解耦。

这些是性能工程，不是理解执行流的前提。最小实现应该刻意保持同步、单进程、纯 Python，把 CPU overhead 问题留到后续。

### 5.7 Tensor / Pipeline / Data Parallel

- **TP**：把模型权重切到多卡，attention/MLP 做 all-reduce；每张卡持有部分 KV heads 或部分 hidden。
- **PP**：模型按层切分到多 stage，micro-batch 流水。
- **DP**：多个 EngineCore 各自独立处理请求，API server 做路由；DP coordinator 负责负载均衡和同步。

最小实现不需要这些；但要知道 vLLM 的 Worker/Executor 抽象就是为了支持这些并行模式。

---

## 6. 哪些是“核心”，哪些是“工程化”

如果目标是“建立 inference engine 的认知”，建议按下面的优先级：

### 必须理解 / 首版实现

1. **Request 生命周期**：WAITING/RUNNING/FINISHED，`num_computed_tokens`、`num_tokens`、`block_table`。
2. **Token-budget Scheduler**：running 优先、waiting 接纳、`max_num_batched_tokens`、`max_num_seqs`。
3. **Paged KV Cache**：BlockPool、block table、按需分配、释放。
4. **ModelRunner**：把每个请求的 `(start_pos, new_tokens, block_table)` 变成 paged forward。
5. **Sampler**：logits → token，stop/length 判断。
6. **Engine step loop**：schedule → execute → update → output。
7. **Continuous batching + chunked prefill**：通过统一 token 预算自然获得。

### 可以第二阶段加

- Preemption / recompute
- Prefix caching
- 简单 metrics / trace
- 更真实的 tokenizer（BPE）
- PyTorch 化的 batched forward

### 暂不做

- TP/PP/DP
- CUDA kernel / FlashAttention
- OpenAI API server
- Speculative decoding
- Structured output
- LoRA / multimodal
- KV connector / disaggregation
- 量化
- 生产级容错、sleep mode、分布式部署

---

## 7. 对 custom inference 的启示

### 7.1 先定义状态，再写计算

Inference engine 的难点不是矩阵乘法，而是“每个请求现在处于什么状态”。只要把下面这些状态定义清楚，计算可以慢慢替换：

```text
Request:
  prompt_token_ids
  output_token_ids
  num_computed_tokens
  status
  block_table
  sampling_params
  finish_reason
```

### 7.2 调度器只做“分配 token”，不做“执行”

Scheduler 的输出应该是一个纯数据契约，例如：

```text
SchedulerOutput:
  scheduled: [
    {request_id, start_pos, num_new_tokens, token_ids, block_table}
  ]
  total_num_scheduled_tokens
```

Scheduler 不应该知道模型结构、attention kernel、tensor 形状。这样 scheduler 可以独立测试，也可以替换后端。

### 7.3 KV cache 管理要独立于模型

`KVCacheManager` 只管理 block table / 引用计数 / 分配释放；`PagedKVCache` 只管理实际 K/V 存储。两者分离，才能在未来替换成 GPU tensor、不同 attention backend、prefix caching。

### 7.4 用“全量重算”做正确性 oracle

Paged KV cache 很容易写错。最有效的验证方式：

- 实现一个不走 KV cache 的 `forward_full(token_ids)`；
- 实现一个走 paged KV 的 `forward_paged(...)`；
- 对同一段 token，比较最后位置 logits；
- 再比较 greedy 生成结果是否完全一致。

这比“看起来输出像话”可靠得多，尤其当 toy model 的随机权重输出是乱码时。

### 7.5 不要把性能优化混进第一版

Persistent batch、CUDA graph、async scheduling、TP 都会让代码难以理解。第一版应该刻意“慢”：

- 单进程；
- 同步；
- 每个请求顺序 forward；
- Python list 存 KV；
- 显式 for 循环做 attention。

先让控制流正确，再让数据流变快。

---

## 8. 推荐阅读顺序

如果准备继续深入 vLLM 源码，建议按这个顺序：

1. **先读设计文档**
   - [vLLM V1 博客](https://vllm-project.github.io/2025/01/27/v1-alpha-release.html)
   - [Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
2. **读 Scheduler**
   - `vllm/v1/core/sched/scheduler.py`
   - 重点看 `schedule()` 的 running/waiting 两段、`allocate_slots()` 失败后的抢占、`update_from_output()` 的 stop/free 逻辑。
3. **读 KV Cache**
   - `vllm/v1/core/kv_cache_manager.py`
   - `vllm/v1/core/block_pool.py`
   - 重点看 `allocate_slots()`、`free()`、`get_computed_blocks()`、`cache_full_blocks()`。
4. **读 ModelRunner**
   - `vllm/v1/worker/gpu/model_runner.py`
   - 重点看 `prepare_inputs()`、`prepare_attn()`、`execute_model()`、`sample_tokens()`；先忽略 CUDA graph/TP/量化分支。
5. **读 Attention Backend**
   - `vllm/v1/attention/backend.py`
   - 选一个 backend（如 Triton/FlashAttention）看 paged KV 的读取方式。
6. **读 Sampler / OutputProcessor**
   - `vllm/v1/worker/gpu/sample/sampler.py`
   - `vllm/v1/engine/output_processor.py`
7. **读论文**
   - [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

阅读源码时建议带着三个问题：

- 这个模块的输入/输出契约是什么？
- 它维护了哪些跨 step 的状态？
- 如果去掉某个优化，最小的正确实现是什么？

---

## 9. 参考资料

### 官方文档 / 博客

- [vLLM V1: A Major Upgrade to vLLM’s Core Architecture](https://vllm-project.github.io/2025/01/27/v1-alpha-release.html)
- [Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [Paged Attention（历史文档，解释概念）](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [vLLM V1 User Guide](https://docs.vllm.ai/en/latest/usage/v1_guide/)
- [vLLM API Reference: vllm.v1](https://docs.vllm.ai/en/latest/api/vllm/v1/)

### 源码

- [Scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [KVCacheManager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)
- [BlockPool](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)
- [EngineCore](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py)
- [LLMEngine](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/llm_engine.py)
- [GPUModelRunner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py)
- [Attention Backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py)
- [Sampler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/sampler.py)
- [OutputProcessor](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/output_processor.py)

### 论文

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
