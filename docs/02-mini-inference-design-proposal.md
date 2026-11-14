# 最小 Custom Inference 设计方案（讨论稿）

> 状态：**V1 设计讨论稿**。V0.1 已按“模块齐全、实现极简”的方式实现，见 [V0.1 设计与实现](03-v0.1-design-and-implementation.md)（代码在 [`v0_1/`](../v0_1/)）。  
> 目标：用尽可能小的代码量，把 vLLM V1 的核心执行流表达清楚；不是做一个能跑真实模型的推理框架。  
> 前置阅读：[vLLM 架构与核心模块调研](01-vllm-architecture-research.md)

---

## 0. 一页纸版本

**推荐方案：纯 Python + 随机权重 toy decoder，先做逻辑 batch，不做真实模型/GPU。**

最小闭环只有六个模块：

```text
Request  ->  Scheduler  ->  KVCacheManager/BlockPool
                |                    |
                v                    v
           SchedulerOutput ---> ModelRunner ---> Sampler
                ^                    |
                |                    v
           update_from_output <--- ModelRunnerOutput
```

三个最重要的契约：

1. `SchedulerOutput`：这一步每个请求算多少 token、从哪个位置开始、用哪些 block；
2. `ModelRunnerOutput`：这一步每个请求采样出的 token；
3. `Request` 状态：`num_computed_tokens`、`block_table`、`status`。

**MVP-A（建议先做）**：

- token-budget scheduler；
- running 优先 + waiting 接纳；
- paged KV block / block table；
- chunked prefill；
- `forward_full` vs `forward_paged` 正确性对照；
- greedy/temperature/top-p；
- CLI trace。

**MVP-A 不做**：preemption、prefix caching、真实模型、TP、HTTP server、CUDA、性能优化。  
**MVP-B（确认后再做）**：preemption / recompute，可选 prefix caching。

先确认第 13 节的 Q1（技术栈）、Q2（是否首版抢占）、Q3（是否首版 prefix caching）即可开始 M1。

> 快速评审路径：第 0 节 → 第 4 节（路线）→ 第 6.5 节（调度器）→ 第 7 节（例子）→ 第 10 节（里程碑）→ 第 13 节（问题）。其余章节是实现前的参考细节。

---

## 1. 目标

做一个可运行、可测试、可解释的 `mini-vLLM`，它能表达下面这条链路：

```text
add_request
  -> Scheduler 选择本步要算的 token
  -> KVCacheManager 分配 paged KV block
  -> ModelRunner 按 block table 做 paged forward
  -> Sampler 采样下一个 token
  -> Scheduler 更新请求状态 / 判断结束 / 释放 KV
  -> 输出 token
```

具体目标：

- 能同时处理多个请求，展示 **continuous batching**；
- 能把长 prompt 分块，展示 **chunked prefill**；
- 每个请求的 KV 存在非连续的物理 block 中，展示 **paged KV cache / block table**；
- 有明确的 `Request` 状态和 `SchedulerOutput` 契约；
- 用“全量重算”作为 oracle，验证 paged forward 的正确性；
- 纯标准库、CPU、无 GPU、无 PyTorch、无网络依赖；
- 代码量控制在约 500–700 行（核心实现，测试另计），每个模块只做一件事。

最终交付物（第一阶段）：

- 一个 Python package；
- 一个 CLI demo；
- 一组 `unittest` 测试；
- 一份 README，解释每个模块与 vLLM 的对应关系；
- 一份 step-by-step trace，能直观看到每个 step 的 batch 组成、token 预算、KV 使用。

---

## 2. 非目标

明确不做，避免第一版失控：

- 不加载真实 LLM 权重；
- 不实现 BPE/SentencePiece tokenizer；
- 不实现 CUDA / Triton kernel；
- 不实现 FlashAttention；
- 不实现 TP / PP / DP；
- 不实现 OpenAI API server；
- 不实现 streaming SSE / 增量 detokenizer；
- 不实现 prefix caching（第二阶段再讨论）；
- 不实现 speculative decoding；
- 不实现 structured output；
- 不实现 LoRA / multimodal；
- 不实现量化；
- 不实现生产级 metrics / tracing / fault tolerance；
- 不追求性能，不拿它和 vLLM 做 benchmark。

第一版的关键词是：**正确、透明、可验证**，不是“快”或“全”。

---

## 3. 设计原则

### 3.1 先控制流，后数据流

先让“请求如何被调度、KV 如何被分配、token 如何被更新”跑通，再考虑如何把多个请求拼成一个 tensor batch、如何调用高效 attention kernel。

第一版允许：

- 每个请求顺序 forward；
- Python list 存 KV；
- 显式 for 循环做 attention；
- 单进程、同步执行。

这些都会让代码“很慢”，但会让每一步都可断点、可打印、可比较。

### 3.2 每个 vLLM 概念都有一个最小对应物

不追求“像 vLLM 一样多”，而是追求“vLLM 的核心抽象在这里都能找到”：

| vLLM 概念 | mini 对应物 |
|---|---|
| `Request` | `Request` |
| `Scheduler` | `Scheduler` |
| `SchedulerOutput` | `SchedulerOutput` |
| `KVCacheManager` | `KVCacheManager` |
| `BlockPool` | `BlockPool` |
| `GPUModelRunner` | `ModelRunner` |
| `AttentionBackend` | `ToyTransformer.forward_paged()` |
| `Sampler` | `Sampler` |
| `EngineCore.step()` | `LLMEngine.step()` |
| `OutputProcessor` | `LLMEngine` 的输出组装（简化） |
| `Worker` / `Executor` | 省略，直接调用 `ModelRunner` |

### 3.3 契约小而清晰

模块之间尽量只通过 dataclass 传递数据：

- `SchedulerOutput`：调度器告诉执行层“这一步每个请求算多少 token、用哪些 block”；
- `ModelRunnerOutput`：执行层告诉调度器“这一步每个请求采样出了什么 token”；
- `EngineCoreOutput`：调度器告诉前端“这个请求新增了哪些 token、是否结束”。

只要这三个契约稳定，内部实现可以逐步替换。

### 3.4 用 oracle 验证，而不是靠“输出看起来对”

Toy model 的随机权重输出是乱码，不能靠肉眼判断对错。必须有一个不依赖 KV cache 的 `forward_full()`，对同一段 token 计算 logits，和 `forward_paged()` 对比。

### 3.5 不做“提前抽象”

第一版不需要：

- `AttentionBackend` 抽象基类；
- `Executor` 工厂；
- 多态 `Model` 注册表；
- 插件系统；
- 配置系统。

用具体类、直接调用、显式分支，反而更容易看懂。等第二阶段换 PyTorch/真实模型时再抽象。

---

## 4. 技术路线选择

### 4.1 三个候选方案

| 方案 | 内容 | 优点 | 缺点 | 适合阶段 |
|---|---|---|---|---|
| A. 纯 Python + toy decoder | 标准库、随机权重、char tokenizer、手写 paged attention | 零依赖、逻辑最透明、容易验证 | 输出无意义、性能很差 | 第一阶段（推荐） |
| B. PyTorch + tiny GPT | PyTorch、tiny GPT/nanoGPT、真实 tensor batch | 更接近真实模型、可训练/可加载小权重 | 需要处理 tensor shape、mask、设备；控制流容易被框架细节淹没 | 第二阶段 |
| C. HF 小模型 + 自定义 paged attention | Qwen2.5-0.5B / Llama 3.2 1B 等 | 真实 tokenizer/权重、能看真实输出 | 工程量大；attention/kv layout/权重加载复杂 | 第三阶段 |

### 4.2 推荐路线

**A → B → C，分三步，不要合并。**

- **A（现在）**：纯 Python，先把 scheduler / KV manager / model runner / sampler / engine 的契约跑通。
- **B（下一步）**：保留 scheduler / KV manager 的接口，把 `ToyTransformer` 换成 PyTorch 的 tiny GPT；把“每个请求顺序 forward”换成“扁平 token batch + `query_start_loc` / `block_tables`”。
- **C（再下一步）**：加载真实小模型，替换 tokenizer 和 attention 实现；这时才需要考虑 RoPE、GQA、dtype、显存、kernel。

这样每一步只换一层，不会同时面对“调度没懂 + tensor 不会写 + 模型加载失败”三个问题。

---

## 5. 总体架构

```text
┌─────────────────────────────────────────────────────────────┐
│                         LLMEngine                           │
│                                                             │
│  add_request(prompt, params)                                │
│    └── tokenize -> Request -> Scheduler.add_request()       │
│                                                             │
│  step()                                                     │
│    ├── Scheduler.schedule() -> SchedulerOutput              │
│    ├── ModelRunner.execute(SchedulerOutput)                 │
│    │     ├── for each scheduled request:                    │
│    │     │     ToyTransformer.forward_paged(...)            │
│    │     │     Sampler.sample(logits)                       │
│    │     └── ModelRunnerOutput                              │
│    ├── Scheduler.update_from_output(...)                    │
│    │     ├── num_computed_tokens += n                       │
│    │     ├── append sampled token                           │
│    │     ├── stop / length check                            │
│    │     └── free finished KV blocks                        │
│    └── output tokens / finished requests                    │
│                                                             │
│  generate(prompts)                                          │
│    └── add all -> loop step() until no unfinished requests  │
└─────────────────────────────────────────────────────────────┘
          │                                   ▲
          ▼                                   │
┌──────────────────────┐          ┌──────────────────────────┐
│ Scheduler            │          │ KVCacheManager           │
│ - waiting queue      │          │ - block table per request│
│ - running list       │◄────────►│ - allocate / free        │
│ - token budget       │          │ - BlockPool              │
│ - max_num_seqs       │          │ - PagedKVCache (storage) │
└──────────────────────┘          └──────────────────────────┘
```

注意：`PagedKVCache` 是实际 K/V 存储，`KVCacheManager` 只管理 block 元数据。这个分离对应 vLLM 的 `BlockPool` / GPU KV tensors 与 `KVCacheManager` 的职责划分。

---

## 6. 模块设计

### 6.1 `config.py`

只放配置，不放逻辑。

```text
ModelConfig
  vocab_size
  hidden_size
  num_layers
  num_heads
  head_dim
  mlp_hidden_size
  max_model_len
  seed

CacheConfig
  block_size
  num_gpu_blocks

SchedulerConfig
  max_num_batched_tokens
  max_num_seqs

EngineConfig
  model: ModelConfig
  cache: CacheConfig
  scheduler: SchedulerConfig
```

第一版不需要 `VllmConfig` 那种大而全的配置对象；三个小配置足够了。

### 6.2 `request.py`

```text
RequestStatus:
  WAITING
  RUNNING
  FINISHED_STOPPED
  FINISHED_LENGTH_CAPPED
  FINISHED_ABORTED
  (PREEMPTED，第二阶段)

SamplingParams:
  max_tokens
  temperature
  top_p
  stop_token_ids
  seed

Request:
  request_id
  prompt_token_ids
  output_token_ids
  num_computed_tokens
  status
  block_table
  sampling_params
  arrival_time
  finish_reason
  num_preemptions

  properties:
    all_token_ids = prompt + output
    num_tokens = len(all_token_ids)
    num_uncomputed_tokens = num_tokens - num_computed_tokens
    is_finished
```

关键 invariant：

```text
0 <= num_computed_tokens <= num_tokens
```

`num_computed_tokens` 是调度器和 model runner 之间最重要的共享状态。

### 6.3 `tokenizer.py`

Char-level tokenizer：

- `encode(text) -> list[int]`
- `decode(ids) -> str`
- 固定字符表，未知字符映射到空格；
- 空 prompt 映射到一个空格 token；
- 不处理 BPE、UTF-8 多字节边界。

为什么不用 HF tokenizer：

- 第一版不需要真实文本质量；
- 少一个依赖；
- 字符级 token 让 token 数量和文本长度直观对应，方便 trace。

### 6.4 `kv_cache.py`

包含三部分：

#### `Block`

```text
Block:
  block_id
  ref_count
```

第一版没有 prefix caching，所以一个 block 被分配给一个请求时 `ref_count=1`，释放后为 0。保留 `ref_count` 是为了以后加 prefix caching 时不用重写 BlockPool。

#### `BlockPool`

```text
BlockPool:
  blocks: list[Block]
  free_blocks: deque[int]

  allocate(n) -> list[int] | None
  free(block_ids) -> None
  num_free_blocks
  usage
```

`allocate` 在 free block 不足时返回 `None`，不能部分分配后失败。

#### `KVCacheManager`

```text
KVCacheManager:
  block_size
  block_pool
  block_tables: dict[request_id, list[physical_block_id]]

  allocate_slots(request, num_new_tokens) -> bool
  free(request) -> None
  get_block_table(request_id) -> tuple[int, ...]
  usage
```

`allocate_slots` 的逻辑：

```text
start = request.num_computed_tokens
end = start + num_new_tokens
needed_blocks = ceil(end / block_size)
current_blocks = len(request.block_table)

if needed_blocks <= current_blocks:
    return True

new_blocks = block_pool.allocate(needed_blocks - current_blocks)
if new_blocks is None:
    return False

request.block_table.extend(new_blocks)
return True
```

注意：这里分配的是“能覆盖到 end 的 block 数”，不是“num_new_tokens 个 block”。一个 block 可以容纳多个 token。

#### `PagedKVCache`

实际 K/V 存储：

```text
shape = [num_layers][num_blocks][block_size][num_heads][head_dim]
```

为了让代码简单，可以用扁平 `list[float]` 存储，并通过索引函数访问：

```text
offset(layer, block, slot, head) =
  (((layer * num_blocks + block) * block_size + slot) * num_heads + head) * head_dim
```

token 位置 `p` 的地址：

```text
block_id = block_table[p // block_size]
slot     = p % block_size
```

接口：

```text
write(layer, block_id, slot, head, k_vec, v_vec)
read_k(layer, block_id, slot, head) -> list[float]
read_v(layer, block_id, slot, head) -> list[float]
```

### 6.5 `scheduler.py`

```text
Scheduler:
  waiting: deque[Request]
  running: list[Request]
  kv_cache_manager
  max_num_batched_tokens
  max_num_seqs

  add_request(request)
  schedule() -> SchedulerOutput
  update_from_output(scheduler_output, model_output) -> list[EngineCoreOutput]
  has_requests()
```

#### MVP-A 的简化

第一版**不实现 preemption**，但通过配置保证不会出现“永远分配不到 block”的死锁：

```text
num_gpu_blocks >= max_num_seqs * ceil(max_model_len / block_size)
max_num_batched_tokens <= num_gpu_blocks * block_size
prompt_len < max_model_len
```

这三条是配置/输入校验，不是调度算法本身。它们保证 MVP-A 不会因为 KV 容量不足而卡死。

这只是容量上的保证；调度器仍然是按需分配，不是每个请求预先占满。这样既能展示 block table / 动态分配，又能暂时跳过抢占逻辑。

如果 `allocate_slots` 仍然失败，直接抛出带诊断信息的异常（说明配置或 invariant 有问题），不要静默失败。

#### `schedule()` 伪代码

```text
def schedule():
    token_budget = max_num_batched_tokens
    scheduled = []

    # 1. running 优先
    for request in running:
        if token_budget == 0:
            break

        n = min(request.num_uncomputed_tokens, token_budget)
        if n == 0:
            continue

        if not kv_cache_manager.allocate_slots(request, n):
            raise KVExhausted(...)  # MVP-A：配置应保证不会发生

        start = request.num_computed_tokens
        scheduled.append(ScheduledRequest(
            request=request,
            start_pos=start,
            num_new_tokens=n,
            token_ids=request.all_token_ids[start:start+n],
            block_table=tuple(request.block_table),
            needs_logits=(start + n == request.num_tokens),
        ))
        token_budget -= n

    # 2. waiting 接纳
    while (token_budget > 0
           and len(running) < max_num_seqs
           and waiting):
        request = waiting[0]
        n = min(request.num_uncomputed_tokens, token_budget)

        if not kv_cache_manager.allocate_slots(request, n):
            break  # 先不抢占，等 running 释放

        waiting.popleft()
        request.status = RUNNING
        running.append(request)
        scheduled.append(...)
        token_budget -= n

    return SchedulerOutput(
        scheduled=scheduled,
        total_num_scheduled_tokens=sum(x.num_new_tokens for x in scheduled),
    )
```

#### `needs_logits` 为什么重要

对于 chunked prefill，一个请求可能这步只算 prompt 的前半段。此时它还不应该采样下一个 token，因为最后一个已计算位置并不是“当前序列的末尾”。

```text
needs_logits = (start_pos + num_new_tokens == len(all_token_ids))
```

- 如果为 `False`：只算 KV，不采样；
- 如果为 `True`：算完最后一个 token 的 KV，取它的 logits，采样下一个 token。

这个布尔值替代了 vLLM 里 `logits_indices` 的一部分语义，是第一版最容易讲清楚的关键点。

#### `update_from_output()` 伪代码

```text
def update_from_output(scheduler_output, model_output):
    outputs = []

    for item in scheduler_output.scheduled:
        request = item.request

        # 1. 推进已计算 token 数
        #    Mini 选择“执行成功后再推进”，比 vLLM 的乐观推进更直观。
        request.num_computed_tokens += item.num_new_tokens

        # 2. 追加采样结果
        new_tokens = model_output.sampled_token_ids.get(request.request_id, [])
        for token in new_tokens:
            request.output_token_ids.append(token)

        # 3. 判断结束
        finished = False
        finish_reason = None

        if new_tokens:
            last = new_tokens[-1]
            if last in request.sampling_params.stop_token_ids:
                finished = True
                finish_reason = "stop"
            elif len(request.output_token_ids) >= request.sampling_params.max_tokens:
                finished = True
                finish_reason = "length"
            elif len(request.all_token_ids) >= max_model_len:
                finished = True
                finish_reason = "length"

        # 4. 如果 needs_logits 却没有采样结果，说明契约被破坏
        if item.needs_logits and not new_tokens:
            raise RuntimeError("missing sampled token")

        # 5. 结束则释放 KV
        if finished:
            request.finish_reason = finish_reason
            request.status = FINISHED_*
            kv_cache_manager.free(request)
            running.remove(request)

        outputs.append(EngineCoreOutput(
            request_id=request.request_id,
            new_token_ids=new_tokens,
            output_token_ids=list(request.output_token_ids),
            finished=finished,
            finish_reason=finish_reason,
        ))

    return outputs
```

> **与 vLLM 的差异**：vLLM 在 `_update_after_schedule()` 中先乐观推进 `num_computed_tokens`，这样下一步调度可以立即看到进度；speculative token 被拒绝时再回滚。Mini 版在模型执行成功后推进，语义更直观，也避免了“执行失败但状态已前进”的问题。两者在一步结束时的 invariant 相同。

### 6.6 `model.py`

Toy decoder 结构：

```text
Token embedding + Position embedding
  -> N × DecoderLayer
       - LayerNorm -> QKV -> causal multi-head attention -> residual
       - LayerNorm -> MLP -> residual
  -> final LayerNorm
  -> lm_head（与 token embedding 绑定）
```

推荐配置：

```text
vocab_size   = 字符表大小（约 70）
hidden_size  = 32
num_layers   = 2
num_heads    = 4
head_dim     = 8
mlp_hidden   = 64
max_model_len= 64~128
```

权重用固定 seed 的 `random.Random` 初始化。输出是乱码，这是预期行为；模型的作用是提供可重复的 logits，用来验证 paged attention 和调度流。

提供两个 forward：

```text
forward_full(token_ids) -> logits
  不读 KV cache，从头算所有位置
  用作 correctness oracle

forward_paged(token_ids, start_pos, block_table, kv_cache, need_logits)
  逐层：
    - 计算新 token 的 Q/K/V
    - 把 K/V 写入 block table 指定的物理 slot
    - 每个新 token 对 0..position 的 K/V 做 attention
    - 残差 + MLP
  如果 need_logits，返回最后一个位置的 logits
```

`forward_paged` 的伪代码见第 8.4 节。

### 6.7 `model_runner.py`

```text
ModelRunner:
  model
  kv_cache
  sampler

  execute(scheduler_output) -> ModelRunnerOutput
```

逻辑：

```text
for item in scheduler_output.scheduled:
    logits = model.forward_paged(
        token_ids=item.token_ids,
        start_pos=item.start_pos,
        block_table=item.block_table,
        kv_cache=self.kv_cache,
        need_logits=item.needs_logits,
    )

    if item.needs_logits:
        token = sampler.sample(logits, item.request.sampling_params, item.request_id)
        sampled[item.request_id] = [token]

return ModelRunnerOutput(sampled_token_ids=sampled)
```

第一版这里是**逻辑 batch**：多个请求在同一个 `SchedulerOutput` 里，但 forward 是逐个请求顺序执行的。它仍然展示了 continuous batching 的调度语义；真正的 tensor batch 留到第二阶段。

### 6.8 `sampler.py`

```text
Sampler:
  sample(logits, sampling_params, request_id) -> token_id
```

第一版支持：

- `temperature == 0`：argmax（greedy）；
- `temperature > 0`：softmax(logits / T)；
- `top_p < 1`：nucleus sampling；
- 每个 request 一个 RNG，seed 可复现。

不处理：

- top-k；
- repetition penalty；
- logit bias；
- logprobs；
- structured output mask。

### 6.9 `engine.py`

```text
LLMEngine:
  tokenizer
  config
  scheduler
  kv_cache_manager
  model_runner

  add_request(prompt, sampling_params) -> request_id
  step() -> list[StepOutput]
  generate(prompts, sampling_params) -> list[RequestOutput]
  has_unfinished_requests()
```

`step()`：

```text
scheduler_output = scheduler.schedule()
if scheduler_output.is_empty:
    return []

model_output = model_runner.execute(scheduler_output)
core_outputs = scheduler.update_from_output(scheduler_output, model_output)

for core_output in core_outputs:
    text = tokenizer.decode(core_output.output_token_ids)
    ...
```

`generate()`：

```text
request_ids = [add_request(p) for p in prompts]

while has_unfinished_requests():
    step()

return [final_output[rid] for rid in request_ids]
```

第一版不需要 `OutputProcessor` 类；`LLMEngine` 直接组装 `StepOutput` / `RequestOutput` 即可。但要在文档中说明：真实 vLLM 的增量 detokenize、stop string、streaming 都在 `OutputProcessor` 里。

### 6.10 `outputs.py`

建议的数据契约：

```text
ScheduledRequest:
  request
  start_pos
  num_new_tokens
  token_ids
  block_table
  needs_logits

SchedulerOutput:
  scheduled: list[ScheduledRequest]
  total_num_scheduled_tokens: int
  preempted_request_ids: list[str]   # MVP-A 恒为空

ModelRunnerOutput:
  sampled_token_ids: dict[request_id, list[int]]
  num_forwarded_tokens: int

EngineCoreOutput:
  request_id
  new_token_ids
  output_token_ids
  finished
  finish_reason

StepOutput:
  request_id
  new_token_ids
  output_token_ids
  text
  finished
  finish_reason

RequestOutput:
  request_id
  prompt
  prompt_token_ids
  output_token_ids
  text
  finish_reason
```

---

## 7. 一个具体的调度例子

配置：

```text
block_size = 4
max_num_batched_tokens = 8
max_num_seqs = 4
```

当前状态：

| 请求 | 状态 | prompt | output | num_computed | 还需要算 | block table |
|---|---|---:|---:|---:|---:|---|
| A | running | 6 | 0 | 4 | 2（prompt 的后半段） | [7] |
| B | running | 5 | 1 | 5 | 1（decode） | [3, 1] |
| C | waiting | 5 | 0 | 0 | 5（prefill） | [] |

这一步：

```text
token_budget = 8

A: n = min(2, 8) = 2   -> 预算 6，需要新 block（位置 4,5）
B: n = min(1, 6) = 1   -> 预算 5，位置 5 可能已在已有 block 内
C: n = min(5, 5) = 5   -> 预算 0，需要 2 个 block

scheduled = [A:+2, B:+1, C:+5]
```

这是一个 **prefill 收尾 + decode + 新请求 prefill** 混合的 batch。三者的 `needs_logits` 都是 `True`，所以执行后各采样一个 token：

- A 完成 prompt，采样第一个 output token；
- B 采样下一个 output token；
- C 完成 prompt，采样第一个 output token。

如果预算只有 4：

```text
A: n = 2 -> 预算 2
B: n = 1 -> 预算 1
C: n = 1 -> 预算 0，chunked prefill，只算 C 的第 1 个 token

C.needs_logits = False（1 != 5），所以 C 这一步不采样；
下一步继续 C 的剩余 4 个 token。
```

这个例子把 V1 的三个核心点都串起来了：

1. 统一 token 预算；
2. prefill/decode 混合；
3. chunked prefill 自然产生。

---

## 8. 执行流与伪代码

### 8.1 `add_request`

```text
engine.add_request(prompt, params):
    token_ids = tokenizer.encode(prompt)
    validate(token_ids, params)
    request = Request(
        request_id=next_id(),
        prompt_token_ids=token_ids,
        sampling_params=params,
        status=WAITING,
    )
    scheduler.add_request(request)
    return request.request_id
```

### 8.2 `step`

```text
engine.step():
    scheduler_output = scheduler.schedule()

    if scheduler_output.is_empty:
        if scheduler.has_requests():
            raise RuntimeError("no progress: KV cache too small or scheduler bug")
        return []

    model_output = model_runner.execute(scheduler_output)
    core_outputs = scheduler.update_from_output(scheduler_output, model_output)

    step_outputs = []
    for out in core_outputs:
        if out.new_token_ids or out.finished:
            step_outputs.append(make_step_output(out))
    return step_outputs
```

### 8.3 `schedule`

见 6.5 节。第一版不包含抢占，running 循环里分配失败直接报错。

### 8.4 paged forward

对每个请求：

```text
forward_paged(token_ids, start_pos, block_table, kv_cache, need_logits):
    n = len(token_ids)
    x = [embed(token_ids[i], position=start_pos+i) for i in range(n)]

    for layer in layers:
        q, k, v = qkv(layer.norm(x))

        # 1. 先把本步所有新 token 的 K/V 写入物理 cache
        for i in range(n):
            p = start_pos + i
            block = block_table[p // block_size]
            slot  = p % block_size
            write_kv(layer, block, slot, k[i], v[i])

        # 2. 每个新 token 对 [0, p] 的 K/V 做 attention
        for i in range(n):
            p = start_pos + i
            prefix_k = []
            prefix_v = []
            for j in range(p + 1):
                block = block_table[j // block_size]
                slot  = j % block_size
                prefix_k.append(read_k(layer, block, slot))
                prefix_v.append(read_v(layer, block, slot))

            attn = causal_attention(q[i], prefix_k, prefix_v)
            x[i] = residual + out_proj(attn)
            x[i] = residual + mlp(norm(x[i]))

    if not need_logits:
        return None

    return lm_head(norm(x[-1]))
```

注意两个细节：

- 先把本步所有 K/V 写入，再算 attention；对位置 `i` 的 query，只允许看 `0..p`，所以同 chunk 里更靠后的 K/V 不会影响它。
- 对 chunked prefill，`start_pos > 0` 时，`0..start_pos-1` 的 K/V 已经在 cache 里，直接按 block table 读。

### 8.5 更新与结束

见 6.5 节的 `update_from_output`。

结束条件：

- 采样到 stop token；
- output 长度达到 `max_tokens`；
- prompt + output 达到 `max_model_len`；
- 显式 abort（第一版可以不做）。

结束后：

- 从 `running` 移除；
- `KVCacheManager.free(request)`；
- 保留最终 output 供 `generate()` 返回；
- 不再参与后续调度。

---

## 9. 正确性验证

### 9.1 Oracle：`forward_full` vs `forward_paged`

测试 1：单 chunk

```text
ids = tokenizer.encode("hello world")
full_logits = model.forward_full(ids)
paged_logits = model.forward_paged(
    ids, start_pos=0, block_table=..., need_logits=True
)
assert max_abs_diff(full_logits, paged_logits) < 1e-9
```

测试 2：多 chunk

```text
先 paged 算 ids[:3]（need_logits=False）
再 paged 算 ids[3:]（need_logits=True）
和 forward_full(ids) 比较
```

测试 3：碎片化 block table

```text
先给别的请求分配几个 block，让当前请求的物理 block 不连续
再比较 paged 和 full 的 logits
```

测试 4：逐 token decode

```text
对 ids 的每个前缀，逐步 paged 算 1 个 token
与 forward_full(prefix) 的最后一个 logits 比较
```

这四类测试能覆盖：

- block table 地址计算；
- prefix KV 读取；
- chunked prefill；
- decode 路径；
- 非连续物理 block。

### 9.2 Greedy 等价

用同一个模型、同一个 prompt：

```text
paged_engine 生成 T 个 token
reference: 每步 forward_full(all_token_ids) + argmax，生成 T 个 token
assert 两者 token 序列完全一致
```

这比只比较单步 logits 更接近真实生成路径。

### 9.3 调度器测试

- token budget 不会被超过；
- running 优先；
- waiting 在预算/槽位允许时被接纳；
- chunked prefill 的 `needs_logits` 正确；
- `max_num_seqs` 不被超过；
- 请求完成后从 running 移除，block 被释放；
- 所有请求最终都能结束（无死锁）。

### 9.4 连续 batching 测试

- 先加入 A，step 若干次；
- 在 A 还在 decode 时加入 B；
- 后续某个 step 的 `SchedulerOutput.scheduled` 同时包含 A 和 B；
- 两者都继续产出 token，最终都结束。

### 9.5 配置/边界测试

- block pool 分配/释放/耗尽；
- prompt 长度接近 `max_model_len`；
- `max_tokens=1`；
- stop token 在第一个/最后一个位置；
- 空 prompt（映射到一个空格 token）；
- 重复 request_id（应报错）。

### 9.6 测试风格

- 标准库 `unittest`，不引入 pytest；
- 测试文件按模块拆分：`test_kv_cache.py`、`test_scheduler.py`、`test_paged_attention.py`、`test_engine.py`；
- 每个测试尽量小、可读，断言要能指出“哪个 invariant 被破坏”；
- 不测性能，不测随机权重的文本质量。

---

## 10. 里程碑与验收

### M0：调研 + 设计 + V0.1（当前）

交付：

- vLLM 架构调研文档；
- 本设计文档；
- 范围、取舍、测试计划。

验收：

- 能清楚解释 Scheduler / KV Cache / ModelRunner / Sampler 的职责；
- 能解释 `num_scheduled_tokens`、`num_computed_tokens`、block table 的关系；
- 对第一版范围达成一致。

### M1：MVP-A（核心骨架）

范围：

- Request / SamplingParams；
- BlockPool / KVCacheManager / PagedKVCache；
- Scheduler（无抢占）；
- ToyTransformer（full + paged）；
- ModelRunner；
- Sampler（greedy + temperature + top-p）；
- LLMEngine（add_request / step / generate）；
- CLI demo + trace；
- 测试。

验收：

- `python3 -m unittest discover` 全绿；
- demo 能同时生成多个请求；
- trace 中能看到 mixed batch 和 chunked prefill；
- paged logits 与 full logits 在容差内一致；
- greedy 生成与全量重算 reference 完全一致；
- 没有第三方依赖。

### M2：MVP-B（资源压力）

范围（可拆成两个小步）：

- M2a：Preemption / recompute；
- M2b：Prefix caching（可选）。

验收：

- 缩小 block pool 后，请求仍能全部完成；
- 抢占次数 > 0，且没有死锁；
- prefix caching 命中时，block 引用计数正确，输出与不共享时一致。

### M3：PyTorch tiny GPT

范围：

- 用 PyTorch 重写 `ToyTransformer`，保留 `forward_full` oracle；
- 保持 scheduler / KV manager 接口不变；
- 把 model runner 改成扁平 token batch（`query_start_loc`、`block_tables`）；
- 可用 `torch` 的 SDPA 或手写 attention 做 paged gather。

验收：

- 同一 prompt 下，PyTorch 版 greedy 输出与 Python 版一致（或与 full 版一致）；
- scheduler 测试不变；
- 能解释“逻辑 batch”和“tensor batch”的差异。

### M4：真实小模型

范围：

- 加载 HF tokenizer + 小模型（如 Qwen2.5-0.5B）；
- 处理 RoPE、GQA、dtype、chat template；
- 替换 attention backend；
- 加基本 metrics（TTFT、TPOT、吞吐、KV 使用率）。

验收：

- 能生成可读文本；
- 与 HF `generate()` 在 greedy 下接近（允许数值差异）；
- 能解释真实模型带来的额外复杂度。

### M5：serving / 并行（可选）

- FastAPI/OpenAI 兼容接口；
- streaming；
- TP/DP；
- 性能优化。

第一版不碰。

---

## 11. 计划中的目录结构

V1 沿用 V0.1 的“按功能层分目录”规范：

```text
oh_my_inference/
├── README.md
├── docs/
│   ├── 01-vllm-architecture-research.md
│   └── 02-mini-inference-design-proposal.md
├── v0_1/                       # 已实现：单请求、连续 KV cache
├── mini_vllm/                  # V1 计划
│   ├── __init__.py
│   ├── serving/
│   │   └── server.py
│   ├── engine/
│   │   ├── core.py
│   │   ├── scheduler.py
│   │   ├── kv_cache.py
│   │   ├── block_pool.py
│   │   ├── model_runner.py
│   │   ├── tokenizer.py
│   │   └── types.py
│   ├── model/
│   │   ├── toy_transformer.py
│   │   ├── paged_attention.py
│   │   └── sampler.py
│   └── config.py
├── tests/
│   ├── test_kv_cache.py
│   ├── test_scheduler.py
│   ├── test_paged_attention.py
│   └── test_engine.py
└── examples/
    ├── demo.py
    └── continuous_batching.py
```

当前已经实现 `v0_1/`（单请求、连续 KV cache、无 paged block 的 V0.1）；上面的 `mini_vllm/` 是 V1 的计划结构，尚未创建。V1 实现时会沿用同样的功能层边界；不会为每个文件单独建目录。

---

## 12. 风险与取舍

| 风险 | 影响 | 缓解 |
|---|---|---|
| Toy model 输出乱码 | 看起来“不像 LLM” | 明确这是 correctness-first；用 trace 和 logits 对比展示价值；真实模型放 M3/M4 |
| 纯 Python 太慢 | 不能做性能实验 | 第一版不 benchmark；限制 prompt/生成长度；后续换 PyTorch |
| Paged attention 地址计算易错 | logits 不对 | `forward_full` oracle + 碎片化 block 测试 |
| Scheduler 边界多 | 死锁/重复调度 | 先做无抢占版本；用 invariant 测试；容量保证 |
| 过早抽象 | 代码难懂 | 第一版不用 ABC/工厂/插件；具体类直接调用 |
| 范围膨胀 | 迟迟跑不起来 | 明确非目标；M1 只做最小闭环 |
| 接口反复改 | 浪费时间 | 先评审 `SchedulerOutput` / `ModelRunnerOutput` / `Request` 三个契约 |

---

## 13. 需要一起确认的问题

### Q1. 首版技术栈

- **推荐：纯 Python + toy model**。零依赖，控制流最清楚，适合建立认知。
- 备选：直接 PyTorch + tiny GPT。更接近真实模型，但会同时引入 tensor shape、mask、设备管理等问题。

你更希望“先看懂”，还是“先跑真模型”？

### Q2. 首版是否包含 preemption

- **推荐：MVP-A 不包含**，用容量保证避免 KV 不足；MVP-B 再加抢占/重算。
- 备选：第一版就包含。更完整，但调度器和测试会复杂不少。

### Q3. 首版是否包含 prefix caching

- **推荐：暂缓**。它需要 block hash、引用计数、COW/partial block、LRU 驱逐，容易把第一版拖复杂。
- 备选：如果目标就是理解 vLLM 的 prefix caching，可以单独做一个小阶段。

### Q4. Toy model 的形态

- **推荐：随机权重、2 层 decoder、char tokenizer**。输出无意义，但可复现、可验证。
- 备选：训练一个很小的 char-level 模型，让输出稍微像样；但会增加训练脚本和权重文件。
- 备选：直接上 HF 小模型；这会进入 M3/M4 的范围。

### Q5. 交付形态

- **推荐：package + CLI + unittest + trace**。适合逐步扩展，也方便你改代码做实验。
- 备选：Jupyter notebook，按 step 可视化；更适合讲解，但不利于长期演进。

### Q6. 代码与文档语言

- **推荐：文档/注释中文，代码标识符英文**。既方便理解，又保持代码可读性。
- 如果你习惯全英文，也可以调整。

---

## 14. 建议的讨论顺序

1. 先确认 **Q1（技术栈）** 和 **Q2（是否首版抢占）**，这两个决定 M1 的范围。
2. 再确认 **Q3（prefix caching）** 是否进入 M1；我建议不进入。
3. 最后确认 **Q5（交付形态）**，然后开始写 M1。
4. M1 完成后再讨论 M2/M3，不要现在就把真实模型、TP、serving 全排进去。

如果按推荐选项，M1 的边界就是：

```text
纯 Python + toy model
+ token-budget scheduler（无抢占）
+ paged KV / block table
+ logical batch 的 paged forward
+ greedy/temperature/top-p sampler
+ engine step loop
+ CLI trace
+ full-vs-paged correctness tests
```

这是一个**能讲清楚 vLLM 执行流、又不会陷入工程细节**的最小闭环。
