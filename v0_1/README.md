# V0.1 快速开始

V0.1 是“模块齐全、实现极简”的 inference engine：

```text
Server
  -> EngineCore
       -> Scheduler
       -> KVCacheManager
       -> ModelRunner -> TinyAttentionModel + Sampler
```

完整设计、模块职责、执行流和验证记录见：

- [V0.1 设计与实现](../docs/03-v0.1-design-and-implementation.md)
- [V1 调研与设计方案](../docs/02-mini-inference-design-proposal.md)

## 目录

```text
v0_1/
├── serving/
│   └── server.py              # HTTP 入口
├── engine/
│   ├── core.py                # EngineCore: schedule -> execute -> update
│   ├── scheduler.py           # 请求调度、prefill/decode、结束判断
│   ├── kv_cache.py            # KVCacheManager：分配/获取/释放
│   ├── model_runner.py        # 执行一个 SchedulerOutput
│   ├── tokenizer.py           # 文本 <-> token id
│   └── types.py               # Request / 输出契约
├── model/
│   ├── attention.py           # 单层单 head toy model
│   ├── cache.py               # 连续 K/V 存储
│   └── sampler.py             # logits -> token
├── examples/
│   └── prefill_decode_kv_cache_trace.py   # 本地实验脚本
└── tests/
    └── test_v0_1.py
```

## 运行

当前环境只有 `python3`，没有 `python` 别名：

```bash
# 实验
python3 -m v0_1.examples.prefill_decode_kv_cache_trace --max-tokens 6 --temperature 0.8

# 测试
python3 -m unittest v0_1.tests.test_v0_1 -v

# server
python3 -m v0_1.serving.server --port 8000
```

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"hello","max_tokens":6,"temperature":0.8,"stop_chars":""}'
```

输出文本是随机权重产生的乱码；不同 prompt 可能收敛到相似/重复的 token，这是 toy model 的预期行为，不是 KV cache 的 bug。请看 `phase`、`computed`、`cached` 三列。

## 从请求入口开始

1. [HTTP 入口：do_POST()](serving/server.py#L51)
2. [EngineCore.generate()](engine/core.py#L120) → [add_request()](engine/core.py#L51)
3. [EngineCore.step()](engine/core.py#L79)
4. [Scheduler.schedule()](engine/scheduler.py#L70) → [ModelRunner.execute()](engine/model_runner.py#L24)
5. [TinyAttentionModel._step()](model/attention.py#L72)

更完整的解释在 [V0.1 设计文档](../docs/03-v0.1-design-and-implementation.md)。
