# 项目状态与任务认领

> 这里只记录“现在在哪、谁在做、最近改了什么”。
> 设计与里程碑以 [V1 设计方案](02-mini-inference-design-proposal.md) 为准，V0.1 基线见 [V0.1 设计与实现](03-v0.1-design-and-implementation.md)。

## 当前快照

- 基线：`main` @ `da0dde1`
- 版本：V0.1 已实现；V1 设计稿
- 当前里程碑：M1 MVP-A（范围与验收见 [02 §10](02-mini-inference-design-proposal.md)）
- 测试：`python3 -m unittest discover -v` -> 4 passed
- 最后更新：2026-11-21

## M1 任务

> 以下任务是实施拆分，不是原设计文档的既定编号；范围和验收以 `docs/02` 为准。

| ID | 任务 | 依赖 | 状态 |
|---|---|---|---|
| M1-01 | 批量契约与配置 | — | 待认领 |
| M1-02 | BlockPool + PagedKVCache | M1-01 | 待认领 |
| M1-03 | ToyTransformer full + paged | M1-02 | 待认领 |
| M1-04 | 批量 ModelRunner | M1-03 | 待认领 |
| M1-05 | Token-budget Scheduler | M1-04 | 待认领 |
| M1-06 | EngineCore 多请求生命周期 | M1-05 | 待认领 |
| M1-07 | Sampler greedy / temperature / top-p | M1-01 | 待认领 |
| M1-08 | CLI trace + 集成测试 | M1-06, M1-07 | 待认领 |

## 当前认领

| 任务 | 主机 / agent | 分支 | 状态 |
|---|---|---|---|
| — | — | — | — |

规则：开工前加一行并 push；完成后改为“已完成”；冲突时以先 push 且已开工的主机为准。

## 决策记录

| 日期 | 决策 | 原因 / 取舍 |
|---|---|---|
| 2026-11-14 | 先实现 V0.1，再接 V1 | 先固定模块边界，避免一次引入太多概念 |
| 2026-11-21 | 状态单独记账，设计归 `docs/02` | 避免 roadmap 与设计文档重复；多主机只需同步一份状态 |

待定问题（Q1–Q4）见 [02 §13](02-mini-inference-design-proposal.md)，确定后在此加一行结论。

## 变更日志

| 日期 | 主机 / agent | 变更 | 验证 |
|---|---|---|---|
| 2026-11-21 | seven-z790-eagle / opencode | 建立状态板；删除与 `docs/02` 重复的 roadmap 内容 | 文档链接检查通过；4 tests passed |
| 2026-11-14 | — | V0.1 基线 | commit `7bb88e7`；4 tests passed |
