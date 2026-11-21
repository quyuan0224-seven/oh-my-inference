# 项目状态与任务认领

> 这里只记录“现在在哪、谁在做、最近改了什么”。
> 设计与里程碑以 [V1 设计方案](02-mini-inference-design-proposal.md) 为准，当前基线见 [mini_vllm 设计与实现](03-mini-vllm-design-and-implementation.md)。

## 当前快照

- 基线：`main` 当前 HEAD（见 `git log -1`）
- 版本：mini_vllm 当前基线；V1 开发中
- 当前里程碑：M1 MVP-A（范围与验收见 [02 §10](02-mini-inference-design-proposal.md)）
- 测试：`python3 -m unittest discover -v` -> 4 passed
- 最后更新：2026-11-21

## M1 小步计划

> 以下任务是实施拆分，不是原设计文档的既定编号；范围和验收以 `docs/02` 为准。

| ID | 小步 | 依赖 | 状态 |
|---|---|---|---|
| S0 | 重命名 `v0_1` -> `mini_vllm` | — | ✅ 已完成 |
| S1 | 批量契约重构 | S0 | 进行中 |
| S2 | 多请求 logical batch | S1 | 待开始 |
| S3 | chunked prefill | S2 | 待开始 |
| S4 | token budget | S3 | 待开始 |
| S5 | BlockPool | S4 | 待开始 |
| S6 | PagedKVCache 存储 | S5 | 待开始 |
| S7 | `forward_paged` + oracle | S6 | 待开始 |
| S8 | 引擎切换 paged KV | S7 | 待开始 |
| S9 | 多头 / 多层模型 | S8 | 待开始 |
| S10 | top-p 采样 | S1 | 待开始 |
| S11 | CLI trace + 集成测试 | S9, S10 | 待开始 |

## 当前认领

| 任务 | 主机 / agent | 分支 | 状态 |
|---|---|---|---|
| S0–S1 | seven-z790-eagle / opencode | `main` | 进行中 |

规则：开工前加一行并 push；完成后改为“已完成”；冲突时以先 push 且已开工的主机为准。

## 决策记录

| 日期 | 决策 | 原因 / 取舍 |
|---|---|---|
| 2026-11-14 | 先实现 mini_vllm 初版，再接 V1 | 先固定模块边界，避免一次引入太多概念 |
| 2026-11-21 | 状态单独记账，设计归 `docs/02` | 避免状态文档与设计文档重复；多主机只需同步一份状态 |
| 2026-11-21 | 直接把 `v0_1` 重命名为 `mini_vllm`，不保留旧命名 | 避免双包维护；后续直接在 `mini_vllm/` 上演进 |

待定问题（Q1–Q4）见 [02 §13](02-mini-inference-design-proposal.md)，确定后在此加一行结论。

## 变更日志

| 日期 | 主机 / agent | 变更 | 验证 |
|---|---|---|---|
| 2026-11-21 | seven-z790-eagle / opencode | S0：`v0_1/` -> `mini_vllm/`，更新导入、命令、文档路径和命名 | `unittest discover` 4 passed；example 可运行 |
| 2026-11-21 | seven-z790-eagle / opencode | 建立状态板；删除与 `docs/02` 重复的 roadmap 内容 | 文档链接检查通过；4 tests passed |
| 2026-11-14 | — | mini_vllm 初版基线 | commit `7bb88e7`；4 tests passed |
