# AGENTS.md

> 只保留硬性约定。当前状态、任务认领和历史在 [docs/00-status.md](docs/00-status.md)。

## 开工前

1. `git pull --ff-only`
2. 读状态页的“当前快照”和“当前认领”
3. 在认领表加一行：任务、主机 / agent、分支、状态
4. 分支用 `task/<任务ID>-<host>`；纯文档小改可直接 `main`

## 硬性规则

- 只做当前里程碑范围内的任务；改范围先更新状态页
- Python >= 3.10，M0–M2 只用标准库
- 分层：`serving -> engine -> model`，`model` 不反向依赖 `engine`
- 改公共契约必须同步 `engine/types.py`、实现、测试和文档
- 不做无关重构；先正确，后性能
- 测试：`python3 -m unittest discover -v`，必须全绿
- 完成任务：更新状态页的任务状态 + 在“变更日志”加一行，代码和文档同一次 commit
- 提交信息用 Conventional Commits，例如 `feat(scheduler): add token budget`

## 不能破坏的 invariants

1. prefill + decode 的 logits 与 `forward_full` 一致
2. 请求结束必须释放 KV cache
3. `generate()` 只能在 engine 空闲时调用
4. `SchedulerOutput` 是 Scheduler -> ModelRunner 的唯一输入契约
5. `needs_logits=False` 的请求不得产出 token
6. 每个调度步至少推进一个 token
7. KV 分配失败必须报错，不能静默跳过

## 冲突处理

- 用 `git pull --rebase`；不要 force push `main`
- 认领表冲突：以先 push 且已开始工作的主机为准，另一方换任务
- 决策有分歧：在状态页的“决策记录”加一行，写清选项和取舍
