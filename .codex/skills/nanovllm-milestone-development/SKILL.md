---
name: nanovllm-milestone-development
description: Implement changes to the nano-vllm-mixed-batch inference runtime through small, evidence-backed milestones. Use for any code, test, or behavior-changing configuration work in this repository; not for read-only exploration or general explanations.
---

# nano-vLLM Milestone-Driven Development

Use this skill for every behavior-changing task in this repository. The goal is a sequence of small, explainable, empirically verified changes that preserve the supported upstream paths while building true mixed-batch execution.

For every implementation milestone, after reading this skill, also read and apply
[`nanovllm-code-modification-learning`](../nanovllm-code-modification-learning/SKILL.md)
and [`nanovllm-minimal-change`](../nanovllm-minimal-change/SKILL.md).
This skill owns the milestone lifecycle and correctness constraints; the linked
skills own the user-facing learning walkthrough and minimal-diff review.

Whenever creating or updating a milestone engineering record or a project
code-change log, also read and apply
[`nanovllm-code-change-log`](../nanovllm-code-change-log/SKILL.md).

## Scope and baseline

- Treat the checked-out upstream-derived baseline as an asset. Do not change public behavior, pure prefill, pure decode, prefix-cache lifecycle, block ownership, or CUDA-graph decode behavior unless the current milestone explicitly names that behavior.
- Start each milestone by inspecting `git status`, the relevant call path, and the existing tests or executable checks. Search and trace actual code; do not infer behavior from the target architecture or from old notes.
- The initial project baseline is `bb823b3e06983d71485a8e1f23715ebd87d98ef8`. Preserve a clear before/after comparison for any milestone that touches its runtime behavior.
- Keep the current milestone narrow. Record newly discovered work as future work rather than implementing it early.

## Required milestone lifecycle

Follow these stages in order. Do not edit product source during Investigation.

1. **Investigation** — Identify the smallest real blocker. For each relevant finding, capture the file, function/class, current behavior, state read or mutated, applicable assumption/invariant, and why it matters.
2. **Design** — Based only on the findings, specify current behavior, desired behavior, invariant introduced or changed, minimal modification points, and behavior that must remain unchanged. Reuse existing abstractions where possible. Do not wait for a separate design approval unless the user asks for one or the design materially expands scope.
3. **Implementation** — Make the smallest diff that implements only this milestone. Do not combine unrelated scheduler, KV, attention, sampling, memory-management, or cleanup work unless they are inseparable for the stated invariant.
4. **Verification** — Run the relevant existing checks plus targeted checks for the new behavior. Inspect runtime state, queue/status transitions, tensor shapes, metadata, and outputs when they participate in the change. Verify preserved legacy paths as well as the new path.

When verification fails, observe the failure, find the earliest incorrect state transition and its owner, return to Investigation, and make the next minimal fix. Do not replace the design merely to hide a failed check.

## Mixed-batch project invariants

Apply these whenever the milestone reaches the corresponding layer:

- A plan is not a completed computation: `num_scheduled_tokens` must not advance `num_computed_tokens` until successful execution commits it.
- `num_computed_tokens` denotes the contiguous request-local prefix with valid, visible KV. Allocated KV capacity may exceed it; attention must never consume uncomputed KV.
- Prefix-cache adoption is valid computed progress, but it does not consume the current GPU token budget.
- A SchedulerOutput must respect the global token budget and contain each request at most once, in a documented stable order.
- Sampling eligibility is per request. A partial prompt may commit compute/KV progress but must not emit a generated token.
- Pure prefill and pure decode remain valid compatibility paths while mixed execution is being introduced. Mixed path correctness takes priority over preserving a CUDA-graph fast path; do not broaden that performance work into the same milestone.

## Learning and change review protocol

Every milestone must make the codebase and the change teachable. Do not merely say that files changed: explain how the affected subsystem worked before the change, why the change is required, and how it works afterwards.

### 项目记录语言与表达

- 面向用户的计划、Change Preview（改动预览）、Change Review（改动复盘）、验证计划、完成报告、工程记录和项目日志必须使用中文。
- 优先使用短段落和分点，先说明结论，再说明原因与证据；避免只罗列文件名或术语。
- 专业名词在一次记录中首次出现时，须在括号中给出简短中文解释；代码标识符、命令和接口名保持原样。例如：`SchedulerOutput`（调度器生成的批次执行计划）。
- 仅当英文是代码、命令、文件路径、通行缩写或与中文并列有助于定位时保留英文；不可用英文标题替代中文说明。

### Change Preview — required before editing

Before modifying product source, tests, or behavior-changing configuration, present a Change Preview in the user-visible response. It must include:

1. **Relevant runtime path** — show the caller-to-consumer path, including the data structure that carries the changed behavior, and locate this milestone in the wider runtime.
2. **Current behavior** — identify the files, functions/classes, data flow, and the current assumption that blocks the desired behavior.
3. **Planned files** — for every expected changed file, state its relevant class/function, current responsibility, why this milestone needs it, and the expected change. Explicitly name nearby components that will not change yet.
4. **Before / After** — show the conceptual state, structure, or flow transition whenever it clarifies the change.

Do not begin implementation until the planned change is clear enough to explain. A separate approval is not needed unless the design materially expands the user's requested scope.

### Small editing steps

Do not turn a milestone into one opaque patch. Before each logical edit, state:

```text
步骤 N
修改位置：<文件 / 函数>
目的：<一个行为或不变量>
修改前：<当前逻辑>
修改后：<新逻辑>
```

Then make that edit. Keep unrelated concepts in separate steps.

### Change Review — required after editing

After an edit, explain its semantic diff for every changed file: what changed, why, the important code path, old behavior, new behavior, and the invariant affected. End with a short, ordered **Recommended reading** list naming the two to five code locations the user should inspect personally.

### Verification plan — required before tests

Before running tests, state what the milestone must prove. For each planned test, explain the constructed input/state, expected behavior, invariant checked, production path represented, and whether it requires a GPU. Tests are architecture evidence, not a black box.

### CPU-first verification

Classify checks as CPU/pure Python, static/import/type/syntax, mocked component, GPU runtime, or performance. Run all applicable non-GPU checks first.

Scheduler decisions, request state transitions, token budgets, SchedulerOutput construction, queue ordering, metadata construction, fail-fast behavior, bookkeeping, and configuration validation should normally be testable without GPU.

If GPU verification cannot run locally, report exactly:

```text
本地已验证：
...

尚未验证：
...

原因：
需要 CUDA/GPU

后续 GPU 测试：
...
```

Never present an unavailable GPU test as passed.

## Verification standard

Use focused tests or a disposable control-plane fixture before relying on real-model execution. A milestone is not verified merely because it compiles or looks plausible.

For scheduler/progress work, assert at least the relevant subset of:

- scheduled-token total and sequence-count limits;
- request uniqueness and output order;
- waiting/running/finished transitions;
- block-table and KV allocation ownership;
- computed versus scheduled progress boundaries;
- prefix-hit accounting;
- legacy pure-prefill and pure-decode behavior.

For runner/attention work, additionally inspect packed input order, `q_len`/`k_len`, positions, `cu_seqlens`, `slot_mapping`, block tables, attention dispatch, selective logits, and post-execution commit behavior.

## Engineering record

At completion of every milestone, create or update `docs/milestones/<milestone-slug>.md`. Do not pre-create empty records. Use this structure:

```markdown
# 里程碑：<名称>

## 目标

## 调查

## 设计

## 实现

## 验证

## GPU 验证缺口

## 工程记录

### 架构决策

决策、原因、备选方案，以及不选择备选方案的理由。

### 不变量

条件、负责方、原因，以及违反时的表现。

### 缺陷或新发现

观察到的现象、根本原因、影响和后续工作。

## 后续工作
```

最终里程碑报告必须覆盖：系统理解、改动文件、运行流程、关键差异、验证、GPU 验证缺口、工程记录，以及下一个最小的独立里程碑。说明改了什么、为何要改、影响的不变量、实际运行的验证及其结果，以及刻意保持不变的部分。

## Explicit prohibitions

- Do not implement multiple milestones in one pass.
- Do not refactor for style, generality, or hypothetical future needs while correctness is uncertain.
- Do not claim success without reporting the checks actually run and their results.
- Do not silently discard unrelated existing changes.
