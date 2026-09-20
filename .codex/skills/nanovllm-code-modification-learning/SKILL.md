---
name: nanovllm-code-modification-learning
description: Explain nano-vllm-mixed-batch code changes as a focused learning walkthrough. Use with every implementation milestone; do not use for read-only questions or unrelated repository work.
---

# 代码修改学习

## 目的

修改已有代码库时，除了完成工程任务，也要帮助用户逐步理解：

- 原代码如何工作；
- 为什么需要当前修改；
- 修改影响哪条调用链或数据流；
- 修改后系统行为有什么变化；
- 哪些代码值得深入学习，哪些可以暂时略过。

目标不是逐行教学，而是让用户逐渐能够独立阅读、修改和调试本项目。

## 适用范围与边界

- 每次修改本仓库的源代码、测试或行为配置时都使用本 skill，并与 `nanovllm-milestone-development` 一起遵守。
- 本 skill 不替代里程碑开发 skill 的调查、改动预览、验证和工程记录要求。
- 保持最小改动，保留上游行为，一次只改变一个不变量，不重构无关模块，也不扩大当前里程碑范围。
- 先理解现有设计；只要现有架构能够支持需求，优先扩展它，而不是重新设计。

## 一次只处理一个里程碑

每次只实现一个明确的里程碑，并按以下顺序推进：

```text
调查
↓
方案
↓
修改
↓
讲解
```

不要跨多个里程碑做大范围修改。

## 调查时如何讲解

先沿真实执行路径阅读相关代码，不要一开始逐行解释整个文件。确定并说明：

- 当前调用链；
- 核心 class、function 与数据结构；
- 数据从哪里来、最终流向哪里；
- control flow（控制流：代码按什么顺序执行）；
- data flow（数据流：数据如何在模块间传递）；
- state transition（状态转换：请求状态如何变化）；
- module boundary（模块边界：哪个模块负责什么）；
- interface contract（接口约定：调用双方必须共同满足的条件）；
- 当前关键不变量。

## 修改前：Change Preview（改动预览）

在修改文件前，以通俗中文简要说明以下内容：

1. **文件在系统中的作用**：用调用链定位该文件，例如：

   ```text
   Scheduler
       ↓
   SchedulerOutput
       ↓
   LLMEngine
       ↓
   ModelRunner
   ```

2. **为什么必须修改**：说明因果关系，而不只说“为了支持新功能”。例如，旧接口隐含“batch（批次）中请求类型相同”的假设；mixed batch（混合批次）打破这个假设，因此 scheduler（调度器）必须显式描述 execution plan（执行计划）。

3. **准备修改什么**：列出文件、class、method、data structure（数据结构）及各自职责。

4. **修改前后的数据流**：突出数据表达能力的变化。例如：

   ```text
   修改前：Scheduler → list[Sequence] → ModelRunner
   修改后：Scheduler → SchedulerOutput → ModelRunner
   ```

## 修改时：按逻辑步骤推进

不要一次做大量修改。优先拆成逻辑单元：

```text
步骤 1：定义接口或数据结构
步骤 2：修改生产者
步骤 3：修改中间传递层
步骤 4：修改消费者
步骤 5：补充必要的 fail-fast（尽早失败）检查或不变量
```

每完成一个逻辑步骤，简要说明：

- 改了什么；
- 为什么改；
- 新增了什么语义；
- 哪些旧行为保持不变。

不需要解释所有语法细节。

## 关键差异如何解释

对于重要修改，按以下结构说明语义变化：

```text
修改前：原代码做什么
问题：旧设计为什么不能满足当前需求
修改后：现在如何处理
影响：运行时语义发生了什么变化
```

例如，`runner.execute(seqs)` 可能隐含“ModelRunner 默认所有 `seqs` 属于同一种执行模式”；改为 `runner.execute(scheduler_output)` 表示“Scheduler 显式描述执行计划，ModelRunner 只负责执行，不再猜测 Scheduler 的意图”。应解释这种接口语义变化，而不是机械复述 diff。

## 学习优先级

将阅读到的代码按当前里程碑分为三类：

### 建议重点学习

例如 scheduler、batching（批处理）、prefill/decode（预填充/解码）、KV cache、block manager、model runner、attention backend（注意力后端）、CUDA Graph、内存管理、tensor parallel（张量并行）、execution plan 和 sequence state transition（请求状态转换）。

对这类代码说明：为什么存在、输入是什么、输出是什么、调用关系是什么、关键不变量是什么。

### 理解接口即可

例如 configuration（配置）、wrapper（封装层）、adapter（适配层）、CLI（命令行入口）、logging（日志）和 serialization（序列化）。说明职责以及输入输出，不深入内部实现，除非影响当前里程碑。

### 了解即可

例如 helper（辅助函数）、格式化、样板代码、与当前里程碑无关的兼容代码和普通防御性判断。除非影响当前修改，不花大量篇幅解释。

## 默认讲解结构

除非用户明确要求逐行解释，否则从上到下使用：

```text
系统结构
↓
调用链
↓
关键数据结构
↓
关键函数
↓
关键代码
↓
必要实现细节
```

优先沿真实执行路径讲解。若当前里程碑只涉及 `Scheduler → SchedulerOutput → ModelRunner`，只重点解释这段路径。

## 完成后：整体 Walkthrough（运行流程回顾）

里程碑完成后，从真实入口重新走一遍相关调用链，说明：

```text
修改前如何运行
↓
这次需求打破了什么旧假设
↓
新增或改变了什么接口或状态
↓
修改后如何运行
```

用调用链标出真正改变的层，而不是罗列全部文件。

## 工程记录中的学习信息

除里程碑开发 skill 要求的工程记录外，确保记录中能回答：

- **架构决策**：为什么采用当前设计，为什么不选明显的替代方案；
- **不变量**：当前必须始终成立的条件、责任归属和违反后的症状；
- **缺陷或新发现**：隐藏假设、潜在问题、阅读时发现的架构事实及后续工作。

不属于当前里程碑的问题只记录，不顺手修改。

## 默认交互方式

```text
调查相关代码
↓
给出 Change Preview
↓
执行一个逻辑修改步骤
↓
解释关键变化
↓
继续下一个逻辑步骤
↓
完成后做整体 Walkthrough
```

不必每改一行都停下来询问；在没有额外设计决策时可以继续执行。但也不能完成整个里程碑后只给一个 diff。工程修改和代码学习必须同步进行。
