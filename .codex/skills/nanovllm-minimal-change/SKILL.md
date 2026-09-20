---
name: nanovllm-minimal-change
description: Keep nano-vllm-mixed-batch implementation diffs minimal by enforcing single-source state, just-in-time decisions, and non-redundant runtime defenses. Use with every implementation milestone, not for read-only exploration.
---

# 最小改动与防御边界

## 目的

每次实现里程碑时，在满足当前需求的前提下，让代码只新增必要的状态、接口和检查。避免为了未来可能的需求提前增加字段、快照、属性或防御逻辑。

本 skill 与 `nanovllm-milestone-development`、`nanovllm-code-modification-learning` 一起使用：前者负责里程碑与正确性，后者负责学习讲解，本 skill 负责收紧代码设计。

## 单一状态来源

- 一个运行时事实只保留一个权威来源。已有字段已经准确表达该事实时，不在计划对象、包装对象或缓存中复制它。
- 只有存在已证实的异步、并发、跨阶段可变性，或需要保留历史值时，才建立快照。说明谁会在“读取快照前”修改原字段，以及快照为何不可由原字段安全推导。
- 计划对象只保存原对象无法表达、但消费者确实需要的新语义。例如，`Sequence` 未表达“本轮执行阶段”，则计划项可保存 `is_prefill`；已有的 `num_scheduled_tokens` 不重复保存。

## 按需计算，不提前预埋行为

- 派生值在最接近实际消费者的位置计算，除非跨模块传递该值能消除已证实的歧义或重复复杂逻辑。
- 不为后续里程碑提前增加字段、属性或分支。当前里程碑只记录和实现当前不变量。
- 例如，采样资格只在 sampler（采样器）即将使用它的里程碑中计算；在 M1 的批次计划中不预存 `should_sample`。
- 若一个字段、属性或辅助构造函数只服务于假设中的后续实现，应删除并在后续 Change Preview 中重新评估。

## 防御性代码的边界

- 先说明要防御的失败模式：它会造成什么错误，在哪个模块第一次可能发生，以及是否会静默损坏结果。
- 每个不变量优先放一处最接近实际风险边界的检查。不要在构造函数、生产者和消费者重复验证同一条件，除非每处面对不同的外部输入或状态变化。
- 保留能避免静默错误的检查。例如，模型返回 token 数与计划请求数不一致时，必须在 `postprocess()` 阻止 `zip()` 静默截断。
- 保留真实执行边界的检查。例如，M1 的 `ModelRunner` 必须在执行前拒绝含 prefill 与 decode 的混合计划，因为该版本尚无 mixed 执行路径。
- 对调度器内部已保证、且没有外部构造入口会破坏的条件，不新增重复的 `__post_init__` 断言。
- 防御代码应针对当前失败模式；不要把“也许有用”的检查当作默认要求。

## 数据结构与命名

- 容器名必须反映其中元素的真实类型。若列表从 `Sequence` 变为 `ScheduledSequence`，使用 `scheduled` 而不是误导性的 `scheduled_seqs`。
- 新接口应只携带跨模块必须传递的信息。消费者能够从原对象可靠读取的信息，不包装为新属性。
- 允许为兼容调用链保留很小的访问属性；不要把它扩展成通用框架。

## 修改前检查清单

每次编辑前，对计划的每个新增字段、属性、辅助函数或断言回答：

1. 它解决当前里程碑的哪个具体问题？
2. 现有状态或现有控制流能否已经表达或保证它？
3. 能否在真正使用它的模块临时计算？
4. 若删除它，当前行为或当前不变量会在哪里失效？
5. 它是否只是为未来里程碑预埋？若是，删除并留到未来的 Change Preview。

在 Change Preview 中简要写出这些结论；在 Change Review 中说明主动删除或未新增了哪些冗余设计，以及保留的防御检查防止什么具体错误。
