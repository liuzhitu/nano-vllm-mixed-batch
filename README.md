# nano-vllm-mixed-batch

本项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，为其补充同一批次内同时执行 decode 与 prefill 的 mixed-batch 推理能力。

## 实现内容

相对 mixed-batch 改动前的上游基线，核心运行时代码仅修改了 3 个文件：增加了 52 行代码**。



| 文件 | 新增 / 删除 | 新增或修改的函数、类型 | 作用 |
|---|---:|---|---|
| `nanovllm/engine/scheduler.py` | +55 / -28 | 新增 `ScheduledSequence`、`SchedulerOutput`；修改 `Scheduler.schedule()`、`Scheduler.postprocess()` | 用每个请求自己的阶段描述本轮执行计划；先选择 decode，再用剩余 token 预算安排 prefill；后处理按请求提交 token 或部分 prefill 的 KV 进度。 |
| `nanovllm/engine/model_runner.py` | +34 / -9 | 修改 `warmup_model()`、`prepare_sample()`、`run()` | `run()` 接收执行计划；mixed batch 使用一次 varlen forward；只为 decode 与完成 prefill 的请求采样，部分 prefill 返回 `None` 占位。 |
| `nanovllm/engine/llm_engine.py` | +5 / -5 | 修改 `LLMEngine.step()` | 将同一份执行计划依次交给模型执行与调度器后处理，避免执行阶段信息在中间丢失。 |

执行路径为：

```text
Scheduler.schedule()
    ↓
SchedulerOutput[ScheduledSequence]
    ↓
ModelRunner.run()
    ↓
一次 varlen forward（mixed）或原有 decode 快路径（pure decode）
    ↓
Scheduler.postprocess()
```

## 验证结果

原有的 `example.py` 与 `bench.py` 均已在本地 Qwen3-0.6B、NVIDIA GeForce RTX 4090 上通过；前者确认离线批量问答生成可用，后者确认 256 请求端到端生成可完成。

另外设计了一个小 token 预算的 mixed-batch GPU 测试如下。

| 场景 | 测试| 结果 |
|---|---|---|
| 纯路径回归 | 两个 prompt（5、7 token），各生成最多 2 token；先检查 pure prefill 的 varlen 路径，再检查 pure decode 的原路径、请求完成与 KV 释放。 | PASS |
| decode + 完整 prefill | 先使一个 6-token prompt 进入 running，再加入一个 5-token prompt；计划为 `[decode, prefill]`，共 6 个计算 token，一次 varlen forward，两个请求都必须采样。 | PASS |
| decode + 部分 prefill | 先使一个 6-token prompt 进入 running，再加入一个 20-token prompt；批次预算为 16，因此计划必须为 `1 decode + 15 prefill`。断言部分 prefill 返回 `None`、保留在 waiting、只提交 15 token 的 KV 进度。 | PASS |
| 采样随机数 | 将“只有一个 decode”的 CUDA RNG 状态和采样 token，与“同一个 decode 加一个部分 prefill”比较；两者必须完全相同。 | PASS |
| 序列数上限 | 先让 4 个请求进入 running，再加入新 prefill；4 个 decode 必须保持 FIFO 顺序并占满序列槽位，新请求留在 waiting。 | PASS |
| prefix cache | 先建立 256-token 前缀缓存；随后让一个 decode 与共享该前缀、只缺 1 token 的新请求同批执行。缓存的 256 token 不得占用本轮 token 预算。 | PASS |
| KV 抢占与 EOS | 以确定性的小 KV 容量验证 decode 被抢占后释放所有权并以 prefill 重排；以 `eos=42` 验证完成后释放 KV、移出 running。 | PASS |


同一套输入还与原始实现进行了输出 token 对照，结果一致。

## 性能结果

性能使用原始 `bench.py`，固定随机种子，创建 256 个请求；prompt 长度与生成上限都随机分布在 100–1024 token，总生成量为 133,966 token。下方先保留两次实验各自的原始输出，再给出采用的稳态对比。

**原nano-vllm输出：**

```text
Run 1  Total: 133966tok, Time: 23.87s, Throughput: 5612.04tok/s
Run 2  Total: 133966tok, Time: 22.36s, Throughput: 5991.02tok/s
Run 3  Total: 133966tok, Time: 22.46s, Throughput: 5965.06tok/s
Run 4  Total: 133966tok, Time: 22.25s, Throughput: 6020.38tok/s
Run 5  Total: 133966tok, Time: 22.39s, Throughput: 5982.44tok/s
```

**mixed-batch 输出：**

```text
Run 1  Total: 133966tok, Time: 22.60s, Throughput: 5928.74tok/s
Run 2  Total: 133966tok, Time: 22.52s, Throughput: 5948.63tok/s
Run 3  Total: 133966tok, Time: 22.56s, Throughput: 5938.01tok/s
Run 4  Total: 133966tok, Time: 22.57s, Throughput: 5934.51tok/s
Run 5  Total: 133966tok, Time: 22.51s, Throughput: 5951.75tok/s
```



| 指标 | nano-vllm | mixed-batch  | 差异 |
|---|---:|---:|---:|
| 全部 5 次平均吞吐 | 5914.19 tok/s | 5940.33 tok/s | +26.14 tok/s（+0.44%） |
| 稳态吞吐（第 2–5 次平均） | 5989.73 tok/s | 5943.23 tok/s | -46.51 tok/s（-0.78%） |
| 稳态样本标准差 | 23.11 tok/s | 8.27 tok/s | -14.84 tok/s |
