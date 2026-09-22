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

## 性能结果

原始 `bench.py`实验，固定随机种子，创建 256 个请求；prompt 长度与生成上限都随机分布在 100–1024 token，总生成量为 133,966 token。下方先保留两次实验各自的原始输出，再给出采用的稳态对比。

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

## 混合批次性能实验

GPU 已经在为一批请求逐 token decode（解码）时，一个长 prompt（输入上下文）突然到达。

固定 N=16 个正在 decode 的请求，再注入长度为 4K、8K、16K 的一个 prompt；短请求长度为 128 token，批次 token 预算为 2048。

每个场景都记录原有 decode 请求从上一个输出 token 到下一个输出 token 的实际间隔，即 TPOT（Time Per Output Token，每输出 token 时间）。计时在每次 `LLM.step()` 前后执行 CUDA 同步，因此包含真实 GPU iteration（一次执行轮次）耗时。

```text
16 个短请求先完成 prefill，进入 RUNNING
    ↓
长 prompt 突然进入 waiting 队列
    ↓
逐轮记录：长 prompt 的 prefill 进度、原 decode 请求是否产出 token、下一 token 的累计等待时间
```

实验脚本是 [bench_tpot_interference.py](bench_tpot_interference.py)，以完全相同的参数分别在基线 checkout（`bb823b3e`）和当前 mixed-batch checkout 运行，得到的是两种真实调度器行为的对照。

### 实测结果（RTX 4090 / Qwen3-0.6B）

实验固定 16 个正在 decode 的请求、128-token 短 prompt、2048-token 批次预算。每个长度运行一次；`max TPOT` 表示已有 decode 请求从上一个输出 token 到下一个输出 token 的最大等待，包含被长 prefill 阻塞的所有 iteration。

| 长 prompt | Baseline：decode / prefill 轮次 | Chunked：decode / prefill 轮次 | Baseline max TPOT | Chunked max TPOT | 尖峰降低 | 长 prompt 完成时间：Baseline → Chunked |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 0 / 2 | 3 / 3 | 64.33 ms | 40.47 ms | 37.1% | 60.26 → 103.05 ms（+71.0%） |
| 8K | 0 / 4 | 5 / 5 | 142.10 ms | 41.22 ms | 71.0% | 138.82 → 165.01 ms（+18.9%） |
| 16K | 0 / 8 | 9 / 9 | 360.80 ms | 66.38 ms | 81.6% | 357.34 → 393.01 ms（+10.0%） |

结果表明在长 prefill 突发到达时，Chunked Prefill 让每个 prefill iteration 都继续输出 16 个 decode token，并将 16K 场景的 TPOT 尖峰从 360.80 ms 降到 66.38 ms。代价是长 prompt 自身完成更慢；4K 因为多执行一次 mixed iteration，代价最大，而 16K 的完成时间仅增加 10.0%。

