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

## 突发长 Prompt 对 Decode TPOT 的性能实验

总吞吐不能揭示在线服务中的关键时序：GPU 已经在为一批请求逐 token decode（解码）时，一个长 prompt（输入上下文）突然到达。本实验固定 N=16 个正在 decode 的请求，再注入长度为 4K、8K、16K 的一个 prompt；短请求长度为 128 token，批次 token 预算为 2048。每个场景都记录原有 decode 请求从上一个输出 token 到下一个输出 token 的实际间隔，即 TPOT（Time Per Output Token，每输出 token 时间）。计时在每次 `LLM.step()` 前后执行 CUDA 同步，因此包含真实 GPU iteration（一次执行轮次）耗时。

```text
16 个短请求先完成 prefill，进入 RUNNING
    ↓
长 prompt 突然进入 waiting 队列
    ↓
逐轮记录：长 prompt 的 prefill 进度、原 decode 请求是否产出 token、下一 token 的累计等待时间
```

实验脚本是 [bench_tpot_interference.py](bench_tpot_interference.py)，只调用公开的 `LLM.add_request()` 和 `LLM.step()`；它不注入替代调度策略。因此，以完全相同的参数分别在基线 checkout（`bb823b3e`）和当前 mixed-batch checkout 运行，得到的是两种真实调度器行为的对照。

先创建独立基线 worktree，并复制**仅此实验脚本**到该 worktree。必须从基线目录启动 Python，才能让 `import nanovllm` 指向基线代码而不是当前 checkout。

```powershell
git worktree add ..\nano-vllm-baseline bb823b3e06983d71485a8e1f23715ebd87d98ef8
Copy-Item .\bench_tpot_interference.py ..\nano-vllm-baseline\

# 在基线 worktree（bb823b3e）运行
Set-Location ..\nano-vllm-baseline
python bench_tpot_interference.py --model C:\path\to\Qwen3-0.6B `
  --label baseline --output-dir results\tpot-interference\baseline

# 在当前 mixed-batch checkout 运行
Set-Location ..\nano-vllm-mixed-batch
python bench_tpot_interference.py --model C:\path\to\Qwen3-0.6B `
  --label chunked --output-dir results\tpot-interference\chunked
```

两次运行必须使用同一 GPU、模型、`--max-num-batched-tokens`、N 和 prompt 长度。每个输出目录包含：

| 文件 | 内容 | 用途 |
|---|---|---|
| `iterations.csv` | 每个 iteration 的 GPU 耗时、原 decode 产出数、长 prompt 本轮/累计 prefill 进度、该轮若产出 decode token 时的累计 TPOT | 绘制 TPOT 随 iteration 变化的曲线，定位 spike（尖峰） |
| `summary.csv` | 每种 prompt 长度的长 prefill iteration 数、prefill 时仍在 decode 的 iteration 数、零 decode iteration 数、TPOT 最大值/p50/p95、长 prompt 完成时间 | 做 Baseline 与 Chunked Prefill 的汇总对比 |

预期观察不是“Chunked 的总运行时间一定更短”，而是调度公平性：

| 实现 | 长 prompt prefill 时的行为 | TPOT 预期 |
|---|---|---|
| Baseline | waiting 中的长 prefill 优先，且每个 chunk 独占 iteration；已有 decode 请求没有新 token | 最后一次 decode 的累计等待跨越全部长 prefill iteration，形成明显 spike |
| Chunked Prefill | 先排入已有 decode，再以剩余 token 预算推进长 prefill | 每个长 prefill iteration 仍有 decode token；TPOT 接近混合 iteration 耗时，波动更小 |

实际数值依赖 GPU、模型版本、FlashAttention 和频率状态；下方记录的是本仓库当前提交在固定环境中的实测结果，原始逐轮数据也随仓库保存。

### 实测结果（RTX 4090 / Qwen3-0.6B）

实验固定 16 个正在 decode 的请求、128-token 短 prompt、2048-token 批次预算，并分别在上游基线 `bb823b3e` 和当前 mixed-batch 实现执行同一脚本。每个长度运行一次；`max TPOT` 表示已有 decode 请求从上一个输出 token 到下一个输出 token 的最大等待，包含被长 prefill 阻塞的所有 iteration。

| 长 prompt | Baseline：decode / prefill 轮次 | Chunked：decode / prefill 轮次 | Baseline max TPOT | Chunked max TPOT | 尖峰降低 | 长 prompt 完成时间：Baseline → Chunked |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 0 / 2 | 3 / 3 | 64.33 ms | 40.47 ms | 37.1% | 60.26 → 103.05 ms（+71.0%） |
| 8K | 0 / 4 | 5 / 5 | 142.10 ms | 41.22 ms | 71.0% | 138.82 → 165.01 ms（+18.9%） |
| 16K | 0 / 8 | 9 / 9 | 360.80 ms | 66.38 ms | 81.6% | 357.34 → 393.01 ms（+10.0%） |

结果验证了两类互补指标：原有 `bench.py` 表明 mixed-batch 的稳态总体吞吐基本持平（-0.78%）；本实验表明在长 prefill 突发到达时，Chunked Prefill 让每个 prefill iteration 都继续输出 16 个 decode token，并将 16K 场景的 TPOT 尖峰从 360.80 ms 降到 66.38 ms。代价是长 prompt 自身完成更慢；4K 因为多执行一次 mixed iteration，代价最大，而 16K 的完成时间仅增加 10.0%。

原始证据保存在 [Baseline 汇总](tpot-results/baseline/summary.csv)、[Chunked 汇总](tpot-results/chunked/summary.csv) 及两侧的 `iterations.csv`；后者可直接绘制逐 iteration 的 TPOT 曲线。由于每个长度目前只有一次运行，表中 `p95` 与 `max` 接近；正式统计结论应以同一配置重复 3–5 次后的均值和离散度为准。
