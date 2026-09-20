# Mixed-batch 性能基线：环境背景

## 采集信息

- 采集时间：2026-09-20 09:12:46（`nvidia-smi` 输出）
- 工作目录：`~/nano-vllm-mixed-batch`
- 预期基线提交：`1e187b7`（待在实例上复核）

## 宿主机与 GPU

| 项目 | 值 |
| --- | --- |
| 操作系统内核 | Linux `5.15.0-25-generic-x86_64`，glibc `2.35` |
| GPU | 1 × NVIDIA GeForce RTX 4090 |
| 显存 | 24,564 MiB（约 24 GB） |
| GPU UUID | `GPU-d9d68272-da43-536f-cc62-14341cfe6700` |
| NVIDIA 驱动 | `570.124.04` |
| 驱动报告的 CUDA 兼容版本 | 12.8 |
| 初始 GPU 状态 | P8，35°C，20W，1 MiB 已用，0% 利用率，无运行进程 |
| CUDA 编译器 | **未安装**：`nvcc: command not found` |

## 当前 Python 环境

| 项目 | 值 |
| --- | --- |
| Python | 3.12.3 |
| Python 路径 | `/root/miniconda3/bin/python3` |
| PyTorch | `2.8.0+cu128` |
| PyTorch CUDA runtime | 12.8 |
| `torch.cuda.is_available()` | `True` |
| Triton | 3.4.0 |
| Transformers | 5.14.1 |
| FlashAttention | **未安装**：`ModuleNotFoundError: No module named 'flash_attn'` |
| xxhash | **未安装**：`ModuleNotFoundError: No module named 'xxhash'` |

另有非阻断性环境告警：`libgomp: Invalid value for environment variable OMP_NUM_THREADS`；基准运行前应将该变量设为合法正整数。

## 结论与下一步

硬件、驱动和当前 PyTorch CUDA runtime 均可识别 RTX 4090，因此实例适合作为单卡性能基线硬件。项目依赖 `flash-attn`、`xxhash`，且 FlashAttention 的常规源码安装需要 CUDA 编译器；当前缺失的 `nvcc` 是阻塞项。

下一步顺序：先切换或配置带 CUDA Toolkit（含 `nvcc`）的宿主机镜像，再创建项目专用 Conda/venv 环境并安装依赖。虚拟环境不能提供宿主机 CUDA 编译器，不能单独解决该阻塞项。

## 当前基线实例快照（2026-09-20）

此快照替代上方的旧实例信息，作为性能基线的候选运行环境。基线代码使用当前远端 `main`，而非本地未推送的 `1e187b7`。

| 项目 | 值 |
| --- | --- |
| 采集时间 | `2026-09-20T10:02:27+08:00` |
| 主机名 | `autodl-container-7dde4b87a0-1130918c` |
| 操作系统内核 | Linux `5.15.0-133-generic`，x86_64 |
| GPU | 1 × NVIDIA GeForce RTX 4090，24,564 MiB |
| NVIDIA 驱动 | `580.76.05` |
| 驱动报告的 CUDA 兼容版本 | 13.0 |
| 已发现的 CUDA Toolkit 路径 | `/usr/local/cuda-12.8/bin/nvcc` |
| 当前 shell 的 `CUDA_HOME` | 未设置 |
| 当前 shell 的 `nvcc` | 不在 `PATH`，`nvcc` 调用失败 |
| 当前 `OMP_NUM_THREADS` | `16` |
| Python | 3.12.3，`/root/miniconda3/bin/python` |
| PyTorch | `2.8.0+cu128` |
| Triton | `3.4.0` |
| Transformers / FlashAttention / xxhash | 未安装 |
| Git 分支与提交 | `main`，`bb823b3e06983d71485a8e1f23715ebd87d98ef8` |
| 根分区 | 30 GB，总用量 747 MiB，约 30 GB 可用 |
| 共享内存 | `/dev/shm`，60 GB 可用 |

### 运行前待办

1. 将 `/usr/local/cuda-12.8/bin` 加入 `PATH` 并设置 `CUDA_HOME=/usr/local/cuda-12.8`，以使 `nvcc --version` 可用。
2. 在项目虚拟环境内安装 `transformers`、`flash-attn` 和 `xxhash`；PyTorch 与 CUDA 版本已作为候选组合记录。
3. 安装过程使用 `PIP_NO_CACHE_DIR=1`，将临时编译目录设为 `/dev/shm`，避免占满 30 GB 根分区。
