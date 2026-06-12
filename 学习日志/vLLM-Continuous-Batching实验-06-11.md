---
title: vLLM Continuous Batching 实验：理解推理调度的核心优化
date: 2026-06-11
tags:
  - R/技术框架/vLLM
  - R/核心算法/LLM推理
  - R/核心算法/推理优化
  - output/active
---

# vLLM Continuous Batching 实验：理解推理调度的核心优化

> 本文在 RTX 3060 12GB (WSL2) 上，通过三组对比实验深入理解 vLLM 的 Continuous Batching 机制。包含原理讲解、实验数据分析和调参建议。

---

## 零、为什么做这个实验

### 要解决的真实问题

部署一个 LLM 服务时，最关心两件事：

1. **吞吐量（能服务多少用户）** —— 一张昂贵的 GPU，每秒能处理多少请求？直接决定单位成本。
2. **延迟（用户等多久）** —— 用户发出请求后，多快看到第一个字？多快拿到完整回答？

这两者常常**互相矛盾**：想提高吞吐就要塞更多请求一起算，但请求一多，单个请求就可能被拖慢。vLLM 之所以出名，核心就是用 **Continuous Batching + PagedAttention** 在这对矛盾里找到了远好于传统方案的平衡点。

> 但"vLLM 很快"只是结论。**我想亲眼看到它有多快、为什么快、以及调哪个参数会让它变慢**——这才是这次实验的目的。

### 为什么必须用"实验"而不是"看文档"

LLM 推理的性能高度依赖**具体硬件 + 具体模型 + 具体负载**：

- 官方 benchmark 多在 A100/H100 上跑 7B+ 模型，**和我的 RTX 3060 + 1.5B 模型完全不是一个量级**。
- 参数（如 `max_num_seqs`）的"最优值"没有标准答案，取决于你的显存、算力、请求特征。
- 只有自己压测，才能得到**属于自己环境的那条吞吐曲线**，知道"并发开到多少会饱和"。

### 这次实验想回答的 3 个问题

| 问题 | 对应实验 | 想验证的事 |
|---|---|---|
| Q1：并发增加，吞吐和延迟怎么变？ | 实验 1 | Continuous Batching 到底带来多大提升？拐点在哪？ |
| Q2：限制同时处理的请求数会怎样？ | 实验 2（`max_num_seqs`） | 这个参数如何成为吞吐的"天花板" |
| Q3：限制单批 token 数会怎样？ | 实验 3（`max_num_batched_tokens`） | Chunked Prefill 如何影响首 token 延迟 |

---

## 一、原理：Continuous Batching vs Static Batching

### 先理解：为什么"批处理"本身能提速？

在讲两种 batching 的区别前，要先明白一个底层事实——**LLM 推理的 decode 阶段是"显存带宽瓶颈"，不是"算力瓶颈"**。

每生成一个 token，GPU 都要把整个模型的权重（1.5B 模型约 3GB）从显存读进计算单元过一遍。这个"搬运权重"的开销是固定的，无论你是给 1 个请求算，还是给 50 个请求一起算：

```
单个请求 decode 1 个 token：
  读 3GB 权重  →  只算了 1 个 token 的矩阵乘  →  GPU 算力大量闲置

50 个请求一起 decode：
  读 3GB 权重（同样一次） →  算了 50 个 token 的矩阵乘  →  GPU 算力被喂饱
```

> **关键洞察**：权重只读一次，却能服务整个 batch。所以 **batch 越大，单 token 的"权重搬运成本"被摊得越薄，吞吐就越高**——这是所有 batching 技术的物理基础，也是后面实验 1 中"并发 1→5 吞吐涨 3 倍"的根本原因。

那么问题来了：既然 batch 越大越好，为什么传统的 Static Batching 不行？

### Static Batching（传统方式）

```
时间 →  step1   step2   step3   step4   step5   step6
Req A   [prefill][decode][decode][decode][done ] [     ]  ← 4 tokens
Req B   [prefill][decode][decode][decode][decode][done ]  ← 5 tokens
Req C   [prefill][decode][done ] [     ] [     ] [     ]  ← 3 tokens

Batch 必须等最长的请求 B 完成(6 steps), 才能 开始处理下一批。
C 在 step3 就完成了, 但它的 KV cache 位置被白白占着。
```

**问题：**
- 短请求被长请求拖慢
- 已完成请求的 KV cache 空间浪费
- GPU 利用率低（完成后位置空闲但不分配给新请求）

### Continuous Batching（vLLM / Orca 论文提出）

```
时间 →  step1   step2   step3   step4   step5   step6
Req A   [prefill][decode][decode][decode][done ]
Req B   [prefill][decode][decode][decode][decode][done ]
Req C   [prefill][decode][done ]
Req D                     [prefill][decode][done ]         ← C 完成后立即加入
Req E                                       [prefill][decode][done ]  ← A 完成后立即加入
```

**每个 step, scheduler 做 3 件事：**
1. **检查 running 队列**：已生成 EOS 的序列立即移除，释放 KV cache
2. **从 waiting 队列取新请求**：有空位就加入，做 prefill
3. **执行 decode**：对所有 running 序列推进一个 token

**核心优势：**
- 短请求不再等长请求 → **TTFT 和延迟都降低**
- KV cache 立即回收 → **显存利用率提高，能服务更多并发**
- GPU 始终满载 → **吞吐量大幅提升**

### vLLM 的实现细节

vLLM 的 Continuous Batching 建立在 **PagedAttention** 之上：
- KV cache 按 page 管理（类似虚拟内存），不需要连续显存
- 新请求分配物理 page，完成后 page 立即归还 page pool
- 配合 **Chunked Prefill**：长 prompt 分 chunk 处理，避免单个长 prompt 阻塞整个 batch

三个关键调优参数控制 scheduler 行为：

| 参数 | 含义 | 默认值 | 影响 |
|---|---|---|---|
| `max_num_seqs` | batch 中最大序列数 | 256 | 限制并发度，影响 CUDA Graph 大小 |
| `max_num_batched_tokens` | 单 batch 最大 token 数（prefill + decode） | 8192 | 控制 prefill chunk 大小，影响 TTFT |
| `gpu_memory_utilization` | GPU 显存利用率阈值 | 0.9 | 决定 KV cache pool 大小 |

#### 逐个理解这三个参数的"物理意义"

**`max_num_seqs`——"同时能开几条流水线"**

把它想象成餐厅同时能上几桌客人。设为 256，就是 scheduler 允许同一个 batch 里最多有 256 个请求一起 decode。
- 它是**并发的硬上限**：哪怕有 1000 个请求涌进来，一个 batch 也最多放 256 个，其余在 waiting queue 排队。
- 副作用：vLLM 会为不同 batch size 预先捕获 CUDA Graph（一种把 GPU 操作录制成"快捷方式"的加速技术）。`max_num_seqs` 越大，要捕获的 graph 越多，**占用更多显存、启动更慢**。
- 实验 2 就是把它从 256 砍到 8，观察"流水线变窄"对吞吐的影响。

**`max_num_batched_tokens`——"每一步最多嚼多少 token"**

scheduler 每个 step 能处理的 token 总数上限（prefill 的输入 token + decode 的输出 token 加起来）。
- 主要影响 **prefill 阶段**。一个 2000 token 的长 prompt，如果上限是 8192，一步就能 prefill 完；如果上限是 512，就得切成 4 块（**Chunked Prefill**），分 4 步处理。
- 切块的好处：长 prompt 不会"霸占" GPU，短请求能插队，整体更公平。
- 切块的代价：被切的那个请求 prefill 要走多步，它自己的 **TTFT 变长**。
- 实验 3 就是把它从 8192 砍到 512，观察 prefill 切块对 TTFT 的影响。

**`gpu_memory_utilization`——"划多少显存给 vLLM"**

允许 vLLM 使用的显存比例。模型权重装完后，**剩下的显存几乎全部拿来做 KV cache**（存放每个请求的注意力上下文）。
- KV cache 越大，能同时容纳的请求/上下文越多 → 间接决定了真实的并发容量。
- 这是本次实验的固定参数（0.85），但要知道：它才是并发容量的"总闸门"，`max_num_seqs` 只是在这个闸门内再设一道限制。

> **一句话串起来**：`gpu_memory_utilization` 决定"显存能放下多少请求"，`max_num_seqs` 决定"调度器愿意同时处理多少请求"，`max_num_batched_tokens` 决定"每一步的工作量怎么切"。三者层层收窄，共同决定最终性能。

---

## 二、实验设计

### 环境

- RTX 3060 12GB, WSL2, vLLM 0.22.1
- 模型: Qwen/Qwen2.5-1.5B (bfloat16)
- 基础参数: `--max-model-len 4096 --gpu-memory-utilization 0.85`

### 三个核心指标的含义（看懂数据的前提）

后面所有表格都围绕这三个指标，先搞清楚它们各自衡量什么、用户/运维分别关心哪个：

| 指标 | 全称 | 含义 | 谁关心 | 越高/越低好 |
|---|---|---|---|---|
| **RPS** | Requests Per Second | 每秒完成的请求数 = 吞吐量 | 运维/成本 | 越高越好 |
| **TTFT** | Time To First Token | 从发请求到收到**第一个 token** 的时间 | 用户体验 | 越低越好 |
| **Latency** | End-to-End Latency | 从发请求到收到**完整回答**的时间 | 用户体验 | 越低越好 |

几个容易混淆的点：

- **RPS vs Latency 是两个维度**：RPS 衡量"系统整体能扛多少"，Latency 衡量"单个用户等多久"。高并发下 RPS 可能很高，但单个请求的 Latency 反而变长（因为在排队）。
- **为什么 TTFT 特别重要**：聊天场景里，用户看到"开始打字"就觉得系统在响应了。TTFT 低 = 体感快。TTFT 主要由 **prefill 阶段**（处理输入 prompt）耗时决定，所以实验 3 调 `max_num_batched_tokens`（控制 prefill）时，TTFT 变化最明显。
- **mean / p50 / p95 / max 的区别**：mean 是平均值，p95 表示"95% 的请求都比这个值快"（即最慢的 5% 有多慢）。**p95 比 mean 更能反映真实体验**——少数请求卡顿，用户就会抱怨。
- **为什么用 streaming（流式）API**：只有流式返回才能精确测出"第一个 token 到达的时刻"，否则无法区分 TTFT 和总延迟。

### 测试方法

- 使用 **streaming API** 发送并发请求（为了精确测 TTFT）
- 测量指标：**RPS**、**TTFT**、**Latency**（含义见上表）
- 每组 40 个请求，mixed prompt（短+中混合，模拟真实负载）
- 每个 `max_tokens=32`（限制输出长度，让测试快速可复现）
- 用 `asyncio.Semaphore` 控制并发数：同一时刻最多 N 个请求在途，模拟 N 个并发用户

### 三组实验

| 实验 | 变量 | 固定参数 |
|---|---|---|
| 实验 1：不同并发数 | concurrency = 1, 5, 10, 20, 40 | 默认参数 (max_num_seqs=256, max_num_batched_tokens=8192) |
| 实验 2：max_num_seqs | max_num_seqs = 8 vs 256 (默认) | concurrency = 5, 10, 20, 40 |
| 实验 3：max_num_batched_tokens | = 512 vs 8192 (默认) | concurrency = 5, 10, 20, 40 |

---

## 三、实验 1：不同并发数

> **这组实验想看什么**：在参数全默认的情况下，单纯增加"同时发起的请求数"，吞吐和延迟如何变化。这能直接画出本机的**吞吐曲线**，找到"性价比最高的并发区间"和"开始饱和的拐点"。并发=1 作为基准（相当于退化成无 batching 的串行处理），用来对比 batching 带来的提升倍数。

**服务器配置**：默认参数（max_num_seqs=256, max_num_batched_tokens=8192）

| 并发数 | RPS | TTFT mean | TTFT p95 | Latency mean | Latency max | Wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.9 | 0.061s | 0.108s | 0.420s | 1.387s | 1.39s |
| 5 | **88.1** | 0.038s | 0.045s | 0.365s | 0.454s | 0.45s |
| 10 | **90.7** | 0.044s | 0.054s | 0.382s | 0.441s | 0.44s |
| 20 | 73.9 | 0.071s | 0.078s | 0.460s | 0.541s | 0.54s |
| 40 | 69.1 | 0.128s | 0.131s | 0.534s | 0.579s | 0.58s |

### 分析

1. **吞吐量 (RPS) 在并发=10 时达到峰值 90.7 RPS**
   - 并发 1→5：RPS 从 28.9 飙升到 88.1（**3x 提升**）— 这就是 Continuous Batching 的威力
   - 并发 5→10：小幅提升到 90.7（已经接近 GPU 算力上限）
   - 并发 20+：开始下降 — GPU 算力饱和，更多请求只能排队

2. **TTFT 在高并发下上升**
   - 并发 1-10：TTFT ~40ms，几乎没有排队
   - 并发 40：TTFT 升到 128ms — 请求在 waiting queue 中等待

3. **Latency max 在并发=1 时异常高（1.387s）**
   - 这是 JIT 编译导致的首次延迟（Triton kernel 冷启动）
   - 并发 5+ 后消失，因为第一个请求触发了 JIT，后续请求受益

4. **Wall time 在并发 5-10 几乎相同（0.44-0.45s）**
   - 说明 GPU 在这个区间刚好被充分利用
   - 这就是最佳并发区间

> **结论：** RTX 3060 + Qwen2.5-1.5B 的最佳并发区间在 **5-10**，此时吞吐最高、延迟最低。

---

## 四、实验 2：max_num_seqs 的影响

> **这组实验想看什么**：验证 `max_num_seqs` 是不是吞吐的"天花板"。选 **8 vs 256** 这组对照，是因为实验 1 测出最佳并发在 10 左右——把上限压到 **8（低于最佳并发）**，就能人为制造"流水线不够用"的场景，观察请求被迫排队后吞吐和 TTFT 的恶化程度。这是一种"反向验证"：通过把参数调坏，看它到底卡住了什么。

**对比配置：**

| 配置 | max_num_seqs | CUDA Graph 大小 | KV cache |
|---|---|---|---|
| 默认 | 256 | 51 PIECEWISE + 35 FULL | 228K tokens |
| 限制 | 8 | 5 PIECEWISE + 4 FULL | 248K tokens |

**并发=10 时的对比（最佳区间）：**

| max_num_seqs | RPS | TTFT mean | TTFT p95 | Latency mean |
|---:|---:|---:|---:|---:|
| 256 (默认) | **90.7** | 0.044s | 0.054s | 0.382s |
| 8 | 54.4 | 0.112s | 0.368s | 0.438s |

**并发=40 时：**

| max_num_seqs | RPS | TTFT mean | TTFT p95 | Latency mean |
|---:|---:|---:|---:|---:|
| 256 (默认) | 69.1 | 0.128s | 0.131s | 0.534s |
| 8 | 20.1 | 0.797s | 1.628s | 1.142s |

### 分析

1. **max_num_seqs 是吞吐量的天花板**
   - 设为 8 时，scheduler 最多同时处理 8 个序列
   - 即使有 40 个并发请求，也只分批处理（每批 ≤ 8 个）
   - 吞吐从 90.7 跌到 54.4（-40%），高并发时更惨（69→20，-71%）

2. **TTFT 受影响巨大**
   - 高并发时 TTFT 从 128ms 暴涨到 797ms
   - 因为请求在 waiting queue 里等前面的 batch 完成

3. **为什么不是设越大越好？**
   - `max_num_seqs` 决定了 CUDA Graph 捕获的最大 batch size
   - 设太大会导致：更多 CUDA Graph 内存占用、更长的 graph 捕获时间
   - 在 RTX 3060 上，默认 256 其实远超需要（最佳并发才 10）

4. **KV cache 略有增加**
   - 限制 max_num_seqs=8 后，CUDA Graph 更小，省出 ~20K tokens 的 KV cache 空间
   - 但这点增加远不够弥补并发能力的损失

> **结论：** `max_num_seqs` 应该 ≥ 你的预期峰值并发数。在 RTX 3060 上跑 1.5B 模型，默认 256 够用。如果你知道最大并发不会超过 16，可以设 16 来减少 CUDA Graph 内存。

---

## 五、实验 3：max_num_batched_tokens 的影响

> **这组实验想看什么**：验证 `max_num_batched_tokens` 如何通过 **Chunked Prefill** 影响 TTFT。选 **512 vs 8192** 这组对照，是因为 512 这个值**比我们测试用的 prompt 还短**——这样能强制触发 prefill 切块，把"长 prompt 被切成多步处理"的代价放大出来，从 TTFT 数据上直接看到影响。

**对比配置：**

| 配置 | max_num_batched_tokens | 编译范围 |
|---|---|---|
| 默认 | 8192 | (1, 8192) |
| 限制 | 512 | (1, 512) |

**并发=10 时的对比：**

| max_batched_tokens | RPS | TTFT mean | TTFT p95 | Latency mean |
|---:|---:|---:|---:|---:|
| 8192 (默认) | **90.7** | 0.044s | 0.054s | 0.382s |
| 512 | 83.4 | 0.043s | 0.094s | 0.390s |

**并发=5 时的对比（低并发影响更明显）：**

| max_batched_tokens | RPS | TTFT mean | TTFT p95 | Latency mean |
|---:|---:|---:|---:|---:|
| 8192 (默认) | 88.1 | 0.038s | 0.045s | 0.365s |
| 512 | 36.0 | 0.149s | 0.609s | 0.498s |

### 分析

1. **max_num_batched_tokens 控制 prefill chunking**
   - 设为 8192：一个 prompt 可以在一次 forward 中全部 prefill
   - 设为 512：长 prompt 被分成多个 512-token 的 chunk，逐步 prefill

2. **低并发时影响巨大**
   - 并发=5 时 RPS 从 88.1 跌到 36.0（-59%）
   - 因为我们的 medium prompt ~30 tokens，加上 system prompt 等可能超过 512 tokens
   - Prefill 被分块后，需要多个 step 才能完成，增加 TTFT

3. **高并发时差异缩小**
   - 并发=20-40 时，两者 RPS 接近（75 vs 73）
   - 因为高并发下每个请求的 prefill 本来就需要排队，分块影响被摊薄

4. **什么时候应该限制它？**
   - 当你的 prompt 非常长（>4K tokens）且并发也很高时
   - 限制 `max_num_batched_tokens` 可以防止一个长 prompt 的 prefill 占满整个 batch
   - 这就是 **Chunked Prefill** 的核心价值：长 prefill 不会阻塞短请求

> **结论：** 对于短 prompt 场景（<500 tokens），保持默认 8192 即可。对于长 prompt + 高并发场景，适当限制（如 2048）可以改善 TTFT 公平性。

---

## 六、综合调参建议

### RTX 3060 12GB + Qwen2.5-1.5B

```
最佳配置：
  --max-model-len 4096           # 平衡上下文长度和并发数
  --gpu-memory-utilization 0.85  # WSL2 桌面环境下安全值
  --max-num-seqs 256             # 默认值，足够
  --max-num-batched-tokens 8192  # 默认值，短 prompt 场景无需改
```

### 通用调参思路

```
                        ┌─────────────────────┐
                        │  显存预算有限？       │
                        └────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Yes                      │ No
                    ▼                          ▼
            降低 max_model_len          保持默认
            或 enforce_eager                 │
                    │                        │
                    ▼                        ▼
            ┌───────────────┐      ┌──────────────────┐
            │ 并发需求高？   │      │ prompt 很长？     │
            └───────┬───────┘      └────────┬─────────┘
                    │ Yes                   │ Yes
                    ▼                       ▼
              保持 max_num_seqs        降低 max_num_batched
              = 峰值并发数 × 1.5        _tokens (如 2048)
```

### 三者的关系

```
max_model_len ─→ 决定 KV cache 单请求大小
                  │
gpu_mem_util  ───→ 决定 KV cache pool 总大小 ─→ 决定最大并发容量
                  │
max_num_seqs  ───→ 限制同时处理的序列数 (≤ 并发容量)
                  │
max_num_batched_tokens ─→ 限制单 step 的 prefill + decode token 总数
                           → 影响 TTFT 和调度公平性
```

---

## 七、踩坑记录

### 坑 5：vLLM 服务器杀掉后 GPU 显存不释放

**现象：**

`fuser -k 8000/tcp` 杀掉 APIServer 进程后，`nvidia-smi` 显示显存仍然占满（~12GB）。

**原因：**

vLLM V1 引擎使用多进程架构：APIServer（父进程）+ EngineCore（子进程）。杀掉父进程时，子进程可能变成孤儿进程继续占用 GPU。

**解决方案：**

```bash
# 1. 杀掉端口上的进程
fuser -k 8000/tcp

# 2. 检查并杀掉残留的 EngineCore
ps aux | grep EngineCore | grep -v grep | awk '{print $2}' | xargs kill -9

# 3. 确认显存释放
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
# 应该回到 ~2700 (桌面占用)
```

### 坑 6：HF_HUB_OFFLINE 模式避免网络超时

**现象：**

启动 `vllm serve` 时卡在 `Error retrieving file list: [Errno 110] Connection timed out`。

**原因：**

vLLM 启动时会尝试连接 HuggingFace 验证模型版本。在代理环境下，即使模型已缓存，网络请求也可能超时。

**解决方案：**

```bash
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B ...
```

`HF_HUB_OFFLINE=1` 让 vLLM 直接使用本地缓存，跳过所有网络请求。模型已下载过一次后，加上这个参数可以避免网络问题。

---

## 八、核心概念速查表

全文出现的术语集中解释，方便回查：

| 术语 | 一句话解释 |
|---|---|
| **Prefill（预填充）** | 处理输入 prompt 的阶段，一次性把所有输入 token 算成 KV cache。计算密集，决定 TTFT。 |
| **Decode（解码）** | 逐个生成输出 token 的阶段。显存带宽密集，是 batching 提速的主战场。 |
| **KV cache** | 存放每个 token 注意力中间结果的缓存。避免重复计算，但占显存，是并发容量的瓶颈。 |
| **Static Batching** | 传统批处理：整批一起开始、等最慢的一起结束。短请求被长请求拖累。 |
| **Continuous Batching** | vLLM 的逐 step 调度：完成的请求立即移除、新请求随时加入。吞吐和延迟双赢。 |
| **PagedAttention** | vLLM 的核心技术：像操作系统管内存一样，把 KV cache 切成 page 按需分配，消除碎片。 |
| **Chunked Prefill** | 把长 prompt 的 prefill 切成多个小块分步处理，避免长请求霸占 GPU。 |
| **CUDA Graph** | 把一串 GPU 操作"录制"成一个可重放的图，减少 CPU 调度开销，加速 decode。 |
| **TTFT** | Time To First Token，首 token 延迟，体感"响应快不快"。 |
| **RPS** | Requests Per Second，吞吐量，衡量"系统能扛多少"。 |
| **p95** | 95 百分位：95% 的请求都比这个值快。比平均值更能反映真实体验。 |

---

## 附录：完整实验数据

### 实验 1：默认参数，不同并发数

| 并发 | RPS | TTFT mean | TTFT p95 | Lat mean | Lat max | Wall(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.85 | 0.061s | 0.108s | 0.420s | 1.387s | 1.39s |
| 5 | 88.14 | 0.038s | 0.045s | 0.365s | 0.454s | 0.45s |
| 10 | 90.72 | 0.044s | 0.054s | 0.382s | 0.441s | 0.44s |
| 20 | 73.87 | 0.071s | 0.078s | 0.460s | 0.541s | 0.54s |
| 40 | 69.06 | 0.128s | 0.131s | 0.534s | 0.579s | 0.58s |

### 实验 2：max_num_seqs=8，不同并发数

| 并发 | RPS | TTFT mean | TTFT p95 | Lat mean | Lat max | Wall(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 38.47 | 0.153s | 0.636s | 0.494s | 1.040s | 1.04s |
| 10 | 54.36 | 0.112s | 0.368s | 0.438s | 0.736s | 0.74s |
| 20 | 34.15 | 0.440s | 0.802s | 0.780s | 1.171s | 1.17s |
| 40 | 20.09 | 0.797s | 1.628s | 1.142s | 1.991s | 1.99s |

### 实验 3：max_num_batched_tokens=512，不同并发数

| 并发 | RPS | TTFT mean | TTFT p95 | Lat mean | Lat max | Wall(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 36.04 | 0.149s | 0.609s | 0.498s | 1.110s | 1.11s |
| 10 | 83.39 | 0.043s | 0.094s | 0.390s | 0.480s | 0.48s |
| 20 | 75.07 | 0.061s | 0.068s | 0.442s | 0.533s | 0.53s |
| 40 | 71.72 | 0.123s | 0.125s | 0.514s | 0.558s | 0.56s |
