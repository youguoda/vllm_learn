---
title: 论文笔记：Efficient Memory Management for Large Language Model Serving with PagedAttention
date: 2026-04-22
updated: 2026-06-06
tags:
  - R/技术框架/vLLM
  - R/核心算法/LLM推理
  - R/核心算法/注意力机制
  - R/技术框架/推理优化
  - reading/paper
status: active
aliases:
  - vLLM PagedAttention论文笔记
  - PagedAttention论文
  - Efficient Memory Management for Large Language Model Serving with PagedAttention
source: https://arxiv.org/abs/2309.06180
doi: 10.1145/3600006.3613165
related:
  - "[[LLM内功-KV Cache]]"
  - "[[LLM内功-Paged Attention]]"
  - "[[SGLang-vs-vLLM-研究笔记]]"
  - "[[SGLang与vLLM推理框架对比分享]]"
  - "[[vLLM]]"
---

# 论文笔记：Efficient Memory Management for Large Language Model Serving with PagedAttention

> [!info] 论文信息
> - **作者：** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
> - **机构：** UC Berkeley, Stanford University, UC San Diego, Independent Researcher
> - **会议：** SOSP 2023
> - **arXiv：** [2309.06180](https://arxiv.org/abs/2309.06180)
> - **代码：** [vllm-project/vllm](https://github.com/vllm-project/vllm)
> - **一句话：** 把操作系统虚拟内存分页思想搬到 LLM KV Cache 管理里，用非连续 block 存储、按需分配和 Copy-on-Write，把显存浪费变成可控的小碎片。

## 0. 核心结论

PagedAttention 不是在改模型结构，也不是在减少 attention 的理论计算量，而是在解决 **LLM serving 的 KV Cache 显存管理问题**。

论文的主张很清楚：

1. LLM 推理吞吐的关键瓶颈之一是 GPU memory，不只是算力。
2. KV Cache 巨大、动态增长、长度未知，传统连续分配会造成严重浪费。
3. 将 KV Cache 切成固定大小 block，并用 block table 做逻辑块到物理块映射，可以消除外部碎片，并把内部碎片限制在最后一个 block。
4. block 级共享让 parallel sampling、beam search、shared prefix 这类场景可以少复制 KV Cache。
5. vLLM 在同等延迟水平下，相比 Orca / FasterTransformer 可提升 2-4x 吞吐；在部分设置下提升更高。

> [!important] 记忆句
> **PagedAttention = OS paging for KV Cache。** Token 像 byte，KV block 像 page，请求像 process，block table 像 page table。

## 1. 问题背景：为什么 KV Cache 会卡住吞吐？

自回归 LLM 生成时，每一步只生成一个 token。为了避免重复计算历史 token 的 K/V，会缓存每个 token 的 Key 和 Value，也就是 [[LLM内功-KV Cache|KV Cache]]。

服务端想提高吞吐，需要把多个请求 batch 起来。但 batch size 被显存限制，而显存里除了模型参数，最大动态部分就是 KV Cache。

论文给出的 A100 40GB + 13B 模型例子：

| 显存部分     |       占比 | 特点                  |
| -------- | -------: | ------------------- |
| 模型参数     |    约 65% | 常驻，不随请求变化           |
| KV Cache | 约 30% 以上 | 每个请求独立，随输入/输出长度动态增长 |
| 激活等其他内存  |       少量 | 临时占用                |

对于 OPT-13B，单个 token 的 KV Cache 约为：

```text
2 (K/V) x 5120 (hidden size) x 40 (layers) x 2 bytes = 800 KB/token
```

如果最大序列长度是 2048，一个请求的 KV Cache 最多约 1.6GB。几十 GB 显存看起来很多，但并发请求一上来，KV Cache 很快成为瓶颈。

## 2. 传统系统的三类内存浪费

Orca、FasterTransformer 这类系统通常把一个请求的 KV Cache 存成连续 tensor。问题是：请求最终会生成多长，服务端事先不知道。

| 浪费类型 | 原因 | 结果 |
|---|---|---|
| Reserved slots | 为未来 token 预留位置 | 当前还没用，但别人也不能用 |
| Internal fragmentation | 按最大长度或预测长度分配，实际生成更短 | 请求结束后发现很多位置从未用过 |
| External fragmentation | 不同大小连续块分配/释放后形成碎片 | 总空闲显存够，但找不到连续空间 |

论文的 profiling 结果很扎心：传统系统里真正存放有效 token state 的 KV Cache 只占 **20.4% - 38.2%**。换句话说，很多显存被“占着但没干活”。

> [!note] 这里的关键
> Fine-grained batching / iteration-level scheduling 解决的是 batch 中请求长度不齐导致的计算浪费；PagedAttention 解决的是 KV Cache 显存布局和分配浪费。两者互补，不是同一个问题。

## 3. PagedAttention 的核心机制

### 3.1 Block 化 KV Cache

PagedAttention 把每个 sequence 的 KV Cache 划分成固定大小的 KV blocks。逻辑上，请求的 token 仍然是连续序列；物理上，对应的 KV blocks 可以散落在 GPU memory 的任意位置。

```text
逻辑 KV blocks:
Request A: [logical 0][logical 1][logical 2]

Block table:
logical 0 -> physical 7
logical 1 -> physical 1
logical 2 -> physical 3

物理显存:
[block 0][block 1][block 2][block 3] ... [block 7]
           ^A1               ^A2          ^A0
```

这样做带来三个直接收益：

- 不需要一开始预留最大长度，只给 prompt 和当前生成需要的 block。
- 物理 block 固定大小，因此不用寻找连续大块，外部碎片基本消失。
- 只可能浪费最后一个 block 的未填满位置，内部碎片被限制在 `block_size - 1` 个 token 以内。

### 3.2 Block Table

每个请求维护一张 block table，记录：

| 字段 | 含义 |
|---|---|
| logical block id | 请求内部的逻辑块编号 |
| physical block id | GPU 上实际 block 编号 |
| filled positions | 该 block 已写入多少 token 的 KV |

生成新 token 时，如果最后一个 logical block 还有空位，就写进去；如果满了，KV Cache Manager 再分配一个新的 physical block，并更新 block table。

### 3.3 分块 Attention 计算

标准 attention 对所有历史 K/V 做 attention。PagedAttention 的变化是：K/V 按 block 取出，attention kernel 根据 block table 去非连续物理内存里读 KV。

设 block size 为 `B`：

```text
K_j = (k_{(j-1)B+1}, ..., k_{jB})
V_j = (v_{(j-1)B+1}, ..., v_{jB})
```

Attention 输出变成对多个 KV block 的分块计算和累加。模型语义不变，只是 KV Cache 的存储布局和读取方式变了。

> [!tip] 默认 block size
> 论文实验发现 block size 太小会降低 GPU 读取和处理 KV 的并行度，太大又会增加内部碎片并降低共享概率。vLLM 默认 block size 设为 **16 tokens**。

## 4. vLLM 系统设计

vLLM 架构核心由几部分组成：

| 组件 | 职责 |
|---|---|
| Centralized Scheduler | 每个 iteration 选择哪些请求进入 batch |
| KV Cache Manager | 管理 physical block 池、分配/回收 block |
| Block Table | 每个请求的 logical block 到 physical block 映射 |
| GPU Worker | 执行模型 shard，按 block table 读取 KV Cache |
| Cache Engine | 在 GPU/CPU 上管理 block 存储与拷贝 |

每个 decoding iteration 大致流程：

1. Scheduler 选择本轮要运行的 sequence。
2. KV Cache Manager 为新 token 可能需要的新 logical block 分配 physical block。
3. Scheduler 将 input token ids 和 block table 广播给 GPU workers。
4. Workers 执行模型，在 attention 层根据 block table 读取非连续 KV blocks。
5. 本轮采样出的 token 回到 scheduler，进入下一轮。

分布式方面，vLLM 支持 Megatron-LM 风格 Tensor Parallelism。每个 worker 存自己 attention heads 对应的 KV shard，但共享同一套 logical-to-physical block 映射；worker 之间同步计算中间结果，而不是同步内存管理状态。

## 5. KV Cache 共享：论文里最容易被低估的部分

PagedAttention 不只是省碎片，还把 KV Cache 的共享粒度降到 block。

### 5.1 Parallel Sampling

同一个 prompt 采样多个输出时，prompt 部分 KV Cache 完全相同，可以共享 physical blocks。

当不同 sample 开始写入不同输出 token 时，如果要写的 block 被多个 sequence 引用，就触发 Copy-on-Write：

1. 检查 physical block 的 ref count。
2. 如果 `ref_count > 1`，分配新 block。
3. 拷贝旧 block 内容到新 block。
4. 当前 sequence 写新 block，旧 block ref count 减 1。

论文结果：

| 数据集 | Parallel sampling 共享节省 |
|---|---:|
| Alpaca | 6.1% - 9.8% |
| ShareGPT | 16.2% - 30.5% |

### 5.2 Beam Search

Beam search 的共享更强，因为不同 beam candidates 不只共享 prompt，还可能共享中间生成前缀。传统系统常需要在 beam 之间复制大段 KV Cache；vLLM 只共享 physical blocks，并在写旧共享 block 时 CoW。

论文结果：

| 数据集 | Beam search 共享节省 |
|---|---:|
| Alpaca | 37.6% - 55.2% |
| ShareGPT | 44.3% - 66.3% |

### 5.3 Shared Prefix

服务端可以预先缓存常见 shared prefix 的 KV Cache，例如 system prompt、few-shot examples。新请求如果带同样前缀，就把逻辑块映射到这些缓存好的 physical blocks，只对用户输入部分执行 prompt phase。

论文的翻译任务实验：

| 场景 | 相比 Orca (Oracle) 吞吐提升 |
|---|---:|
| 1-shot shared prefix | 1.67x |
| 5-shot shared prefix | 3.58x |

> [!warning] 边界
> 原论文确实提到 across requests 的 shared prefix，但它依赖服务提供方预定义/缓存公共前缀，并不是后来 SGLang RadixAttention 那种自动维护 radix tree、跨请求做通用前缀匹配的机制。

## 6. 调度与抢占

vLLM 使用 FCFS 保证公平性。请求太多、GPU physical blocks 不够时，需要抢占。

抢占策略：

- **All-or-nothing eviction：** 一个 sequence 的 block 要么全保留，要么全驱逐。
- **Gang scheduling：** 同一个 request 里的多个 sequences，例如 beam candidates，因为可能共享 KV blocks，需要一起抢占或一起恢复。
- **FCFS fairness：** 早到请求优先，晚到请求优先被抢占。

恢复方式有两种：

| 方式 | 做法 | 适合情况 |
|---|---|---|
| Swapping | 把被驱逐 blocks 拷到 CPU RAM，需要时再拷回 GPU | block size 较大时更有利 |
| Recomputation | 把 prompt + 已生成 token 拼起来，重新做一次 prompt phase 重算 KV | block size 较小时更有利 |

论文结论：小 block size 下 swapping 会产生大量小 CPU-GPU 传输，PCIe 有效带宽低；block size 16-64 时，两者端到端性能接近。

## 7. GPU Kernel 与实现细节

PagedAttention 的代价是 attention kernel 不能再假设 KV Cache 连续。因此 vLLM 做了定制 kernel：

| Kernel 优化 | 目的 |
|---|---|
| Fused reshape and block write | 新 KV Cache 分块、reshape、写入 block table 指定位置 |
| Fusing block read and attention | attention 时按 block table 读取 KV，并直接计算 |
| Fused block copy | CoW 时批量复制不连续 block，避免大量小 `cudaMemcpyAsync` |

论文 microbenchmark 显示，PagedAttention kernel 因为 block table 访问、额外分支、变长处理，attention kernel latency 比 FasterTransformer 高 **20% - 26%**。但端到端 serving 仍显著更快，因为它换来了更大的 batch 和更高显存利用率。

实现规模：

- Python：约 8.5K 行，主要是 scheduler / block manager 等控制逻辑。
- C++/CUDA：约 2K 行，主要是 PagedAttention 等核心 kernel。
- 前端兼容 OpenAI API，允许每个请求设置 sampling 参数、最大长度、beam width 等。

## 8. 实验结果怎么记

### 8.1 基础采样

在 ShareGPT 数据集上，vLLM 在相似 latency 下可承受：

| 对比对象 | 请求率提升 |
|---|---:|
| Orca (Oracle) | 1.7x - 2.7x |
| Orca (Max) | 2.7x - 8x |
| FasterTransformer | 最高 22x |

论文摘要中总体表述是：相比 state-of-the-art systems，vLLM 吞吐提升 **2-4x**。

### 8.2 Batch size 背后的解释

vLLM 的吞吐提升不是因为单步 attention 更快，而是因为显存管理更高效，可以同时容纳更多请求进入 batch。

论文中 OPT-13B + ShareGPT 的例子：vLLM 同时处理请求数比 Orca (Oracle) 多 **2.2x**，比 Orca (Max) 多 **4.3x**。

### 8.3 Chatbot 场景

论文用 ShareGPT 合成聊天历史，并限制 OPT-13B 的 prompt / output 长度各 1024 tokens。vLLM 在 chatbot workload 上相比 Orca baselines 可支撑约 **2x** 请求率。

## 9. 和 Orca / SGLang 的关系

### 9.1 vLLM vs Orca

Orca 的 iteration-level scheduling 解决的是 **请求间细粒度调度**，避免整批等待和 padding 浪费；vLLM 的 PagedAttention 解决的是 **KV Cache 内存管理**。

论文也明确说二者是 complementary：Orca 式调度让 LLM serving 更灵活，PagedAttention 让 KV Cache 容量能装下更多并发序列。

### 9.2 vLLM vs SGLang

| 维度 | vLLM / PagedAttention | SGLang / RadixAttention |
|---|---|---|
| 核心问题 | KV Cache 显存碎片和按需分配 | 跨请求公共前缀自动复用 |
| 数据结构 | Block table | Radix tree |
| 共享粒度 | KV block | 前缀路径上的 KV blocks |
| 原论文 shared prefix | 服务端预定义共享前缀 | 自动发现/维护公共前缀 |
| 适合场景 | 通用 serving、长输出、多采样、beam search | 多轮对话、RAG、agent、prompt 复用密集场景 |

> [!tip] 对外讲法
> PagedAttention 解决“显存被浪费”的问题；RadixAttention 进一步解决“重复前缀没有被复用”的问题。它们是递进关系，不是简单替代关系。

## 10. 局限与读后思考

1. **Kernel 有额外开销。** 非连续 KV 访问需要 block table indirection，因此单个 attention kernel 不一定更快。
2. **Block size 是工程权衡。** 小 block 减少碎片但降低并行度；大 block 提升读取效率但增加内部碎片和降低共享概率。
3. **共享前缀能力有限。** 原论文支持 predefined shared prefixes，不等于自动跨请求 prefix cache。
4. **CPU 端调度复杂度增加。** Scheduler、block table、ref count、CoW、preemption 都会引入控制面复杂度。
5. **论文实验是 2023 年系统状态。** 现代 vLLM 已加入 chunked prefill、prefix caching、spec decode、更多 kernel 和分布式能力，不能把论文当成当前 vLLM 全貌。

## 11. 工作场景联想

如果在 [[vLLM|天数智芯 vLLM / IxServer]] 做性能测试，PagedAttention 这篇论文能帮我建立几个判断：

- **TTFT 高**：更多看 prefill、排队、chunked prefill、prefix cache、PD 分离中的 KV 传输。
- **TPOT 高**：更多看 decode batch、attention kernel、KV Cache 读取效率、GPU 利用率。
- **QPS 上不去但单请求延迟正常**：检查 batch size 是否被 KV Cache 容量、max_num_seqs、max_model_len、gpu_memory_utilization 限住。
- **高并发 OOM 或频繁抢占**：关注 block 分配、swap/recompute、长尾输出长度、并发请求长度分布。
- **多 sample / beam workload**：PagedAttention 的 CoW 和 block sharing 应该明显降低 KV Cache 占用。

## 12. 复习卡片

**Q：PagedAttention 解决什么问题？**  
A：解决 LLM serving 中 KV Cache 动态增长、长度未知、连续分配导致的显存浪费和共享困难。

**Q：为什么它能减少浪费？**  
A：固定大小 block 按需分配；物理 block 不要求连续；每个请求通过 block table 映射逻辑块到物理块；浪费最多出现在最后一个未填满 block。

**Q：它会改变模型输出吗？**  
A：不会。PagedAttention 改的是 KV Cache 存储与访问方式，不改模型结构和数学语义。

**Q：Copy-on-Write 用在哪里？**  
A：parallel sampling、beam search、shared prefix 等多个 sequence/request 共享 physical block 时，只有写入共享 block 才复制。

**Q：为什么 vLLM 吞吐更高？**  
A：不是因为单个 attention kernel 一定更快，而是因为显存利用率高，可以容纳更大的有效 batch。

**Q：默认 block size 为什么是 16？**  
A：论文发现 16 足够利用 GPU 并行度，同时又能避免明显内部碎片，适合多数 workload。

## 13. 一句话总结

> PagedAttention 把 KV Cache 从“每个请求一整段连续显存”变成“按需分配的固定 KV blocks + block table 映射”，用一点 kernel indirection 换来近零显存浪费、更大 batch、block 级共享和 2-4x serving 吞吐提升。
