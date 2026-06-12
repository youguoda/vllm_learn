---
title: SGLang-vLLM学习-05-15
date: 2026-05-15
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📅 今日任务

> 📖 对比两者的 Tensor Parallelism 实现

- [ ] 阅读源码理解 vLLM 的 TP 实现方式（tensor 拆分策略、通信原语）
- [ ] 阅读源码理解 SGLang 的 TP 实现方式
- [ ] 对比两者在 TP 初始化、权重分发、All-Reduce 通信上的差异
- [ ] 记录关键差异点和各自的设计取舍

**预期产出：** TP 对比笔记

## 📍 本周阶段

| 项目 | 内容 |
|------|------|
| **阶段** | M2: 架构深入 |
| **周次** | Week 4：架构深入 — 调度与并行策略 |
| **本周目标** | 理解两者的调度策略和并行策略差异 |
| **距分享剩余** | 51 天 |

## 📝 学习记录

%% 在下方记录今天的学习内容 %%

### 核心笔记

-

### 疑问与待查

-

### 心得体会

-

## 🔗 关联

- [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### TP 的本质：把矩阵乘切开，用通信换显存
Megatron 式切法（两框架都沿用）：

```
Attention: QKV投影按"头"切到各卡（列并行）→ 各卡算各自的头
           输出投影行并行 → all-reduce 求和
MLP:       up_proj 列并行 → 激活 → down_proj 行并行 → all-reduce
⇒ 每个 transformer 层固定 2 次 all-reduce（前向）
```

记忆法：**列切不通信、行切要求和**。TP=4 时每卡只放 1/4 权重和 1/4 KV 头。

### 两框架的实现对照
| | vLLM | SGLang |
|--|------|--------|
| 参数 | `--tensor-parallel-size N` | `--tp-size N` |
| 线性层 | `ColumnParallelLinear/RowParallelLinear` | 同构思想，部分复用 vLLM 算子 |
| 通信 | NCCL + 自研 custom all-reduce（小消息低延迟） | NCCL，亦有 custom 优化 |
| 权重加载 | 各 rank 只加载自己分片 | 同 |

实践提醒：TP 内通信走 NVLink 才划算，跨 PCIe/跨机 TP 性能急剧下降——**TP 不出节点**是经验法则。

### 何时用 TP
1. 单卡放不下权重（必须用）
2. 想降低单请求延迟（多卡分担计算，但有通信税）
3. 只想提吞吐且单卡放得下 → **多副本数据并行通常优于 TP**

### 自测题
1. 为什么 KV Cache 也随 TP 切分？（K/V 头被分到各卡，各存各的）
2. TP=2 吞吐为什么到不了 2 倍？（all-reduce 通信 + 切分后矩阵变小、kernel 效率降）
