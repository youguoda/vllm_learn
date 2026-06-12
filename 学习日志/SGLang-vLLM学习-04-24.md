---
title: "SGLang vs vLLM 学习日志 - 04-24"
date: 2026-04-24
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/核心算法/LLM推理
  - R/AI工具/vLLM
status: in-progress
related:
  - "[[SGLang与vLLM推理框架对比分享]]"
---

# 📚 SGLang vs vLLM 学习日志 — 4月24日 周五

> **距技术分享还有 72 天** | **阶段：M1: 基础认知** | **Week 1/10 | 学习第 3 天**

---

## 📍 本周阶段

**Week 1：环境搭建 + vLLM 上手**
> 本周目标：vLLM 能跑起来，理解 PagedAttention 核心思想

---

## ✅ 今日任务

- [ ] 🔧 体验 vLLM 的 Continuous Batching，调参数（max_num_seqs, gpu_memory_utilization）
- [ ] 预期产出：参数调优笔记

### 任务指引

今天的关键是**体验 Continuous Batching 的效果**。建议关注：

1. **Continuous Batching 原理**：与 static batching 的区别，为什么能提升吞吐
2. **关键调优参数**：
   - `max_num_seqs`：最大并发序列数
   - `gpu_memory_utilization`：GPU 显存利用率阈值（默认 0.9）
   - `max_num_batched_tokens`：单 batch 最大 token 数
3. **观察指标**：吞吐量(TPS)、首token延迟(TTFT)、显存占用
4. **实验方法**：用不同并发数发请求，观察参数变化对性能的影响

### 实验思路
```bash
# 对比不同并发下的表现
for concurrency in 1 10 50; do
  # 发送并发请求并记录 throughput/latency
done
```

---

## 📝 学习记录

### 核心概念理解

*(在这里记录你的理解)*

### 实践操作记录

*(记录安装步骤、命令、参数等)*

### 遇到的问题

*(记录学习过程中的疑问)*

### 今日心得

*(一句话总结今天的收获)*

---

## 📅 本周进度一览

| 日期 | 任务 | 状态 |
|------|------|------|
| 4/22 周三 | 读 vLLM 原始论文（PagedAttention），记核心笔记 | ☐ |
| 4/23 周四 | 安装 vLLM，用 Qwen2.5-1.5B 跑通基础推理 | ☐ |
| 4/24 周五 | **🔧 体验 vLLM 的 Continuous Batching，调参数（max_num_seqs, gpu_memory_utilization）** | **⬅️ 当天** |
| 4/25 周六 | 读 vLLM 架构文档，画 PagedAttention 内存管理流程图 | ☐ |
| 4/26 周日 | 体验 Prefix Caching（APC），对比性能 | ☐ |
| 4/28 周一 | 整理本周笔记，写"vLLM 初体验" | ☐ |

---

## 🔗 相关笔记

- [[SGLang与vLLM推理框架对比分享]] — 主项目计划
- [[SGLang-vs-vLLM-研究笔记]] — 研究资料汇总
- [[论文笔记-PagedAttention-vLLM]] — PagedAttention 论文笔记

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### Continuous Batching 是什么
**Static batching**：8 个请求一起进、全部生成完才一起出 → 短请求等长请求，GPU 大量空转。
**Continuous（iteration-level）batching**：调度以"一步生成"为粒度——每生成一个 token 后，完成的请求立刻退出、等待的请求立刻插入。batch 是流动的。

这是 Orca 论文提出、vLLM 发扬的机制，**吞吐收益通常比 PagedAttention 本身还大**。两者关系：PagedAttention 解决"显存装得下更多并发"，continuous batching 解决"装下的并发不空转"。

### 三个关键参数

| 参数 | 默认 | 作用 | 调大的代价 |
|------|------|------|-----------|
| `--max-num-seqs` | 256 | 同时在跑的序列上限 | 单请求延迟上升 |
| `--max-num-batched-tokens` | ~2048(chunked prefill) | 每步最多处理的 token 数 | TTFT 与 ITL 互相挤压 |
| `--gpu-memory-utilization` | 0.9 | KV 块池大小 | 留给系统的余量变小 |

### 动手实验（10 分钟版）
```bash
# 官方压测工具，直接给 TTFT/TPOT/Throughput
vllm bench serve --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-prompts 200 --request-rate 8
# 改 --request-rate 1/8/32 各跑一次，对比指标变化
```
观察规律：并发↑ → 吞吐↑但边际递减 → KV 池满后请求排队，TTFT 飙升。**吞吐饱和拐点就是这张卡的服务容量**。

### 自测题
1. 为什么 continuous batching 能共存不同长度的请求？（每步只算"当前活跃序列的下一个 token"，序列独立进出）
2. KV 池满了 vLLM 怎么办？（抢占 preemption：换出或重算，见 5/13 调度器源码日）
