---
title: SGLang与vLLM学习日志 - 05月13日
date: 2026-05-13
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

# 📅 5月13日 周三 — SGLang & vLLM 学习日志

## 📍 阶段定位

- **里程碑**：M2: 架构深入
- **本周**：Week 4 — 架构深入 · 调度与并行策略
- **本周目标**：理解两者的调度策略和并行策略差异
- **距分享**：剩余 53 天

---

## ✅ 今日任务

- [ ] 📖 阅读 vLLM 调度器源码 `vllm/core/scheduler.py`
- [ ] 🧠 理解 continuous batching 完整流程（请求接入 → 调度 → 执行 → 换出）
- [ ] 📝 整理调度器核心逻辑笔记（调度决策、优先级策略、preemption 机制）

**预期产出**：调度器笔记

---

## 📝 学习记录

### vLLM Scheduler 核心流程

> *在这里记录你的理解……*

### Continuous Batching 关键机制

> *在这里记录你的理解……*

### 与 SGLang 调度策略对比思考

> *在这里记录你的理解……*

---

## 🔗 关联笔记

- [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### vLLM 调度器：每一步在决定什么
`Scheduler.schedule()` 每个引擎步运行一次，本质回答："这一步的 token 预算分给谁？"

```
三个队列：
waiting  ─ 新请求（还没分到 KV 块）
running  ─ 正在 decode 的请求
swapped  ─ 被抢占换出到 CPU 的请求

每步决策（简化）：
1. 先保 running：每个 decode 请求要 1 个新 slot，不够 → 触发抢占
2. 抢占 preemption 两种方式：
   - RECOMPUTE：直接丢 KV，回 waiting 重算（短序列便宜）
   - SWAP：块拷到 CPU，进 swapped（长序列划算）
3. 余下预算按 FCFS 接纳 waiting 的新请求（受 max_num_seqs/
   max_num_batched_tokens 限制）
```

### Chunked Prefill（必须懂的现代特性）
长 prompt 的 prefill 一次占满整步预算 → 所有 decode 卡顿（ITL 尖刺）。Chunked prefill 把 prefill 切成片，与 decode **混在同一步**执行：`--max-num-batched-tokens` 越小，交互越流畅、总吞吐略降——这是"吞吐 vs 流畅"旋钮的实现原理。

### 与 SGLang 调度的对照
| | vLLM | SGLang |
|--|------|--------|
| 接纳顺序 | 近似 FCFS | 按前缀命中长度优先 |
| 显存告急 | 抢占运行中请求 | 先 LRU 淘汰树上闲置前缀 |
| 优化目标 | 公平+吞吐 | 命中率+吞吐 |

### 读码锚点
`vllm/core/scheduler.py` 搜 `_schedule_running` / `_preempt`；跑一条超长 prompt + 高并发，日志里能看到 preemption 计数。

### 自测题
1. 什么时候 RECOMPUTE 比 SWAP 好？（序列短、PCIe 带宽紧张时——重算比搬运快）
2. ITL 突然出现尖刺，先查什么？（是否有长 prefill 挤进来 + chunked prefill 是否开启）
