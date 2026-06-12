---
title: SGLang与vLLM学习日志 - 05月16日
date: 2026-05-16
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📌 今日任务

- [ ] 学习 SGLang 的 Expert Parallelism（DeepEP）机制
- [ ] 学习 SGLang 的 Data Parallelism 策略
- [ ] 对比 EP 和 DP 的调度策略与并行策略差异
- [ ] 整理 EP/DP 学习笔记

## 🎯 本周阶段 & 目标

> **阶段：** M2 - 架构深入
> **本周：** Week 4 — 架构深入：调度与并行策略
> **目标：** 理解两者的调度策略和并行策略差异

## 📝 学习记录

### Expert Parallelism（DeepEP）

*(在此记录 DeepEP 的核心概念、架构设计、通信模式等)*

### Data Parallelism

*(在此记录 SGLang DP 的实现方式、调度策略、与其他框架对比等)*

### EP vs DP 对比分析

*(在此记录两者在调度策略和并行策略上的关键差异)*

## 🔗 关联笔记

- [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### EP（专家并行）：MoE 时代的主角
MoE 模型（如 DeepSeek-V3：每层 256 个专家、每 token 激活 8 个）专家总量放不进单卡 → **把专家分到不同卡**。

```
token 在卡0 算 attention → router 选中专家 #57(卡2)、#103(卡5)
 → all-to-all 把 token 发到专家所在卡 → 专家计算
 → all-to-all 发回来 → 继续下一层
```
瓶颈在 **all-to-all 通信**。SGLang 集成 DeepSeek 开源的 **DeepEP**：定制 all-to-all kernel（NVLink/RDMA 分层、低精度传输、计算通信重叠），是它在 DeepSeek 系模型上领先的关键一环。

### DP（数据并行）与 MLA 的特殊玩法
普通 DP = 起 N 个完整副本分流量（吞吐线性扩，无通信税）。
SGLang 的 **DP Attention**（`--dp-size`）针对 MLA 架构（DeepSeek）：MLA 的 KV 压缩成 latent，TP 切不动 → attention 部分按请求做 DP、MoE 部分做 EP，混合编排。

### 决策树（写进分享很加分）
```
模型是 MoE？
 ├─ 是 → EP（+DeepEP），attention 视架构选 TP 或 DP-MLA
 └─ 否 → 单卡放得下？
      ├─ 放得下 → 多副本 DP 扩吞吐
      └─ 放不下 → TP（节点内）→ 还不够 → +PP（明天）
```

### 自测题
1. EP 和 TP 切的东西有何不同？（TP 切**每个**矩阵的内部；EP 按**整个专家**为单位分布）
2. 为什么 DP 没有通信税却不总是答案？（每副本都要一份完整权重+KV，显存成本×N）
