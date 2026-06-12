---
title: SGLang-vLLM学习-05-01
date: 2026-05-01
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📅 今日任务

> 🎉 劳动节休息 — 如果有空，整理前两天笔记

- [ ] 整理前两天的学习笔记，补全遗漏内容
- [ ] 回顾 SGLang 基本概念和 RadixAttention 机制
- [ ] 记录学习中的疑问，待工作日深入

**预期产出：** 完善的笔记整理

## 📍 本周阶段

| 项目 | 内容 |
|------|------|
| **阶段** | M1: 基础认知 |
| **周次** | Week 2：SGLang 上手 + 初步对比 |
| **本周目标** | SGLang 跑通，体验 RadixAttention 效果 |
| **距分享剩余** | 65 天 |

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

### 假日轻量复习：10 张闪卡（遮住右边自测）

| 问题 | 答案要点 |
|------|---------|
| KV Cache 存的是什么 | 每层每 token 的 K、V 向量，避免重算历史 |
| Prefill vs Decode 瓶颈 | prefill 算力密集 / decode 访存密集 |
| PagedAttention 解决什么 | KV 显存碎片与预留浪费 → <4% |
| Block Table 作用 | 逻辑块号→物理块号的页表 |
| Continuous Batching | token 级调度，请求随完成随进出 |
| APC 复用粒度 | 整块对齐的前缀（块哈希） |
| RadixAttention 复用粒度 | 任意长度前缀（基数树） |
| RadixAttention 淘汰策略 | LRU，从叶子开始 |
| TTFT 由什么决定 | 排队 + prefill（与 decode 无关） |
| SGLang 三贡献 | RadixAttention / Compressed FSM / 前端 DSL |

### 还有 10 分钟的话
把 4/25 的内存生命周期图**凭记忆重画一遍**，对照原图找差异——画图回忆是最高效的巩固方式。
