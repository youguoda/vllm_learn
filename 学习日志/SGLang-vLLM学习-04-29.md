---
title: SGLang与vLLM推理框架对比学习 - 04-29
date: 2026-04-29
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📌 今日任务

- [ ] 📖 读 SGLang 原始论文，重点看 RadixAttention 章节
- [ ] 整理论文阅读笔记，记录核心发现

## 📅 本周阶段

> **Week 2：SGLang 上手 + 初步对比**
> 阶段：M1 - 基础认知
> 本周目标：SGLang 跑通，体验 RadixAttention 效果

## ✍️ 学习记录

### 论文阅读笔记

**SGLang 论文核心要点：**

-

**RadixAttention 机制：**

-

**关键发现与思考：**

-

## 🔗 关联

[[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### SGLang 论文一句话
> LLM 程序（多轮、分支、结构化）天然存在大量**可复用前缀**和**可约束输出**，SGLang 用 RadixAttention 自动复用 KV、用压缩有限状态机加速约束解码，外加一套前端 DSL。

### RadixAttention 核心机制
1. **Radix Tree（基数树/压缩前缀树）**管理全部 KV Cache：树的边是 token 序列段，节点对应缓存的 KV 块
2. 新请求到达 → `match_prefix` 沿树匹配最长公共前缀 → 命中部分直接复用，只 prefill 剩余部分
3. 请求结束后 KV **不立即释放**，留在树上等待未来命中；显存不够时 **LRU 淘汰**叶子节点
4. **Cache-aware 调度**：等待队列按"前缀命中长度"排序，命中多的先跑，提高整体命中率

与 vLLM APC 对比记三点：**自动（无需整块对齐）、树形（支持多分支共享）、调度感知（队列重排）**。

### 论文另外两个贡献（别漏）
- **Compressed FSM**：结构化输出时把"确定性的 token 段"一次跳过（jump-forward），而不是逐 token 走 mask（5/22-23 会深入）
- **前端 DSL**：`sgl.gen()` / fork-join 原语，把多调用程序的依赖图交给运行时优化

### 自测题
1. 多轮对话场景为什么 SGLang 天然占优？（每轮的历史就是下一轮的前缀，radix 树自动命中，无需客户端做任何事）
2. radix 树的淘汰为什么从叶子开始？（内部节点是多个序列的公共前缀，价值更高）

### 关联
[[SGLang-vs-vLLM-研究笔记]] · arXiv:2312.07104
