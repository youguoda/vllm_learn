---
title: "SGLang vs vLLM 学习日志 - 04-22"
date: 2026-04-22
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

# 📚 SGLang vs vLLM 学习日志 — 4月22日 周三

> **距技术分享还有 74 天** | **阶段：M1: 基础认知** | **Week 1/10 | 学习第 1 天**

---

## 📍 本周阶段

**Week 1：环境搭建 + vLLM 上手**
> 本周目标：vLLM 能跑起来，理解 PagedAttention 核心思想

---

## ✅ 今日任务

- [x] 📖 读 vLLM 原始论文（PagedAttention），记核心笔记
- [x] 预期产出：论文阅读笔记

### 任务指引

今天的关键是**通读 PagedAttention 论文**，建立对 vLLM 核心创新的理解。建议关注：

1. **问题定义**：LLM 推理中 KV Cache 的内存浪费问题（60-80%）
2. **PagedAttention 方案**：借鉴 OS 虚拟内存的分页思想
3. **Block 概念**：KV Cache 如何被切分为固定大小的 block
4. **与 Transformer 原生 Attention 的区别**

### 推荐资源
- 论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention* (arXiv:2309.06180)
- [[论文笔记-PagedAttention-vLLM]] — 已有的论文笔记

---

## 📝 学习记录

### 核心概念理解
![[Pasted image 20260611002330.png|862]]



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
| 4/22 周三 | **📖 读 vLLM 原始论文（PagedAttention），记核心笔记** | **⬅️ 当天** |
| 4/23 周四 | 安装 vLLM，用 Qwen2.5-1.5B 跑通基础推理 | ☐ |
| 4/24 周五 | 体验 Continuous Batching，调参数 | ☐ |
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

### 一句话抓住论文
> KV Cache 像 OS 管理内存一样分页管理，把 60-80% 的显存浪费降到 <4%，吞吐量翻数倍。

### 问题：KV Cache 为什么浪费显存
推理时每个 token 在每一层都要缓存 K、V 两个向量（KV Cache）。传统做法（如 FasterTransformer）为每个请求**按 max_seq_len 预分配一整块连续显存**，导致三种浪费：

1. **预留浪费**：请求实际只生成 200 token，却预留了 2048 token 的空间
2. **内部碎片**：连续分配的块没用满
3. **外部碎片**：不同大小的连续块之间的空隙无法利用

论文实测：传统方案实际有效利用率仅 20%~40%。

### 方案：PagedAttention 三要素

| OS 概念 | vLLM 对应 | 说明 |
|---------|----------|------|
| 页（Page） | **Block**（默认 16 token 的 KV） | 固定大小的物理显存块 |
| 页表 | **Block Table** | 每个序列一张表，逻辑块号 → 物理块号 |
| 按需分页 | 按需分配 | 生成满 16 token 才分配下一块 |

关键收益：
- 浪费只发生在**每个序列的最后一个未满块**（平均浪费 < 块大小一半），整体 <4%
- 物理块不必连续 → 没有外部碎片
- **共享**：并行采样 / beam search 的多个分支可共享 prompt 的物理块，写时复制（Copy-on-Write）

### 必须能回答的自测题
1. block 大小为什么不是 1？（太小→block table 巨大、kernel 访存不连续；太大→内部碎片回升）
2. PagedAttention 的代价是什么？（attention kernel 要按 block table 间接寻址，访存不连续带来 ~5-10% kernel 开销——用吞吐换内存）
3. 显存省下来去了哪？（更大的 batch → 更高吞吐，这是"内存效率→吞吐量"的转化链）

### 关联
[[论文笔记-PagedAttention-vLLM]] · arXiv:2309.06180
