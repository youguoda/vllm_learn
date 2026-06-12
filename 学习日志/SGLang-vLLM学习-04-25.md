---
title: "SGLang vs vLLM 学习日志 - 04-25"
date: 2026-04-25
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

# 📚 SGLang vs vLLM 学习日志 — 4月25日 周六

> **距技术分享还有 71 天** | **阶段：M1 基础认知** | **Week 1/10**

---

## 📍 本周阶段

**Week 1：环境搭建 + vLLM 上手**
> 本周目标：vLLM 能跑起来，理解 PagedAttention 核心思想

---

## ✅ 今日任务

- [ ] 📖 读 vLLM 架构文档，画 PagedAttention 内存管理流程图
- [ ] 预期产出：架构图草稿

### 任务指引

今天的关键是**理解 PagedAttention 的内存管理机制**，建议关注：

1. **虚拟内存类比**：KV Cache 如何像 OS 虚拟内存一样分块管理
2. **Block Table 机制**：如何将逻辑上的连续 KV 映射到物理上不连续的 GPU 内存块
3. **内存浪费率**：从传统方法的 60-80% 降到 <4% 是怎么做到的
4. **画出流程图**：包括 prefill → block 分配 → decode → block 释放的完整流程

### 推荐资源

- vLLM 官方文档：https://docs.vllm.ai/
- 论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention* (arXiv:2309.06180)
- [[论文笔记-PagedAttention-vLLM]] — 已有的论文笔记

---

## 📝 学习记录

### 核心概念理解

*(在这里记录你对 PagedAttention 内存管理的理解)*

### 架构图草稿

*(在这里描述或贴上你画的 PagedAttention 内存管理流程图)*

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
| 4/24 周五 | 体验 Continuous Batching，调参数 | ☐ |
| **4/25 周六** | **📖 读 vLLM 架构文档，画 PagedAttention 内存管理流程图** | **⬅️ 今天** |
| 4/26 周日 | 体验 Prefix Caching（APC），对比性能 | ☐ |
| 4/28 周一 | 整理本周笔记，写"vLLM 初体验" | ☐ |

---

## 🔗 相关笔记

- [[SGLang与vLLM推理框架对比分享]] — 主项目计划
- [[SGLang-vs-vLLM-研究笔记]] — 研究资料汇总
- [[论文笔记-PagedAttention-vLLM]] — PagedAttention 论文笔记

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### 一个请求的完整内存生命周期（画图就画这个）

```
请求到达 "今天天气如何？"(7 token, block_size=4)
   │
   ├─ Prefill：一次算完 7 个 token 的 KV
   │   分配 2 个物理块: 逻辑块0→物理块7, 逻辑块1→物理块12
   │   Block Table: [7, 12]   (块12只用了3/4格)
   │
   ├─ Decode 第1步：生成 "今"，KV 写入块12第4格（块满）
   ├─ Decode 第2步：生成 "天"，按需分配新物理块3
   │   Block Table: [7, 12, 3]
   │   ...逐 token 重复，块满才分配新块...
   │
   └─ 遇到 EOS / 达到 max_tokens：
       整条序列的物理块全部归还空闲池（free list）
```

### 图上要标注的 4 个要点
1. **逻辑连续、物理离散**：Block Table 是翻译层，attention kernel 按表间接寻址
2. **按需分配**：永远只欠一个块，浪费 = 最后一个块的空格
3. **共享与 CoW**：beam search 多分支指向同一物理块，引用计数 >1 时写入先复制
4. **空闲池**：分配/释放都是 O(1) 的链表操作，没有碎片整理

### 显存预算心算公式
```
KV bytes = 2(K和V) × layers × kv_heads × head_dim × dtype字节 × tokens
例 Qwen2.5-7B (28层, 4 KV头 GQA, head_dim 128, fp16):
  每 token ≈ 2×28×4×128×2 B ≈ 57 KB → 1 万 token ≈ 0.57 GB
```
这解释了为什么 `--max-model-len` 直接决定能开多大并发。

### 自测题
1. 为什么传统方案的"外部碎片"在 PagedAttention 中消失了？（所有块同尺寸，任何空闲块都能用）
2. swap-out 换出到 CPU 时搬的是什么？（该序列 Block Table 指向的所有物理块内容）
