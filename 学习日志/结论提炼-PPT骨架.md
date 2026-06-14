---
title: 结论提炼 —— PPT 论点骨架（数据归类）
date: 2026-06-14
tags:
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - output/active
related:
  - "[[实验报告]]"
  - "[[选型结论]]"
---

# 结论提炼：PPT 论点骨架

> 把所有数据按"对外讲的场景"重组（[summarize_by_scenario.py](../summarize_by_scenario.py)），提炼成 PPT 可用的论点。

---

## 三句话核心论点（PPT 主线）

1. **KV Cache 管理（块 vs token）是两框架最大差异** —— 并行策略都是 Megatron 底座，不区分；差异在缓存粒度。
2. **复用率是分水岭** —— 无复用两者持平（vLLM 略优 8%），高复用 SGLang 吞吐 2.1x。
3. **选型法则** —— 通用/低复用 → vLLM（生态成熟）；高复用（统一 system prompt/RAG/Agent）→ SGLang（性能）。

---

## 按场景的数据卡片（PPT 每页一张）

### 卡片1：通用批量推理（无共享前缀）
- vLLM 最高 1443 tok/s，SGLang 1342 tok/s（**差 8%，持平**）
- 单并发 TTFT：vLLM 30ms / SGLang 27ms
- **结论**：两者持平，vLLM 略优 + 生态成熟 → 选 vLLM

### 卡片2：高前缀复用（统一 system prompt / RAG / Agent）
- 100% 复用率：vLLM RPS 8.88 vs SGLang **18.95（2.1x）**
- **结论**：SGLang RadixAttention 全局 token 级共享 → 选 SGLang

### 卡片3：延迟敏感（交互式）
- 单并发 TTFT：两者 ~30ms；ITL 11-12ms（无抖动）
- **结论**：延迟持平，由吞吐和复用率决定选型

### 卡片4：进阶特性（按需启用）
- ngram 投机（重复内容 4.5x）、AWQ 量化（省 64%）、FP8 KV（并发翻倍）、结构化输出（100% 合法）
- **结论**：两框架都支持，与选型正交

---

## 反方观点（PPT 防质疑预案）

| 可能的质疑 | 我的回应（带数据） |
|---|---|
| "SGLang 全面更快" | 否，无复用 vLLM 略优 8%（卡片1） |
| "vLLM 多轮对话差" | 否，vLLM APC 也缓存历史，TTFT 逐轮平稳（4.4 节） |
| "2.1x 普遍成立" | 否，是 100% 极端复用特例，社区 ShareGPT 是 29%（[[社区数据交叉验证]]） |
| "结论适用所有硬件" | 否，3060+1.5B 验证趋势，绝对数字需目标环境实测 |

---

## 今日产出

- [x] summarize_by_scenario.py（按场景重组数据）
- [x] 结论提炼-PPT骨架.md（三句话主线 + 4 卡片 + 防质疑预案）

## 一句话

> **PPT 主线三句话**：KV 管理是最大差异 → 复用率是分水岭 → 通用选 vLLM、高复用选 SGLang。每个论点都有本机数据支撑，且备好了防质疑预案。
