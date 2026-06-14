---
title: 分享 PPT 大纲与制作（vLLM vs SGLang）
date: 2026-06-14
tags:
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - output/milestone
related:
  - "[[实验报告]]"
  - "[[结论提炼-PPT骨架]]"
  - "[[M3阶段摘要]]"
---

# 分享 PPT 大纲与制作

> 用 python-pptx 代码生成（[make_ppt.py](../make_ppt.py)），数据/图表全部从实验文件引用，不手动填。输出 `assets/vllm_sglang_share.pptx`（16 张）。

---

## PPT 结构（16 张）

| # | 标题 | 核心内容 |
|---|---|---|
| 1 | 封面 | vLLM vs SGLang 深度对比 |
| 2 | 目录 | 5 个部分 |
| **Part 1：为什么需要推理框架** | | |
| 3 | GPU 利用率浪费 | decode 访存瓶颈，单请求 90 tok/s |
| 4 | KV Cache 核心瓶颈 | 传统浪费 98.8% |
| 5 | 传统内存浪费 | PagedAttention 动机 |
| **Part 2：vLLM 三层优化** | | |
| 6 | PagedAttention | 14268 块×16，并发 55.73x |
| 7 | Continuous Batching | 并发1→5 涨 3 倍 |
| 8 | APC | TTFT 降 4.6x，块粒度 |
| **Part 3：SGLang** | | |
| 9 | RadixAttention | 半块命中 94.9% vs 82.1% |
| 10 | 缓存感知调度+Compressed FSM | LPM/DFS_WEIGHT + 跳步 |
| **Part 4：对比数据** | | |
| 11 | 无前缀持平 | 1443 vs 1342 tok/s |
| 12 | 高复用 2.1x | RPS 8.88 vs 18.95（最关键） |
| 13 | 核心差异 | 命中粒度 1 vs 16 token |
| **Part 5：选型** | | |
| 14 | 选型建议 | 复用率决定选型 |
| 15 | 一句话结论 | 通用 vLLM / 高复用 SGLang |
| 16 | 谢谢 Q&A | 局限声明 |

---

## 制作原则

1. **数据从实验文件引用**：所有数字（2.1x、1443 tok/s、94.9%）来自本机实测，可追溯。
2. **图表用 assets/ 实测图**：嵌入 p1_throughput / prefix_ratio_comparison / kvcache_diff / 架构图等。
3. **三句话主线**：KV 管理是最大差异 → 复用率是分水岭 → 通用 vLLM/高复用 SGLang。
4. **生动类比**：PagedAttention=操作系统页分配器，RadixAttention=前缀树/查字典 vs 走迷宫。

---

## 叙事节奏（讲故事）

```
痛点(GPU摸鱼) → vLLM 三招(分页/批处理/缓存) → SGLang 更细(token级)
→ 数据(无复用持平 / 高复用 2.1x) → 选型(看负载特征)
```

> 高潮在第 12 张（高复用 2.1x）——前面铺垫"为什么 KV 管理重要"，到这里用数据引爆"复用率决定选型"的核心论点。

---

## 今日产出

- [x] PPT 结构确定（16 张）
- [x] make_ppt.py（代码生成，数据/图表引用实验文件）
- [x] assets/vllm_sglang_share.pptx（16 张，含封面到 Q&A）
- [x] Part 1（为什么需要框架）+ Part 2（vLLM）内容填充

## 一句话

> **16 张 PPT 代码生成，数据全部本机实测可追溯**。叙事主线：痛点→vLLM三招→SGLang更细→数据引爆(高复用2.1x)→选型法则。高潮在"复用率决定选型"。
