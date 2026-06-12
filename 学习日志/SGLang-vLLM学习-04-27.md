---
title: "SGLang vs vLLM 学习日志 - 04-27"
date: 2026-04-27
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

# 📚 SGLang vs vLLM 学习日志 — 4月27日 周一

> **距技术分享还有 69 天** | **阶段：M1 基础认知** | **Week 1/10**

---

## 📍 本周阶段

**Week 1：环境搭建 + vLLM 上手**
> 本周目标：vLLM 能跑起来，理解 PagedAttention 核心思想

---

## ✅ 今日任务

- [ ] 💡 自由安排 / 补进度日
- [ ] 📝 补充前几天未完成的笔记
- [ ] 预期产出：补齐学习笔记，梳理本周知识脉络

### 任务指引

今天是**自由安排日**，用来补进度和消化之前的内容。建议：

1. **回顾本周笔记**：检查 4/22 ~ 4/26 的学习记录，标记未完成项
2. **补充笔记**：把之前只记了要点的地方展开，特别是 PagedAttention 核心机制
3. **查漏补缺**：如果有环境搭建或代码实验没跑通的，今天抓紧补上
4. **整理思路**：用自己的话总结 PagedAttention 的核心思想，为周三 Week 2 做准备

---

## 📝 学习记录

### 今日补充内容

*(在这里记录今天补充的笔记)*

### 知识梳理

*(梳理本周学到的关键知识点)*

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
| 4/25 周六 | 📖 读 vLLM 架构文档，画 PagedAttention 内存管理流程图 | ☐ |
| 4/26 周日 | 体验 Prefix Caching（APC），对比性能 | ☐ |
| **4/27 周一** | **💡 自由安排 / 补进度日，补充笔记** | **⬅️ 今天** |
| 4/28 周二 | 整理本周笔记，写"vLLM 初体验" | ☐ |

---

## 🔗 相关笔记

- [[SGLang与vLLM推理框架对比分享]] — 主项目计划
- [[SGLang-vs-vLLM-研究笔记]] — 研究资料汇总
- [[论文笔记-PagedAttention-vLLM]] — PagedAttention 论文笔记

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### Week 1 知识地图（5 分钟复述测试）
合上笔记，按这条因果链讲一遍，讲不顺的环节就是要补的：

```
LLM 推理两阶段：Prefill(算提示词,算力密集) → Decode(逐token,访存密集)
  ↓ decode 阶段 GPU 算力大量闲置 → 想加大 batch 提高利用率
  ↓ batch 开不大的瓶颈是 KV Cache 显存 → 传统预分配浪费 60-80%
  ↓ PagedAttention：分块 + Block Table → 浪费 <4%，同卡并发×几倍
  ↓ Continuous Batching：token 级调度 → 并发不空转
  ↓ Prefix Caching：跨请求复用相同前缀 KV → TTFT 大降
  = vLLM 高吞吐的三大支柱
```

### 本周高频概念速查

| 术语 | 一句话 |
|------|--------|
| TTFT | 首 token 延迟 = 排队 + prefill 时间 |
| TPOT/ITL | 每输出 token 时间 / token 间隔，决定"打字速度" |
| Throughput | 系统每秒吐出的总 token 数（所有请求加总） |
| GQA | 多个 Q 头共享一组 KV 头 → KV Cache 直接缩小数倍 |
| Chunked Prefill | 把长 prefill 切片与 decode 混跑，防止长提示词卡住别人 |

### 常见误区纠偏
- ❌ "PagedAttention 让计算更快" → 它让**显存更省**，吞吐提升来自更大 batch；kernel 本身略变慢
- ❌ "吞吐高 = 体验好" → 吞吐和单请求延迟是一对矛盾，调参就是在二者间选位置
- ❌ "TTFT 短靠 decode 快" → TTFT 只关 prefill 和排队的事
