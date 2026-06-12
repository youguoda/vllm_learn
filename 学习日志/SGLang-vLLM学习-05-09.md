---
title: SGLang与vLLM学习笔记-05-09
date: 2026-05-09
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

# 📖 SGLang与vLLM学习笔记 — 5月9日

## 📌 今日任务

- [ ] 阅读 SGLang HiCache 博客
- [ ] 理解层级 KV Cache 机制
- [ ] 整理 HiCache 笔记产出
- [ ] 对比 PagedAttention vs RadixAttention 设计哲学

## 🎯 本周阶段

> **Week 3：架构深入 — KV Cache 管理**
> 
> 目标：搞懂 PagedAttention vs RadixAttention 设计哲学
> 
> 里程碑：M2 — 架构深入

## 📝 学习记录

### HiCache 核心概念

（在此记录学习内容...）

### PagedAttention vs RadixAttention 对比

| 维度 | PagedAttention (vLLM) | RadixAttention (SGLang) |
|------|-----------------------|------------------------|
| 设计思路 | | |
| 内存管理 | | |
| 前缀共享 | | |
| 适用场景 | | |

### 关键发现与思考

（在此记录个人理解和思考...）

## 🔗 关联链接

- [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### HiCache：把 Radix 树伸到 GPU 之外
动机：radix 树留住的 KV 越多命中越高，但 GPU 显存就那么大 → **分层存储**。

```
L1 GPU 显存   ←最热，直接参与计算
L2 CPU 内存   ←次热，命中后异步搬回 GPU
L3 NVMe/磁盘  ←冷数据，长会话/超多租户场景
```

关键机制：
1. **写穿/写回**：GPU 上新生成的 KV 异步下沉到 CPU 层
2. **命中搬运**：L2 命中时边搬边算（overlap），掩盖 PCIe 延迟
3. **代价判断**：搬运时间 vs 重算时间——前缀越长越值得搬（重算是平方级注意力）

价值场景：客服系统上千个会话间歇性活跃——GPU 只留活跃会话，睡眠会话的 KV 退到 CPU/盘，唤醒时免重算。

### 设计哲学对比总结（本周收官表，写进专题笔记）
| 问题 | vLLM 的回答 | SGLang 的回答 |
|------|------------|--------------|
| 显存不够放谁 | 抢占：换出/重算 *当前* 请求 | 淘汰：LRU 丢 *历史* 前缀 |
| 历史 KV 的身份 | 用完即弃的运行时状态 | 值得管理的"资产" |
| 扩容方向 | 加卡扩并发 | 分层存储扩缓存（HiCache） |

一句话收束：**vLLM 把 KV 当内存管理问题，SGLang 把 KV 当缓存系统问题**——这句可以直接当分享金句。
