---
title: SGLang-vLLM学习-04-26
date: 2026-04-26
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---
## 📅 今日任务

> 🔧 体验 vLLM 的 Prefix Caching（APC），对比开关前后的性能

- [ ] 在 vLLM 中启用 Automatic Prefix Caching（`--enable-prefix-caching`）
- [ ] 构造含共享 system prompt 的多轮请求，观察 cache hit 情况
- [ ] 对比开/关 APC 下的 TTFT 和 throughput
- [ ] 记录不同前缀复用率下的性能变化

**预期产出：** 对比数据

## 📍 本周阶段

|| 项目 | 内容 ||
||------|------||
|| **阶段** | M1: 基础认知 ||
|| **周次** | Week 1：环境搭建 + vLLM 上手 ||
|| **本周目标** | vLLM 能跑起来，理解 PagedAttention 核心思想 ||
|| **距分享剩余** | 70 天 ||

## 📝 学习记录

%% 在下方记录今天的学习内容 %%

### 核心笔记

### 疑问与待查

### 心得体会

## 🔗 关联

- [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### APC 原理：哈希复用而非树

vLLM 的 Automatic Prefix Caching 给**每个满块**计算哈希：`hash(本块 token ids + 前缀所有块的哈希)`。新请求 prefill 时逐块查哈希表，命中则直接复用物理块、跳过计算。被淘汰时用 LRU。

与 SGLang RadixAttention 的本质区别：**vLLM 以"块"为粒度查哈希表，SGLang 以"token 前缀"为粒度走 radix 树**——树能命中任意长度公共前缀，哈希只能命中整块对齐的前缀；但哈希实现简单、并发开销低。

### 实验设计（今天的核心产出）

```bash
# 开启 APC（新版本默认开，可用 --no-enable-prefix-caching 关）
vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-prefix-caching
```

```python
# 构造共享前缀负载：同一个 1000 字 system prompt + 50 个不同问题
import time, openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
SYS = "你是工业质检专家……" * 50   # 拉长前缀放大效果
for q in questions:
    t0 = time.time()
    r = client.chat.completions.create(model=MODEL, stream=True,
        messages=[{"role":"system","content":SYS},{"role":"user","content":q}])
    first = next(iter(r)); print("TTFT:", time.time()-t0)
```

预期：第 1 条请求 TTFT 正常，之后 TTFT 大幅下降（system prompt 的 KV 全部命中）。`/metrics` 里看 `gpu_prefix_cache_hit_rate`。

### 何时无效/有害

- 前缀各不相同的负载：纯开销（哈希计算）
- 命中只省 **prefill**（即 TTFT），对 decode 速度（ITL）没有帮助

### 自测题

1. 为什么块哈希要把"前缀块的哈希"也算进去？（同样内容的块出现在不同上下文位置，KV 值不同，不能混用）
