---
title: SGLang-vLLM学习-05-07
date: 2026-05-07
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📅 今日任务

> 📖 深入 SGLang RadixAttention 源码（`sglang/srt/layers/attention/`），理解 radix tree 操作

- [ ] 阅读 RadixAttention 核心实现代码，梳理类结构与关键方法
- [ ] 理解 radix tree 的插入、查找、淘汰等操作流程
- [ ] 对比 PagedAttention 与 RadixAttention 的 KV Cache 管理策略差异
- [ ] 记录关键数据结构和算法设计决策

**预期产出：** 源码阅读笔记

## 📍 本周阶段

| 项目 | 内容 |
|------|------|
| **阶段** | M2: 架构深入 |
| **周次** | Week 3：架构深入 — KV Cache 管理 |
| **本周目标** | 搞懂 PagedAttention vs RadixAttention 设计哲学 |
| **距分享剩余** | 59 天 |

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

### 源码地标
```
python/sglang/srt/mem_cache/radix_cache.py   ★今天主角
  ├─ class TreeNode        # children: {token段: 子节点}, value: KV索引, last_access_time
  └─ class RadixCache
       ├─ match_prefix(token_ids)   # 沿树走，返回命中的 KV 索引 + 最后节点
       ├─ insert(token_ids, kv)     # 请求结束后把新 KV 挂上树（可能分裂边）
       ├─ evict(num_tokens)         # 显存不足时 LRU 弹出叶子
       └─ inc/dec_lock_ref()        # 正在使用的节点加锁防淘汰
python/sglang/srt/mem_cache/memory_pool.py   # token_to_kv_pool 实际显存池
python/sglang/srt/managers/scheduler.py      # 调度时调用 match_prefix 排序
```

### 读懂 radix 树的三个动作（手画一遍）
1. **匹配**：请求 [A,B,C,D]，树上已有 [A,B,X]→ 公共前缀 [A,B] 命中，C,D 需 prefill
2. **分裂**：树边是压缩的 token 段，命中到段中间时把边切成两段（radix 树与普通 trie 的区别）
3. **淘汰**：`evict` 按 last_access_time 弹叶子；被请求引用（lock_ref>0）的节点跳过

### 与 vLLM 的设计哲学对照（专题笔记核心表）
| 维度 | vLLM PagedAttention+APC | SGLang RadixAttention |
|------|------------------------|----------------------|
| 基本抽象 | 块（page） | 前缀（tree path） |
| 复用粒度 | 整块对齐 | 任意 token 长度 |
| 复用判定 | 块哈希查表 O(1) | 树匹配 O(前缀长) |
| 调度感知 | 否（队列 FCFS 为主） | 是（按命中长度重排队列） |
| 设计初心 | 解决显存碎片 | 解决 LLM 程序的重复计算 |

### 自测题
1. 为什么需要 lock_ref？（正在 decode 的请求其前缀 KV 不能被 LRU 淘汰）
2. radix 树的"边分裂"什么时候发生？（新序列与已有边在中间分叉）
