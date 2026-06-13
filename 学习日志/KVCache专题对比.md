---
title: KV Cache 管理机制专题对比（Week 3 交付）
date: 2026-06-14
tags:
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/核心算法/LLM推理
  - output/milestone
related:
  - "[[vLLM源码精读-PagedAttention内存管理-06-14]]"
  - "[[SGLang源码精读-RadixAttention-06-14]]"
  - "[[KVCache设计哲学-HiCache-06-14]]"
  - "[[前缀复用率实验-06-14]]"
  - "[[KVCache架构对比图-06-14]]"
---

# KV Cache 管理机制专题对比

> Week 3 交付。把 M2/M3 关于 KV Cache 的全部理解（源码 + 实验）汇总成一篇可直接进 PPT 的专题。每节配源码引用或实验数据。

---

## 1. 背景：为什么 KV Cache 是推理的核心瓶颈

LLM 推理的 **decode 阶段是访存密集型**：每生成 1 个 token，都要读取整个模型权重 + 全部历史 token 的 KV。KV Cache 把历史 token 的 Key/Value 存下来避免重算，但它**吃满显存**——KV 越大，能并发的 batch 越小，GPU 算力越摸鱼。

> 实测锚点（[[vLLM-PagedAttention实测-06-13]]）：RTX 3060 上 KV 池 228,288 token，传统"每请求预留 max_len 连续显存"浪费 98.8%，PagedAttention 按需分块降到 21.9%，**同显存多服务 65 倍请求**。这就是 KV 管理为何决定吞吐。

---

## 2. vLLM PagedAttention

### 2.1 数据结构（源码 [[vLLM源码精读-PagedAttention内存管理-06-14]]）

- **`KVCacheBlock`**（kv_cache_utils.py:115）：`block_id` / `ref_cnt` / `_block_hash` / `prev_free_block`+`next_free_block`（双向链表指针）。注意真实字段是 `ref_cnt` 不是 `ref_count`，块内**不存 token**。
- **Block Table**：逻辑块 → 物理块 id 的映射（间接寻址），物理块可散落显存任意位置。
- **`FreeKVCacheBlockQueue`**（:164）：手写带哨兵的双向链表，支持 O(1) 中间删除（为"命中复活"服务）。

### 2.2 分配与释放

- **`allocate_slots`**（kv_cache_manager.py:236）：先 `get_computed_blocks` 查 APC 缓存命中 → `touch` 复活命中块（ref_cnt+1）→ `get_new_blocks` 从 free list 队头补新块。块不够则 `return None` 触发抢占。
- **`free_blocks`**（block_pool.py:419）：ref_cnt-1，归零的块放回 free list 队尾，**哈希保留**（惰性失效，可被下次命中）。真正清哈希只在被 `get_new_blocks` 挤占时（`_maybe_evict_cached_block`）。

### 2.3 APC（Automatic Prefix Caching）

- **链式哈希**（kv_cache_utils.py:541）：`hash(块i) = hash((hash(块i-1), 本块token, extra))`。
- **命中粒度**：16 token 整块，必须块对齐。
- **实验证据**（[[vLLM-PrefixCaching实测-06-13]]）：

| | WARM TTFT | 命中率 |
|---|---:|---:|
| APC ON | 33.3ms | 98.7% |
| APC OFF | 152.6ms | 0% |

→ APC 把 TTFT 砍 4.6 倍，且只省 prefill 不影响 decode。

---

## 3. SGLang RadixAttention

### 3.1 数据结构（源码 [[SGLang源码精读-RadixAttention-06-14]]）

- **`TreeNode`**（radix_cache.py:201）：`children`（defaultdict）/ `parent` / `key`(RadixKey) / `value`(KV 池索引) / `lock_ref` / `last_access_time` / `host_value`(CPU 副本)。`__lt__` 按时间戳 → 可进最小堆做 LRU。
- **`RadixKey`**（:56）：边标签 token 序列 + `extra_key`（lora/salt 隔离），`match()` 用指数搜索+二分找分歧点。

### 3.2 match_prefix / insert / evict

- **`match_prefix`**（:337→`_match_prefix_helper`:622）："走到走不动为止"——整条边匹配则深入，匹配到边中间则 `_split_node` 劈开。任意 token 位置可 split。
- **`insert`**（:397）：同样沿树下行 + split，剩余 token 挂新叶子。
- **`evict`**（:537）：最小堆 LRU，只淘汰叶子（已排除 lock_ref>0），删叶子后父变叶子再入堆（逐层向上）。`inc_lock_ref` 沿路径锁到 root。

tiny_radix.py 实测两次 split（[[SGLang源码精读-RadixAttention-06-14]]）：
```
R1=SYS+(10,11) → R2=SYS+(20,21) 在第5token split → R3=SYS+(10,99) 在第1token再split
最终 SYS 公共边只出现一次
```

### 3.3 HiCache（多级扩展，[[KVCache设计哲学-HiCache-06-14]]）

- 三级：GPU 显存 → CPU host 池（ratio 倍大）→ 磁盘。
- 搬移时机：evict 时不丢弃，`_evict_backuped` 把 KV 降级到 CPU（node.value=None，host_value 保留）。
- 命中：`load_back` 异步搬回 GPU。实测重放首条 45.6ms（介于冷启动 280ms 和全命中 27ms 之间）。
- ⚠️ 环境坑：WSL2/RTX3060 必须 `--hicache-io-backend direct`（默认 kernel 崩溃）。

---

## 4. 核心差异对照表

| 维度 | vLLM | SGLang |
|---|---|---|
| 索引结构 | 哈希表（dict） | Radix 树 |
| 命中粒度 | 16 token（块对齐） | 任意长度（page_size=1） |
| split 操作 | 无 | 有（边可劈开） |
| 引用计数 | block.ref_cnt（单块） | node.lock_ref（沿路径锁到 root） |
| 唯一性保证 | 链式哈希 | 树结构天然保证（无需哈希） |
| 淘汰 | LRU free_list（双向链表） | evict() 最小堆 LRU 叶子 |
| 扩展 | CPU swap（重算/换出） | HiCache 三级缓存 |
| 思想来源 | 操作系统虚拟内存 | 数据库/编译器缓存 |

---

## 5. 实验数据：前缀复用率决定选型

![前缀复用率](../assets/prefix_ratio_comparison.png)

实测（[[前缀复用率实验-06-14]]）：

| 复用率 | vLLM RPS | SGLang RPS |
|---:|---:|---:|
| 0% | 7.31 | 7.08 |
| 75% | 8.21 | 8.05 |
| **100%** | **8.88** | **18.95** |

**结论**：
> 复用率 **75% 以上** SGLang 吞吐优势开始显著，100% 时达 **2.1 倍**（18.95 vs 8.88）；**75% 以下**两者基本持平。客服/agent/RAG/多轮对话等高复用场景 → **SGLang**；通用/低复用负载 → **vLLM**（简单可靠、生态成熟）。

---

## 6. 常见误区

- ❌ **"SGLang KV Cache 机制整体更好"** → 只在高复用负载下吞吐优势明显；低复用时两者持平，中等复用（50%）vLLM 的 TTFT 甚至略优。
- ❌ **"vLLM APC 没用"** → 命中整块时效果接近 radix 树（TTFT 都降到 ~40ms），实现简单可靠。差异只在块边界对齐损失（半块前缀命中 82.1% vs 94.9%）。
- ❌ **"HiCache 没延迟"** → CPU 命中需 PCIe 传输（ms 级），实测从 CPU 搬回 45ms vs GPU 直接命中 27ms。比重算（280ms）省，但不是零成本。

---

## 7. 自测题：为什么块哈希要链式包含前缀哈希？

> **答**：Attention 是全局因果的——同一段 token "今天适合出门" 对应的 KV 值，取决于它前面**所有** token 的上下文。两个"局部内容相同但前文不同"的块，KV 其实完全不同，绝不能复用。
>
> 如果哈希只算本块 16 个 token，就会把这两个不同上下文的块判成"相同"→ 错误复用 → 生成结果错乱。链式哈希 `hash(i)=f(hash(i-1), tokens_i)` 让"哈希相同"严格等价于"从序列开头到这里完全一致"。用 O(1) 的代价保证正确性：改动第 i 块的任一 token，从 i 块起所有后续哈希全变，绝对查不到错误缓存。
>
> 实测印证（verify_hash.py）：改 block0 一个 token，block0 和 block1 哈希全变；部分相同前缀逐块比对 `[True, True, False]`，命中到分歧点为止。

---

## 今日产出

- [x] KVCache专题对比.md（完整，带源码引用 + 实验数字）
- [x] assets/selection_guide.png（选型决策树）
- [x] 自测题书面回答（链式哈希为何必要）
