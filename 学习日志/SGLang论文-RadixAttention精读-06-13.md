---
title: SGLang 论文精读：RadixAttention —— 从 vLLM 块哈希的局限到 radix 树前缀复用
date: 2026-06-13
tags:
  - R/技术框架/SGLang
  - R/核心算法/LLM推理
  - R/核心算法/推理优化
  - output/active
related:
  - "[[vLLM-PrefixCaching实测-06-13]]"
  - "[[论文笔记-PagedAttention-vLLM]]"
source: "SGLang: Efficient Execution of Structured Language Model Programs (arXiv:2312.07104) §3"
---

# SGLang 论文精读：RadixAttention

> 完成 4/29 计划。先用实验亲眼看到 **vLLM 块哈希命中不了的现场**，带着"为什么"去读论文 §3，再手画 radix 树演化，最后自测对比两套机制的粒度差异。

---

## 步骤 1：先制造"vLLM 命中不了"的现场

### 实验设计

vLLM 的 APC 以 **block（16 token）** 为粒度缓存 KV，只能命中**整块**。脚本 [exp_partial_prefix.py](../exp_partial_prefix.py) 构造一个"半块对齐"场景：
- 请求 A：长共享前缀 + "请分析轴承故障。"
- 请求 B：把共享前缀**截断到非块边界**（`len(BASE)//2 + 3` 字符）再接不同内容

### 实测数据（Qwen2.5-1.5B, RTX 3060）

| 请求 | TTFT | prompt | 查询 block | 命中 block | 命中率 |
|---|---:|---:|---:|---:|---:|
| A 冷启动 | 456.8ms | 225 tok | 225 | 0 | 0.0% |
| **A 重复（整块对齐）** | **28.5ms** | 225 tok | 225 | 224 | **99.6%** |
| **B 共享截断前缀** | 32.1ms | 117 tok | 117 | 96 | **82.1%** |
| B 重复（整块对齐） | 28.7ms | 117 tok | 117 | 112 | 95.7% |

### 现场结论

- **A 重复命中 99.6%**：前缀完全块对齐 → 几乎整条复用。
- **B 共享截断只命中 82.1%**：B 与 A 共享前半段，但截断点**不在 16-token 块边界**上。共享部分只能命中到**最后一个完整对齐块**为止，尾部那个"不满块"的 KV 无法复用，必须重算。

> 这就是块哈希的硬伤：**命中粒度被锁死在 16 token**。差几个 token 没对齐，整块就白算。带着这个"为什么会浪费"的问题去读 RadixAttention，就知道它在解决什么了。

---

## 步骤 2：带四个问题读论文 §3

论文：*SGLang: Efficient Execution of Structured Language Model Programs* (arXiv:2312.07104)，§3 "Efficient KV Cache Reuse with RadixAttention"。

### Q1：radix 树的节点/边存什么？

> "we utilize a radix tree to manage a mapping between **sequences of tokens** and their corresponding **KV cache tensors**. These KV cache tensors are stored in a non-contiguous, paged layout, where the size of each page is equivalent to **one token**."

- **边（edge）**：存一串 token（一个子序列），不是单个 token。
- **节点**：代表"从 root 到此处"的一段 token 前缀，关联指向 **KV cache 张量**的引用。
- KV cache 用**非连续分页**布局，**每页 = 1 个 token**（注意：粒度是 token 级，不是 vLLM 的 16-token 块级 —— 这是命中粒度差异的根源）。

### Q2：为什么用 radix 树而不是普通 trie？

> "Unlike typical trees, the edges of a radix tree can be labeled not just with single elements but also with **sequences of elements of varying lengths**, significantly enhancing efficiency."

- 普通 trie：每条边只存 1 个 token → 长前缀要拉出一长串单节点链，节点爆炸。
- radix 树：一条边可以存**任意长度的 token 序列** → 公共前缀压成一条边，**节点数大幅减少**，匹配/维护都更快。

### Q3：命中任意长度前缀怎么做到？（节点 split）

> "In step (4), the node 'b' ... is **split into two nodes** to allow the two chat sessions to share the system prompt."

- 新请求与已有边只共享**一部分**时，把那条边在**分叉点 split**成两个节点：前半段（公共前缀）成为父节点，后半段和新内容成为两个子节点。
- 因为页粒度是 1 token，split 可以发生在**任意 token 位置** → **任意长度的前缀都能命中**，不受块边界约束。这正是 vLLM 块哈希做不到的（步骤 1 的 B 请求就卡在这）。

### Q4：淘汰策略？

> "a simple **LRU eviction policy** that evicts the **least recently used leaf first**. By evicting leaves first, we enable the re-use of their common ancestors until those ancestors become leaves..."
>
> "each node maintains a **reference counter** indicating how many running requests are using it. A node is **evictable if its reference counter is zero**."

- **LRU + 叶子优先**：先淘汰最久未用的**叶子**节点。叶子被淘汰后，其祖先才可能变成叶子被淘汰 → 保护了被多个请求共享的公共前缀。
- **引用计数**：连续批处理下不能淘汰**正在被运行请求使用**的节点。`ref_count > 0` 不可淘汰，`= 0` 才可淘汰。
- KV cache 和运行中请求**共享同一内存池**，不预留固定 cache 区 → 等待请求多时，可以把所有 cache 淘汰换更大 batch（动态平衡）。

### 额外要点：cache-aware scheduling（缓存感知调度）

> 缓存命中率 = `已缓存的 prompt token 数 / 总 prompt token 数`

- 不用 FCFS，而是按**匹配前缀长度排序**，优先调度**共享前缀最长**的请求（longest-shared-prefix-first）。
- **Theorem 3.1**：对一批请求，按 radix 树的 **DFS（深度优先）顺序** 访问，cache size ≥ 最大请求长度时，可达**最优命中率**。longest-shared-prefix-first 等价于 DFS 序。
- **Frontend Hint**：`fork` 原语执行时，前端先把公共前缀发过去，确保它先被正确插入树，再发各分支的剩余部分 —— 前端-运行时协同设计。
- 树结构存在 **CPU** 上，维护开销可忽略；命中时几乎零额外内存/时间开销。
- 分布式：张量并行下每个 GPU 存自己的 KV shard，**树操作相同，无需额外同步**。

---

## 步骤 3：手画 radix 树演化

照计划用三个请求（共享系统提示 S，对应步骤 1 的中文场景）：

> **先看一张通俗简图（抄作业版）**：相同开头只算一次、大家"复印"复用，各自不同的结尾才自己算；显存满时先扔没人共享的叶子。

![[assets/radix树-抄作业版.svg]]


```
R1: [系统提示S] + "分析轴承故障"
R2: [系统提示S] + "分析齿轮磨损"
R3: [系统提示S] + "分析轴承寿命"
```

### ① R1 到达：root 挂一条整边

```
   root
    │
    │ "S分析轴承故障"          ← 一条边存整串 token (radix 树特性)
    ▼
   [n1]  (ref=1, 关联 KV)
```

### ② R2 到达：公共前缀 "S分析" → 原边 split

R2 与 n1 共享 "S分析"，在此处 split：原边断成 "S分析"（父）+ "轴承故障"（子），再挂 R2 的 "齿轮磨损"。

```
   root
    │ "S分析"                  ← split 出的公共前缀, 被复用
    ▼
   [p1] (ref=2)
   ╱        ╲
"轴承故障"   "齿轮磨损"
  ▼            ▼
 [n1]        [n2]
(R1)         (R2 新增, 只需算 "齿轮磨损" 部分)
```

> R2 复用了 "S分析" 的 KV，**只对 "齿轮磨损" 做 prefill**。注意 split 发生在 "分析" 后的任意 token 位置，不受 16-token 块边界限制。

### ③ R3 到达："S分析轴承" → 在 "轴承" 处再 split 一次

R3 = "S分析轴承寿命"，与左支 "S分析"+"轴承故障" 共享到 "S分析轴承"。在 "轴承" 后再 split：

```
   root
    │ "S分析"
    ▼
   [p1] (ref=3)
   ╱         ╲
"轴承"      "齿轮磨损"
  ▼            ▼
 [p2]        [n2] (R2)
(ref=2)
 ╱    ╲
"故障" "寿命"
 ▼      ▼
[n1]   [n3]
(R1)   (R3 新增, 只需算 "寿命")
```

> R3 复用了 "S分析轴承" 的 KV，**只对 "寿命" 做 prefill**。两次 split 后，公共前缀被三个请求层层共享。

### ④ 内存不足 → LRU 叶子优先淘汰

假设此时显存吃紧需要淘汰。可淘汰的只有**叶子**且 **ref=0** 的节点。设访问顺序是 R1→R2→R3（R1 最久未用）：

```
候选叶子: [n1](R1,故障)  [n2](R2,齿轮磨损)  [n3](R3,寿命)
LRU 最久未访问 + ref=0  →  先淘汰 [n1] (R1 的 "故障")

  淘汰 [n1] 后:
   [p2] "轴承" 下只剩 [n3] "寿命"
   → 若仍不够, [p2] 变成只有单子, 但公共前缀 [p1]"S分析"、[p2]"轴承"
     因为还被 R3 引用 (ref>0), 不会被淘汰  ← 保护共享前缀
```

> **关键**：叶子优先 + 引用计数，保证了被多个请求共享的公共前缀（"S分析"、"轴承"）**最后才被淘汰**，最大化复用价值。这对应论文 Figure 3 step (8)(9) 里 evict 叶子 g/h/i/k/l 的行为。

---

## 步骤 4：自测

### 一句话对比

> **vLLM PagedAttention 的 APC**：哈希按 **16-token 块** 为粒度，**必须整块对齐才命中**，块内差一个 token 整块失效（链式哈希）。
>
> **SGLang RadixAttention**：radix 树按 **token 前缀** 为粒度（每页 1 token），匹配到一半可 **split 节点**，**任意长度前缀都能命中**。代价是 radix 树的维护与（多请求并发时的）锁开销，以及引用计数管理。

### 用步骤 1 数据验证

步骤 1 的 **B 请求命中率只有 82.1%** 就是 vLLM 块粒度的直接证据：B 与 A 共享前半段，但截断点不在 16-token 块边界，**尾部不满块无法命中**。

- 如果换成 SGLang，radix 树会在共享前缀的**精确 token 位置 split**，把 A、B 的公共部分**完整复用**（理论命中率接近共享 token 占比，而非被块边界截断）。
- 这就是 RadixAttention 在**多轮对话 / few-shot / RAG** 等"前缀大量共享但不一定块对齐"场景下，相比 vLLM 最高 **6.4× 吞吐提升**的来源（论文 §6.2）。

### 粒度差异一图总结

```
共享前缀 (假设 35 token), block_size=16:

vLLM 块哈希:  [块0:16tok][块1:16tok][半块:3tok]
              ✓ 命中      ✓ 命中     ✗ 不满块, 重算
              → 只复用 32/35 token, 尾部 3 个浪费

SGLang radix: root ──"35个token的精确前缀"──> split 在第35个token处
              ✓ 全部 35 token 复用
              → 任意位置可 split, 无块边界浪费
```

---

## 今日产出

- [x] **exp_partial_prefix.py** 三条 TTFT 数据（A冷456ms / A重复28ms命中99.6% / B截断32ms命中82.1%）
- [x] **论文笔记**（四问四答 + cache-aware scheduling + Theorem 3.1）
- [x] **手画 radix 树演化图**（R1 整边 → R2 split "S分析" → R3 再 split "轴承" → LRU 叶子优先淘汰）

> 这张 radix 树演化图 05-07 读源码（`sglang/srt/mem_cache/radix_cache.py`）、05-08 画架构图都要复用。

## 关键引用（论文原文）

- 节点存储：*"a radix tree to manage a mapping between sequences of tokens and their corresponding KV cache tensors ... each page is equivalent to one token"*
- radix vs trie：*"edges ... labeled not just with single elements but also with sequences of elements of varying lengths"*
- 淘汰：*"LRU eviction policy that evicts the least recently used leaf first ... A node is evictable if its reference counter is zero"*
- 调度：*"sort the requests by matched prefix length and prioritize requests with longer matched prefixes"* + Theorem 3.1 (DFS 序最优)
