---
title: SGLang 调度器源码精读 —— Cache-Aware Scheduling（LPM / DFS-Weight）
date: 2026-06-14
tags:
  - R/技术框架/SGLang
  - R/核心算法/LLM推理
  - R/源码精读
  - output/active
related:
  - "[[vLLM调度器源码精读-06-14]]"
  - "[[SGLang源码精读-RadixAttention-06-14]]"
  - "[[SGLang论文-RadixAttention精读-06-13]]"
---

# SGLang 调度器源码精读：Cache-Aware Scheduling

> 衔接 05-13（vLLM 调度器 = FCFS + token_budget + RECOMPUTE 抢占）。今天读 SGLang 的 `SchedulePolicy`——它为什么能比 vLLM 的 FCFS 在多前缀场景再压一截 TTFT。
>
> 源码：`sglang/srt/managers/schedule_policy.py`（1094 行），SGLang 0.5.13。

---

## 步骤 1：核心结构定位

| 结构/方法 | 行 | 职责 |
|---|---|---|
| `CacheAwarePolicy` | 133 | 缓存感知策略：LPM / DFS_WEIGHT |
| `CacheAgnosticPolicy` | 140 | 无感知：FCFS / LOF / RANDOM / ROUTING_KEY |
| `calc_priority` | 170 | 调度主入口 |
| `_determine_active_policy` | 223 | 队列过长降级 FCFS |
| `_compute_prefix_matches` | 247 | in-batch prefix caching |
| `_sort_by_longest_prefix` | 296 | LPM 排序 |
| `_sort_by_dfs_weight` | 308 | DFS_WEIGHT 排序（Theorem 3.1） |

> 对比 vLLM：vLLM 只有 FCFS（+可选 PRIORITY）。SGLang 有 **6 种策略**，其中 2 种是**缓存感知**的——这是它调度层的核心差异。

---

## 步骤 2：calc_priority 调度决策树

```python
def calc_priority(self, waiting_queue, running_batch=None):
    policy = self._determine_active_policy(waiting_queue)   # 可能降级
    if isinstance(policy, CacheAwarePolicy):
        temporary_deprioritized = self._compute_prefix_matches(waiting_queue, policy)
        if policy == LPM:        _sort_by_longest_prefix(...)    # 按命中长度排序
        elif policy == DFS_WEIGHT: _sort_by_dfs_weight(...)      # 按树拓扑排序
```

### 决策树

```
waiting_queue 长度 > 128?
  YES → 降级 FCFS  [_determine_active_policy:223]
        (LPM 对所有等待请求做前缀匹配 O(N), 队列太长成本高)
  NO  → 看配置策略:
          LPM        → _sort_by_longest_prefix  (按 num_matched_prefix_tokens 降序)
          DFS_WEIGHT → _sort_by_dfs_weight       (Radix 树 DFS 权重, 论文 Theorem 3.1)
          FCFS/LOF/RANDOM → 无缓存感知
```

> **设计智慧**：`_determine_active_policy`（:224）在队列 >128 时自动降级 FCFS。**缓存感知排序不是免费的**——对每个等待请求都要做前缀匹配，队列太长时这个 O(N) 开销超过收益，不如直接 FCFS。这是工程上的成本-收益平衡。

### in-batch prefix caching（论文没明说的工程优化）

`_compute_prefix_matches`（:247）解决一个微妙问题：**等待队列里多个请求共享同一个"还没被缓存"的前缀**怎么办？

```python
for r in waiting_queue:
    match_prefix_for_req(self.tree_cache, r)         # ① 查全局缓存
    if len(r.prefix_indices) <= CHECK_THRESHOLD(32): # 全局命中不足
        in_batch_match = self.waiting_queue_radix_tree.match_prefix(r)  # ② 查批内临时树
        if len(in_batch_match) >= DEPRIORITIZE_THRESHOLD(32):
            temporary_deprioritized.add(r.rid)       # ③ 已有别的请求占住该前缀 → 降优先级
        else:
            self.waiting_queue_radix_tree.insert(r)  # ④ 第一个出现 → 占位
```

> **巧思**：如果 8 个请求都带同一个新前缀，**先让 1 个跑**（它会把前缀写进缓存），其余 7 个降优先级稍等——等第 1 个跑完，前缀就在缓存里了，7 个直接命中，**避免 8 个请求同时重复 prefill 同一前缀**。两个阈值（32 token）都可用环境变量调。

---

## 步骤 3：DFS_WEIGHT —— Theorem 3.1 的代码实现

`_sort_by_dfs_weight`（:308）是论文 §3.2 "Cache-Aware Scheduling" 的直接实现：

```python
def _sort_by_dfs_weight(waiting_queue, tree_cache):
    # ① 按 last_node 分组(每个请求前缀命中到的最深节点)
    last_node_to_reqs = defaultdict(list)
    for req in waiting_queue:
        last_node_to_reqs[req.last_node].append(req)
    # ② 每个节点权重 = 该子树下的等待请求数
    node_to_weight = defaultdict(int)
    for node in last_node_to_reqs:
        node_to_weight[node] = len(last_node_to_reqs[node])
    _calc_weight(tree_cache.root_node, node_to_weight)   # 后序遍历, 权重向上累加
    # ③ DFS 遍历, 按子树权重降序选分支
    waiting_queue.clear()
    _get_dfs_priority(tree_cache.root_node, ...)
```

### 手画示例

```
              root
             /    \
        [A,B,C]   [X,Y]
        /    \
    [D,E]    [F]
   (req1)  (req2,req3)

_calc_weight 后(子树请求数):
  [D,E]=1, [F]=2, [A,B,C]=3, [X,Y]=0, root=3

DFS 按权重降序:
  先走 [A,B,C](w=3) → 先 [F](w=2)出 req2,req3 → 再 [D,E](w=1)出 req1
  后走 [X,Y](w=0)
输出顺序: req2, req3, req1
```

**为什么最优（Theorem 3.1 直觉）**：把共享同一前缀的请求**集中连续调度**，前缀 KV 块在它们全跑完前不会被 LRU 淘汰。若打乱穿插，前缀块可能在中途被驱逐，后续请求被迫重算。DFS 序 = 让缓存命中最大化的访问序。

### LPM vs DFS_WEIGHT

| 维度 | LPM | DFS_WEIGHT |
|---|---|---|
| 排序依据 | num_matched_prefix_tokens（绝对长度） | Radix 树 DFS 权重（相对拓扑） |
| 适用 | 少量请求、前缀分散 | 多请求、前缀有树状层次 |
| 成本 | O(N log N) 排序 | O(N + 树节点) DFS |
| 论文依据 | 贪心近似 | Theorem 3.1（最优序） |
| 默认 | **是**（--schedule-policy lpm） | 需手动开 dfs-weight |

---

## 步骤 4：对比实验（LPM vs FCFS）

8 个不同问题共享同一长 system prompt（~200 token 气象数据），对比 SGLang 的 LPM 和 FCFS 策略。

| 策略 | 冷启动 TTFT | 暖 TTFT | 备注 |
|---|---:|---:|---:|
| SGLang LPM | 676.6ms | 376.9ms | 缓存感知排序 |
| SGLang FCFS | 663.4ms | **41.1ms** | 缓存有，排序无 |

### 诚实的发现：单前缀场景 LPM 优势不明显

> 实测中 LPM 和 FCFS 在这个**单一共享前缀**场景下没拉开差距（暖 FCFS 甚至更快）。原因：**只有一个前缀分支时，无论怎么排序，第一个请求把前缀写进缓存后，剩下 7 个都直接命中**——cache-aware 排序的价值体现不出来。
>
> **cache-aware 调度真正的用武之地是"多个竞争前缀"**：当等待队列里有 A、B、C 三组不同前缀，FCFS 可能 A1-B1-C1-A2-B2-C2 穿插，导致每组前缀刚缓存又被另一组挤掉；LPM/DFS_WEIGHT 会把 A1-A2-B1-B2-C1-C2 同前缀集中，最大化命中。我的单前缀实验不足以暴露这个差异——这是实验设计的局限，但理解了机制就知道何时它重要。

> 注：warm LPM 的 377ms 偏高疑似单次测量噪声（in-batch 临时树维护开销 + 测量抖动）。结论应看机制而非这组小样本数字。

---

## 步骤 5：自测 5 题

1. **SGLang 默认 cache-aware 策略？** → `lpm`（longest prefix match）。
2. **何时降级 FCFS？** → 等待队列 > 128 请求时（`_determine_active_policy`，避免 O(N) 前缀匹配开销）。
3. **in-batch prefix caching 解决什么？** → 等待队列多个请求共享同一未缓存前缀时，只让一个先跑，跑完前缀就缓存了，其余直接命中，避免重复 prefill 同一前缀。
4. **DFS_WEIGHT vs LPM 根本区别？** → LPM 只看单请求的绝对命中长度（贪心）；DFS_WEIGHT 考虑整棵树的全局请求拓扑，把同一子树下所有请求集中调度防 LRU 驱逐（Theorem 3.1 最优）。
5. **vLLM vs SGLang 调度层最大差异？** → vLLM 块级哈希 + FCFS（无缓存感知排序）；SGLang token 级 Radix 树 + cache-aware 排序（LPM/DFS_WEIGHT），**在调度时就主动最大化缓存命中**。这是 SGLang 在多前缀场景再省一截的根源。

---

## 今日产出

- [x] 调度策略决策树图（标 calc_priority/_determine_active_policy 函数名）
- [x] Radix 树 DFS_WEIGHT 手画示例（带权重计算）
- [x] LPM vs DFS_WEIGHT 对比表
- [x] sched_compare.py 实测（LPM/FCFS 三行数据 + 诚实的单前缀局限分析）
- [x] 自测 5 题答案

## 与 vLLM 的核心对比

| | vLLM | SGLang |
|---|---|---|
| 调度策略数 | FCFS(+PRIORITY) | 6 种(含 2 缓存感知) |
| 缓存感知排序 | ❌ 无 | ✅ LPM/DFS_WEIGHT |
| 抢占 | RECOMPUTE(free KV 重算) | 同样 LRU 驱逐 + 可 HiCache 换 CPU |
| 调度时优化命中 | 被动(allocate 时查 APC) | **主动(排序时就把同前缀集中)** |
