# 05-14 详细计划：SGLang 调度器源码 · Cache-Aware Scheduling

> 衔接昨日（05-13）：昨天读懂了 vLLM 的 `_schedule()`——FCFS + Block 级 APC 哈希匹配。
> 今天读 SGLang 的 `SchedulePolicy`，理解它为什么能在同等硬件上把多轮对话 TTFT 再压一截。
> 关键源文件：`sglang/srt/managers/schedule_policy.py`

---

## 步骤 1：定位文件（15 min）

```bash
# 激活 SGLang 环境
source ~/venv-sglang/bin/activate

SGLANG=$(python3 -c "import sglang; print(sglang.__file__.replace('/__init__.py',''))")
echo $SGLANG   # → ~/venv-sglang/lib/.../sglang

# 两个核心文件
ls $SGLANG/srt/managers/schedule_policy.py   # 调度策略（今天重点）
ls $SGLANG/srt/managers/scheduler.py         # 调度主循环
ls $SGLANG/srt/mem_cache/radix_cache.py      # Radix Tree 实现

# 快速看有哪些调度策略
grep -n "class Cache" $SGLANG/srt/managers/schedule_policy.py
grep -n "def _sort"   $SGLANG/srt/managers/schedule_policy.py
grep -n "calc_priority\|_compute_prefix" $SGLANG/srt/managers/schedule_policy.py | head -15
```

预期输出（源码已确认）：
```
CacheAwarePolicy:   LPM, DFS_WEIGHT
CacheAgnosticPolicy: FCFS, LOF, RANDOM, ROUTING_KEY
```

把实际行号记进笔记。

---

## 步骤 2：读 SchedulePolicy 的核心入口（50 min）

### 2-A：`calc_priority()` ——调度主入口

```python
# schedule_policy.py  ~L130
def calc_priority(self, waiting_queue, running_batch=None):
    policy = self._determine_active_policy(waiting_queue)
    # ↑ 关键：等待队列 > 128 时自动降级为 FCFS
    # 原因：LPM 对所有等待请求都要做前缀匹配，O(N) 成本太高

    if isinstance(policy, CacheAwarePolicy):
        temporary_deprioritized = self._compute_prefix_matches(waiting_queue, policy)
        if policy == CacheAwarePolicy.LPM:
            self._sort_by_longest_prefix(waiting_queue, temporary_deprioritized)
        elif policy == CacheAwarePolicy.DFS_WEIGHT:
            self._sort_by_dfs_weight(waiting_queue, self.tree_cache)
```

**在笔记里画决策树**：

```
waiting_queue 长度 > 128?
    YES → FCFS（跳过昂贵的前缀匹配）
    NO  → 看配置的 policy:
            LPM       → 按最长前缀命中排序
            DFS_WEIGHT → 按 Radix Tree DFS 权重排序  ← 论文 Theorem 3.1
            FCFS/LOF/RANDOM → 无缓存感知
```

### 2-B：`_compute_prefix_matches()` ——批内前缀缓存（重要细节）

这段代码实现了论文里没有明确提到的工程优化——**in-batch prefix caching**：

```python
def _compute_prefix_matches(self, waiting_queue, policy):
    temporary_deprioritized = set()
    self.waiting_queue_radix_tree.reset()  # 临时 Radix Tree，仅用于本批

    for r in waiting_queue:
        # 1. 查全局缓存（已有 KV 块）
        match_prefix_for_req(self.tree_cache, r)

        if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
            # 2. 全局缓存命中不足 → 再查等待队列内部
            in_batch_match = self.waiting_queue_radix_tree.match_prefix(r)

            if len(in_batch_match) >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD:
                # 3. 等待队列里有其他请求已占住同一前缀
                #    → 本请求降优先级（等那个请求跑完后缓存就有了）
                temporary_deprioritized.add(r.rid)
            else:
                # 4. 第一个出现该前缀的请求 → 插入临时树，标记为"占位"
                self.waiting_queue_radix_tree.insert(r)

    return temporary_deprioritized
```

**关键理解**：两个阈值都是环境变量可调：
- `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = 32`（默认）：全局命中 ≤ 32 token 才触发批内检查
- `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = 32`：批内共享前缀 ≥ 32 token 才降优先级

---

## 步骤 3：DFS_WEIGHT 策略深读（40 min）——Theorem 3.1 的代码实现

这是 SGLang 论文（arXiv:2312.07104）§3.2 Cache-Aware Scheduling 的直接实现。

### 3-A：理解 `_sort_by_dfs_weight()`

```python
@staticmethod
def _sort_by_dfs_weight(waiting_queue, tree_cache):
    # Step 1: 按 last_node 分组
    # last_node = 每个请求的前缀在 Radix Tree 中命中到的最深节点
    last_node_to_reqs = defaultdict(list)
    for req in waiting_queue:
        last_node_to_reqs[req.last_node].append(req)

    # Step 2: 计算每个节点的权重 = 该子树下的等待请求数
    node_to_weight = defaultdict(int)
    for node in last_node_to_reqs:
        node_to_weight[node] = len(last_node_to_reqs[node])
    _calc_weight(tree_cache.root_node, node_to_weight)
    # _calc_weight 是后序遍历：子节点权重向上累加

    # Step 3: DFS 遍历，按子树权重降序选分支
    waiting_queue.clear()
    _get_dfs_priority(tree_cache.root_node, node_to_weight,
                      last_node_to_reqs, waiting_queue)
```

**在笔记里画示例**（手画 Radix Tree）：

```
              root
             /    \
        [A,B,C]  [X,Y]
        /    \
    [D,E]   [F]
    (req1)  (req2,req3)

_calc_weight 后：
  root = 3, [A,B,C] = 3, [D,E] = 1, [F] = 2, [X,Y] = 0

DFS 优先走 weight 大的子树：
  先遍历 [A,B,C] 分支（weight=3）
    其中先走 [F]（weight=2）→ 输出 req2, req3
    再走 [D,E]（weight=1）→ 输出 req1
  再遍历 [X,Y] 分支（weight=0）
```

**为什么这样做最优？（Theorem 3.1 直觉）**

把共享同一前缀的请求集中在一起调度，前缀对应的 KV 块在它们全跑完之前不会被 LRU 淘汰。反之，若打乱顺序穿插运行，前缀块可能在中间被驱逐，后续请求不得不重算。

### 3-B：LPM vs DFS_WEIGHT 对比

| 维度 | LPM | DFS_WEIGHT |
|------|-----|-----------|
| 排序依据 | `num_matched_prefix_tokens`（绝对长度） | Radix Tree DFS 权重（相对拓扑） |
| 适用场景 | 少量请求、前缀分布分散 | 多请求、前缀有树状层次结构 |
| 计算成本 | 线性排序 O(N log N) | DFS 遍历 O(N + 树节点数) |
| 论文依据 | 贪心近似 | Theorem 3.1（最优调度顺序） |
| 默认策略 | 是（`--schedule-policy lpm`） | 需手动开启 `dfs-weight` |

---

## 步骤 4：对比实验——vLLM FCFS vs SGLang LPM（40 min）

目标：亲眼看到调度策略对 TTFT 的影响。用**多个共享长前缀的请求**来拉大差距。

```bash
# 终端 1：启动 SGLang（LPM 策略）
source ~/venv-sglang/bin/activate
HF_HUB_OFFLINE=1 NO_PROXY="*" python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-1.5B \
  --schedule-policy lpm \
  --port 30000 2>&1 | tee /tmp/sglang_sched.log &

sleep 90
```

```python
# sched_compare.py — 共享长前缀场景，对比 TTFT
import asyncio, time, openai

# 长共享前缀（约 200 token）
SYSTEM = "你是一位专业的气象分析师。以下是过去30天的逐小时气象观测数据：" + \
         ("温度22°C，湿度65%，气压1013hPa，风速3m/s；" * 40)

QUESTIONS = [
    "请分析最近一周的气温趋势并给出预测。",
    "湿度变化与降水概率的关系是什么？",
    "气压的波动规律说明了什么天气现象？",
    "请给出本月平均气象指标的统计摘要。",
    "风速数据是否有异常？",
    "综合以上数据，明天的天气预报是什么？",
    "气象数据中有无极端天气预警信号？",
    "请计算各指标的标准差。",
]

async def timed_request(client, i, question):
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": question},
        ],
        max_tokens=80,
        stream=False,
    )
    ttft = time.perf_counter() - t0
    print(f"[req {i:02d}] TTFT={ttft:.3f}s | {question[:20]}…")
    return ttft

async def run(base_url):
    client = openai.AsyncOpenAI(base_url=base_url, api_key="x")
    ttfts = await asyncio.gather(*[
        timed_request(client, i, q) for i, q in enumerate(QUESTIONS)
    ])
    print(f"\n平均 TTFT: {sum(ttfts)/len(ttfts):.3f}s  最大: {max(ttfts):.3f}s")

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000/v1"
    asyncio.run(run(url))
```

```bash
# 跑 SGLang LPM
NO_PROXY="*" python3 sched_compare.py http://localhost:30000/v1

# 停 SGLang，改策略为 FCFS 重跑对照
# （或直接和昨天跑的 vLLM FCFS 数据做对比）
```

**记录表格**（填入实测数据）：

| 策略 | 平均 TTFT (s) | 最大 TTFT (s) | 备注 |
|------|-------------|-------------|------|
| vLLM FCFS（昨日数据） | | | 无缓存感知 |
| SGLang FCFS | | | 缓存有，排序无 |
| SGLang LPM | | | 有缓存感知排序 |

---

## 步骤 5：PrefillAdder 快读 + 自测（15 min）

`PrefillAdder` 是每步调度时管理 token 预算的对象，理解它就能回答"为什么 long prefill 会拖慢 decode"：

```python
# PrefillAdder 核心属性
rem_total_tokens  # 总可用 token（空闲 + 可驱逐）
rem_input_tokens  # 本步最多 prefill 多少 token（max_prefill_tokens 参数）
rem_chunk_tokens  # Chunked Prefill 时每块最多 token 数

# add_one_req() 返回值决定是否继续加请求：
AddReqResult.CONTINUE  # 继续加
AddReqResult.NO_TOKEN  # 总内存耗尽，停止
AddReqResult.OTHER     # token budget 耗尽（decode 没饿死），停止
```

**自测题（自问自答写进笔记）**：

1. SGLang 默认用哪种 cache-aware 策略？（答：`lpm`，见 `--schedule-policy lpm`）
2. `_determine_active_policy()` 什么时候降级为 FCFS？（答：等待队列 > 128 请求时）
3. In-batch prefix caching 解决了什么问题？（答：等待队列里多个请求共享同一个尚未缓存的前缀时，只让一个先跑，跑完后前缀就缓存了，后续请求直接命中，避免重复 prefill）
4. DFS_WEIGHT 和 LPM 的根本区别？（答：LPM 只看单个请求的绝对命中长度，DFS_WEIGHT 考虑整棵树的全局请求拓扑，把同一子树下的所有请求集中调度以防止 LRU 驱逐）
5. vLLM APC 和 SGLang RadixAttention 在调度层面的最大差异？（答：vLLM 块级哈希匹配 + FCFS；SGLang token 级 Radix Tree + cache-aware 排序，后者能在调度时就把缓存命中最大化）

---

## 今日产出
- [x] 调度策略决策树图（标 calc_priority/_determine_active_policy 函数名）→ [[SGLang调度器源码精读-CacheAware-06-14]]
- [x] Radix Tree DFS_WEIGHT 手画示例（带权重计算过程）
- [x] LPM vs DFS_WEIGHT 对比表
- [x] sched_compare.py 实测（LPM/FCFS + 诚实的单前缀局限分析）
- [x] 自测 5 题答案

> 完成于 2026-06-14。核心：SGLang 有 6 种调度策略(2 缓存感知)，vLLM 只有 FCFS。
> cache-aware 主动在排序时把同前缀集中(防 LRU 驱逐)，多前缀场景才显优势。
> 诚实发现：单前缀实验 LPM vs FCFS 不明显——cache-aware 需多竞争前缀才见效。
