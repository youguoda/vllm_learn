---
title: SGLang 源码精读 —— RadixAttention（TreeNode / match / split / evict）
date: 2026-06-14
tags:
  - R/技术框架/SGLang
  - R/核心算法/LLM推理
  - R/源码精读
  - output/active
related:
  - "[[vLLM源码精读-PagedAttention内存管理-06-14]]"
  - "[[RadixAttention多轮对话实验-06-13]]"
  - "[[SGLang论文-RadixAttention精读-06-13]]"
---

# SGLang 源码精读：RadixAttention

> M2 第二站。昨天读完 vLLM PagedAttention（哈希表 + 块），今天读 SGLang RadixAttention（radix 树 + token），正好形成"**块 vs 树**"的对照。SGLang 0.5.13。
>
> 一句话剧透：**vLLM 用哈希表做"精确查找"，SGLang 用 radix 树做"最长前缀匹配"。** 数据结构的不同选择，直接决定了缓存粒度——这是两个框架最本质的分野。

---

## 步骤 1：真实源码文件

```bash
source ~/venv-sglang/bin/activate
SGLANG=$(python3 -c "import sglang; print(sglang.__file__.replace('/__init__.py',''))")
grep -ln "class RadixCache" $SGLANG/srt/mem_cache/*.py
```

核心文件：`sglang/srt/mem_cache/radix_cache.py`（802 行）。三个关键 class + 方法：

| 结构/方法 | 行号 | 职责 |
|---|---|---|
| `class RadixKey` | 56 | 边标签的 token 序列 + extra_key（lora/salt 隔离） |
| `class TreeNode` | 201 | 树节点：children / value / lock_ref / last_access_time |
| `match_prefix` | 337 → `_match_prefix_helper`(622) | 最长前缀匹配（走到走不动） |
| `insert` | 397 → `_insert_helper`(678) | 插入 + split |
| `_split_node` | 648 | **劈边**——token 级粒度的精髓 |
| `evict` | 537 | LRU 叶子优先淘汰（最小堆） |
| `inc_lock_ref`/`dec_lock_ref` | 566/581 | 沿路径锁到 root |

---

## 步骤 2：数据结构 —— 树节点与"会算最长公共前缀的 key"

### TreeNode（radix_cache.py:201）

```python
class TreeNode:
    def __init__(self, ...):
        self.children = defaultdict(TreeNode)  # 边首token → 子节点
        self.parent = None                     # 父指针 (树!)
        self.key: RadixKey = None              # 边标签 = 一段 token
        self.value: torch.Tensor = None        # ← 指向 KV 池的索引
        self.lock_ref = 0                      # 引用计数 (= vLLM 的 ref_cnt)
        self.last_access_time = time.monotonic()  # LRU 时间戳
        self.host_value = None                 # ← KV 可换出到 CPU host (多一层!)

    @property
    def evicted(self):  return self.value is None
    def __lt__(self, other):  return self.last_access_time < other.last_access_time  # 可进堆
```

对比 vLLM 的 `KVCacheBlock`，几个关键差异：

| 字段 | SGLang TreeNode | vLLM KVCacheBlock |
|---|---|---|
| 拓扑 | `parent` + `children` → **树** | `prev/next_free_block` → **链表** |
| 边标签 | `key` 是**变长** token 段 | block 固定 16 token |
| KV 指向 | `value` (tensor 索引) | `block_id` |
| 引用计数 | `lock_ref` | `ref_cnt` |
| LRU | `last_access_time` + `__lt__`（堆） | free list 队列位置 |
| 分层 | `host_value`（可换 CPU） | 无（只在 GPU） |

> 关键：`__lt__` 按时间戳比较 → TreeNode 可直接丢进**最小堆**做 LRU。这跟 vLLM "手写双向链表"是两条完全不同的 LRU 实现路线。

### RadixKey（radix_cache.py:56）—— 藏着两个工程亮点

**亮点 1：`extra_key` 做命名空间隔离**

```python
class RadixKey:
    __slots__ = ("token_ids", "extra_key", "is_bigram")
    # extra_key: e.g. lora_id, cache_salt
```

`match_prefix` 的 docstring 明说：相同 token 但不同 `extra_key` 的请求**永不共享前缀节点**，用于隔离 LoRA / sampling salt / RAG context。

> 这正是 06-13 结构化输出实验里我用 `--salt` 隔离缓存的底层机制——salt 进了 extra_key，不同 salt 的前缀在树上是两条互不相干的分支。

**亮点 2：`match()` 用指数搜索 + 二分找分歧点（138 行）**

```python
def match(self, other, page_size=1) -> int:
    # 不是逐 token Python 循环! 而是:
    # 1. gallop: 翻倍窗口 [lo, lo+1, lo+2, lo+4...] 做 C 级切片比较
    # 2. 找到第一个不相等的窗口后, 在窗口内二分定位精确分歧点
```

> 为什么这么讲究？因为多轮对话的共享前缀可能上千 token，逐个比较太慢。指数搜索让"长前缀完全匹配"的常见情况只需 O(log n) 次 C 级切片比较，避免 Python 逐 token 慢循环。**这是把"最高频路径"优化到极致的典范。**

---

## 步骤 3：match_prefix —— "走到走不动为止"

`_match_prefix_helper`（622）是命中查找的心脏：

```python
def _match_prefix_helper(self, node, key):
    child_key = key.child_key(self.page_size)   # 用首 token 找子边
    value = []
    while len(key) > 0 and child_key in node.children:   # 还有 token 且有匹配的边
        child = node.children[child_key]
        prefix_len = child.key.match(key)        # 这条边能匹配多长?
        if prefix_len < len(child.key):
            # === 只匹配了边的一半 → split! ===
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break                                # split 后就停 (走不动了)
        else:
            # 整条边都匹配 → 吃掉, 继续往下走
            value.append(child.value)
            node = child
            key = key[prefix_len:]
    return value, node
```

> **这就是计划说的"走到走不动为止"，也是 radix 树粒度细的根源**：
> - 整条边匹配 → 继续深入
> - 匹配到边中间 → **当场 split，命中到精确的 token 位置**
> - 不像 vLLM 必须对齐 16-token 块边界
>
> 命中的 KV 索引（`value`）一路收集，`torch.cat` 拼成完整的命中前缀 KV——直接复用，跳过 prefill。

---

## 步骤 4：insert + split —— 边可以被"劈开"

`_split_node`（648）是 token 级粒度的物理实现：

```python
def _split_node(self, key, child, split_len):
    new_node = TreeNode()
    new_node.key   = child.key[:split_len]          # 父 = 共享前缀部分
    new_node.value = child.value[:split_len].clone() # KV 也跟着切!
    child.key      = child.key[split_len:]          # 子 = 剩余部分
    child.value    = child.value[split_len:].clone()
    new_node.children = {...: child}                # 原节点变成新节点的子
    new_node.parent.children[...] = new_node        # 接回树
    return new_node
```

> 一条边 `(1,2,3,4,5,10,11)` 在 `split_len=5` 处劈开 → 父 `(1,2,3,4,5)` + 子 `(10,11)`。**KV value 同步 `[:5]` / `[5:]` 切开**——这是 token 级，不受任何块对齐约束。

### tiny_radix.py 实测（复现两次 split）

写了极简版 [tiny_radix.py](../tiny_radix.py)，插入 04-29 手画图的三个请求：

```
① R1 = SYS+(10,11):
└─ (1,2,3,4,5,10,11)  val=KV_A

② R2 = SYS+(20,21):  [SPLIT] 边在第 5 个 token 处劈开
└─ (1,2,3,4,5)
   └─ (10,11)  val=KV_A
   └─ (20,21)  val=KV_B

③ R3 = SYS+(10,99):  [SPLIT] 边 (10,11) 在第 1 个 token 处再劈
└─ (1,2,3,4,5)
   └─ (10,)
      └─ (11,)  val=KV_A
      └─ (99,)  val=KV_C
   └─ (20,21)  val=KV_B
```

> 两次 split 分别发生在**第 5、第 1 个 token** 处——任意位置，无需块对齐。最终 SYS 公共边只出现一次，被三个请求共享。**这跟 06-13 手画的 radix 演化图完全吻合**，现在我知道它在代码里是怎么发生的了。

---

## 步骤 5：evict —— 最小堆 + 叶子优先 + 路径锁

`evict`（537）：

```python
def evict(self, num_tokens):
    leaves = list(self.evictable_leaves)         # 只取叶子(已排除 lock_ref>0)
    eviction_heap = [(priority, node) for node in leaves]
    heapq.heapify(eviction_heap)                 # ← 最小堆做 LRU

    while num_evicted < num_tokens and eviction_heap:
        _, x = heapq.heappop(eviction_heap)      # 弹出最久未用的叶子
        self.token_to_kv_pool_allocator.free(x.value)   # 释放 KV
        self._delete_leaf(x)
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            heapq.heappush(eviction_heap, (..., x.parent))  # 父变叶子→入堆候选
```

三个设计：
1. **只淘汰叶子**：`evictable_leaves` 集合，且已排除 `lock_ref>0` 的。
2. **最小堆做 LRU**：靠 TreeNode 的 `last_access_time`（vs vLLM 链表队头）。
3. **逐层向上**：删一个叶子后，若父节点变成叶子且未锁，重新入堆——保证靠近 root 的共享前缀**最后才被淘汰**。

### 路径锁（inc_lock_ref，566）—— 比 vLLM 多的一招

```python
def inc_lock_ref(self, node):
    while node != self.root_node:
        node.lock_ref += 1          # 从命中节点一路锁到 root
        node = node.parent
```

> 一个请求命中某节点，会把**从该节点到 root 的整条路径** lock_ref+1。这保证整条前缀链在请求运行期间都不被淘汰。vLLM 是单块各自 ref_cnt，SGLang 是"路径锁"——因为树有父子关系，锁住叶子必须锁住它的所有祖先。

---

## 步骤 6：vLLM vs SGLang 缓存机制差异表

| 对比点 | vLLM (PagedAttention) | SGLang (RadixAttention) |
|---|---|---|
| **索引结构** | hash table（`cached_block_hash_to_block` dict） | **radix 树**（TreeNode + children） |
| **查找方式** | 精确哈希查找 O(1) | **最长前缀匹配**（沿树下行 + split） |
| **命中粒度** | 16-token 整块（必须对齐） | **任意 token 前缀**（page_size=1） |
| **split 操作** | 不需要（块固定） | **有**（`_split_node` 劈边） |
| **唯一性保证** | 链式哈希（hash 含父块 hash） | **树结构天然保证**（路径即前缀，无需哈希） |
| **引用计数** | `block.ref_cnt`（单块） | `node.lock_ref`（**沿路径锁到 root**） |
| **LRU 实现** | 手写双向链表（队头淘汰） | **最小堆**（按 last_access_time） |
| **释放** | ref_cnt=0 放回 free list（哈希保留） | lock_ref=0 进 evictable，evict 时堆淘汰 |
| **分层缓存** | 无（只 GPU） | `host_value`（可换出 CPU host） |
| **额外隔离** | extra_keys（lora 等） | `extra_key`（lora/salt/RAG） |

实测印证（06-13 半块前缀实验）：vLLM 命中 96/117 = 82.1%，SGLang 命中 111/117 = 94.9%。**多命中的 15 token = vLLM 块对齐损失 = SGLang split 救回的部分。** 现在从源码知道了：SGLang 靠 `_split_node` 在第 96+x 个 token 处精确劈开，vLLM 的 hash table 只能查到块边界。

---

## 设计哲学：两种数据结构，两种世界观

读完两份源码，最深的体会是——**vLLM 和 SGLang 在缓存上的分野，根子是"选了不同的数据结构"**。

### 哈希表 vs radix 树：精确查找 vs 前缀匹配

```
vLLM 哈希表:  "给我一个块的哈希, 我 O(1) 告诉你它在不在"
              → 擅长精确匹配, 但只能以【整块】为单位提问
              → 像查字典: 必须给完整的词, 查不了"前 3 个字母"

SGLang radix树: "给我一串 token, 我告诉你最长能匹配多少"
              → 天生支持【最长前缀】, 任意位置可分叉
              → 像走迷宫: 沿着公共路径走, 走到岔路口为止
```

> **这不是谁更聪明，而是工具的取舍。** 哈希表查找快、实现简单、并发友好（vLLM 高并发跑分略优可能与此有关）；radix 树粒度细、共享充分，但要维护树结构、split 有开销、并发要加锁。

### 三个让我印象深刻的设计

**1. 树结构本身就是"哈希"——正确性免费**
> vLLM 要费力维护链式哈希来保证"哈希相同 ⟺ 前缀一致"。SGLang 不需要哈希——**从 root 到某节点的路径，本身就唯一地代表那段前缀**。树的拓扑结构天然编码了前缀关系，正确性是数据结构白送的。这是用"结构"换"计算"的典范。

**2. split：把"粒度细"做成一个 O(1) 指针操作**
> 粒度细听起来很贵，但 `_split_node` 只是改几个指针 + 切一下 tensor 视图。任意 token 位置命中，不需要重算、不需要对齐，只需在分歧点把一个节点裂成两个。**优雅之处在于：细粒度的代价被压缩成了一次局部的树修改。**

**3. 路径锁 + 叶子优先淘汰：让"共享前缀"自然沉淀到树根**
> 越靠近 root 的节点，被越多请求共享、被越多路径锁锁住、越不会被淘汰。LRU 只从叶子开始啃。于是高频共享的 system prompt 自然"沉"在树根活得最久，一次性的尾巴在叶子先被淘汰。**这是一种自组织——不需要谁去标记"这是热点前缀"，树的形状自己演化出了冷热分层。**

### 一句话总结

> **vLLM 把 KV 缓存当"物理内存页"管（哈希表 + free list），SGLang 把它当"前缀树"管（radix tree）。** 前者继承操作系统的内存管理智慧，后者继承编译器/字符串算法的 trie 智慧。同一个问题（KV 复用），两种经典数据结构，各自把一种做到极致——这就是为什么"无共享前缀时两者持平，高复用时 SGLang 占优"。

---

## 今日产出

- [x] **真实源码路径**：radix_cache.py（RadixKey:56 / TreeNode:201 / match:337 / insert:397 / split:648 / evict:537）
- [x] **TreeNode 字段注释**：children/parent/value/lock_ref/last_access_time/host_value，对照 vLLM
- [x] **match/insert/split/evict 逐行注释**：走到走不动 + 任意位置 split + 堆 LRU + 路径锁
- [x] **tiny_radix.py 运行结果**：两次 split（第5、第1个token处），SYS 公共边只出现一次 ✓
- [x] **vLLM vs SGLang 差异对照表**（10 个维度）

## M2 下一站

- [ ] 05-13：抢占（preemption）时 KV 重算 vs 换出——SGLang `host_value` 换 CPU 的具体路径
- [ ] SGLang scheduler 怎么用 match_prefix 的结果做 cache-aware 调度（longest-prefix-first）
- [ ] `cpp_radix_tree/` —— radix 树有没有 C++ 加速版？锁粒度在哪
