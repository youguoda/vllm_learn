---
title: vLLM 源码精读 —— PagedAttention 内存管理（Block / allocate / free / 链式哈希）
date: 2026-06-14
tags:
  - R/技术框架/vLLM
  - R/核心算法/LLM推理
  - R/源码精读
  - output/active
related:
  - "[[vLLM-PagedAttention实测-06-13]]"
  - "[[vLLM-PrefixCaching实测-06-13]]"
  - "[[M1阶段总结-初体验]]"
---

# vLLM 源码精读：PagedAttention 内存管理

> M2 第一站。M1 用黑盒验证了 PagedAttention"是什么"（碎片 98.8%→21.9%、命中率 82.1%），这次钻进源码看"怎么实现"。vLLM 0.22.1，v1 引擎。
>
> 读完最大的感受：**PagedAttention 的内存管理本质上就是一个"操作系统的物理页分配器"**——free list、引用计数、LRU 淘汰、写时哈希，每一个都能在操作系统课本里找到对应。下面会生动讲这个设计哲学。

---

## 步骤 1：真实源码文件（实测路径）

用 `grep` 定位，不靠记忆：

```bash
VLLM=$(python3 -c "import vllm; print(vllm.__file__.replace('/__init__.py',''))")
grep -rln "num_gpu_blocks" $VLLM/v1/core/ --include="*.py"
```

三个核心文件（vLLM 0.22.1）：

| 文件 | 职责 | 关键内容（行号） |
|---|---|---|
| `v1/core/kv_cache_utils.py` | Block 数据结构 + free list + 哈希算法 | `KVCacheBlock`(115)、`FreeKVCacheBlockQueue`(164)、`hash_block_tokens`(541) |
| `v1/core/block_pool.py` | 块池：分配/释放/缓存/淘汰 | `get_new_blocks`(333)、`touch`(402)、`free_blocks`(419)、`_maybe_evict_cached_block`(365) |
| `v1/core/kv_cache_manager.py` | 上层编排：allocate_slots / free | `get_computed_blocks`(194)、`allocate_slots`(236)、`free`(429) |

> 调用层级：`kv_cache_manager`（编排）→ `coordinator` → `block_pool`（真正动手）→ `FreeKVCacheBlockQueue`（链表）。

---

## 步骤 2：Block 数据结构 —— 一块 KV 的"户口本"

`KVCacheBlock`（kv_cache_utils.py:115），实测字段**和计划预期略有出入**（以真实代码为准）：

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                      # 物理块编号 0 ~ num_gpu_blocks-1
    ref_cnt: int = 0                   # 引用计数 (计划写的 ref_count, 实际是 ref_cnt)
    _block_hash: BlockHashWithGroupId | None = None   # APC 哈希, 只有【满块】才有
    prev_free_block: "KVCacheBlock | None" = None     # ← 双向链表指针
    next_free_block: "KVCacheBlock | None" = None     # ← 双向链表指针
    is_null: bool = False              # 哨兵空块, 永不缓存
```

### 关键认知 1：block 里【不存 token】

计划预期有个 `token_ids` 字段，**实际没有**。token 只参与哈希计算，块本身只是"一格物理显存的元数据"。token→物理块的映射在别处（Request.block_hashes + cached_block_hash_to_block 字典）。

### 关键认知 2：哈希 setter 有"防覆盖"断言

```python
@block_hash.setter
def block_hash(self, block_hash):
    assert self.block_hash is None, "The block already has a hash..."
    self._block_hash = block_hash
```

> 一个块一旦算出哈希就不许改，必须先 `reset_hash()` 清空。这是防御式编程——哈希是缓存命中的身份证，绝不能在使用中被悄悄篡改。

### 关键认知 3：free list 是手写的双向链表（设计精华）

`FreeKVCacheBlockQueue`（164）。注释解释了为什么**不用** Python 内置 `deque`：

> "to support removing a block in the middle of the queue in O(1) time"

**为什么需要"从中间 O(1) 删除"？** 因为缓存命中时，一个待淘汰的块（在 free list 中间）要被"复活"——必须 O(1) 把它从队列里抽走。deque 做不到中间删除，所以 vLLM 自己用 `prev/next` 指针手搓了一个带**假头假尾哨兵**的双向链表：

```
fake_head ⇄ [块3] ⇄ [块7] ⇄ [块12] ⇄ fake_tail
            ↑队头(LRU,先淘汰)        ↑队尾(刚释放)

popleft()  → 从队头取(最久未用) → 分配新块时用
append()   → 加到队尾          → 释放块时用
remove(b)  → O(1) 抽走任意块    → 缓存命中复活时用
```

> **LRU 顺序**：队头 = 最久未用 = 优先淘汰。释放一个请求的块时**倒序**放回（尾块先进队头先淘汰），所以共享前缀（头块）活得最久——这跟昨天读 SGLang radix 树的"叶子优先淘汰"是同一个思想。

---

## 步骤 3：allocate / free —— 分配与释放的精确流程

### allocate_slots（kv_cache_manager.py:236）

源码注释里有张精妙的 token 布局图：

```
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
  已算的     刚命中缓存的    外部缓存的     要算的    投机解码预留
```

核心流程（去掉边界处理后的骨架）：

```python
def allocate_slots(request, num_new_tokens, new_computed_blocks, ...):
    # 1. 算需要几个新物理块
    num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(...)

    # 2. 【守卫】free 块不够 → 直接返回 None (上层会触发抢占/排队)
    if num_blocks_to_allocate > block_pool.get_num_free_blocks():
        return None

    # 3. 先把【命中的缓存块】挂到请求上 (touch: 复活并 ref_cnt+1)
    if new_computed_block_list 非空:
        coordinator.allocate_new_computed_blocks(...)

    # 4. 再从 free_list 取【新块】补齐 (get_new_blocks)
    new_blocks = coordinator.allocate_new_blocks(...)

    # 5. 把已满的块算哈希、登记进缓存字典 (cache_blocks)
    coordinator.cache_blocks(request, num_tokens_to_cache)
    return new_blocks
```

> **印证计划的核心论点**：分配永远是"**先查缓存（get_computed_blocks → touch 复活命中块），再从 free_list 补齐（get_new_blocks）**"。命中的块不重新算 KV，直接复用——这就是 APC 省 prefill 的代码实现。

**get_new_blocks（block_pool.py:333）—— 分配时的"驱逐"**：

```python
def get_new_blocks(num_blocks):
    ret = self.free_block_queue.popleft_n(num_blocks)   # 队头取(LRU)
    for block in ret:
        self._maybe_evict_cached_block(block)  # ← 取到的块若有旧哈希, 驱逐它!
        block.ref_cnt += 1                     # 标记为"在用"
    return ret
```

> 关键细节：从 free list 取出的块**可能还带着上一个请求的哈希**（缓存未失效）。一旦要把它分配给新请求，就调 `_maybe_evict_cached_block` 把旧哈希从缓存字典里删掉、`reset_hash()`。**这就是 LRU 淘汰的真正发生点——不是定时清理，而是"被复用挤占时才驱逐"。**

### free（kv_cache_manager.py:429 → block_pool.py:419）

```python
def free_blocks(ordered_blocks):
    for block in ordered_blocks:
        block.ref_cnt -= 1                    # 引用计数 -1
    # 只有 ref_cnt 归零的块, 才放回 free_list 队尾
    self.free_block_queue.append_n(
        [b for b in blocks if b.ref_cnt == 0 and not b.is_null]
    )
```

> **最妙的设计**：`free` 只做 `ref_cnt-1` 和"放回队尾"，**完全没碰哈希**！块的 `_block_hash` 还留着，它在 `cached_block_hash_to_block` 字典里的登记也还在。所以这个块虽然"空闲了"，下一个带相同前缀的请求**仍能命中它、touch 复活它**。
>
> 哈希什么时候才真正清除？→ 只有当它被 `get_new_blocks` 取走分配给别人时（`_maybe_evict_cached_block`）。**这就是"缓存不立即丢弃，等被挤占才失效"的精确实现。**

### touch（block_pool.py:402）—— 缓存命中的"复活术"

```python
def touch(blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)   # ← O(1) 从 free list 中间抽走
        block.ref_cnt += 1
```

> 这就是步骤 2 里"为什么 free list 要支持 O(1) 中间删除"的答案：命中的块此刻正躺在 free list 里待淘汰，touch 把它 O(1) 抽出来、ref_cnt+1 转为"在用"。**命中 = 把待回收的块抢救回来。**

---

## 步骤 4：链式哈希 —— "差一 token 全失效"的代码根源

`hash_block_tokens`（kv_cache_utils.py:541），核心就一行：

```python
return BlockHash(
    hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
)
#                  ↑ 父块哈希      ↑ 本块token        ↑ 额外key(lora等)
```

**`hash(块i) = hash( (hash(块i-1), 本块token, extra) )`** —— 父块哈希被"编织"进子块哈希。

### 用 verify_hash.py 实测验证

复刻这个算法跑了一遍（[verify_hash.py](../verify_hash.py)），三个结论：

```
[破坏] 只改 block0 的第 1 个 token (0 → 99):
  block0 hash 变了? True
  block1 hash 变了? True   ← block1 的 token 一个没动, 但父哈希变了, 哈希也变!

[复用] 完全相同的前缀 → 三块哈希完全一致 → True (能命中)

[部分命中] 前2块相同、第3块不同 → 逐块比对 [True, True, False]
                                    → 命中到分歧点为止
```

> **这就是 06-13 两个实验现象的代码层解释**：
> - "system prompt 加随机字符 → 命中率归零"：第一个块哈希变了，链式传导，后面全作废。
> - "B 共享截断前缀命中率 82.1%"：截断点不在块边界，最后那个不满块（`block_hash=None`）本就不缓存，加上块粒度对齐损失，命中到分歧块为止。

为什么要链式而不是每块独立哈希？→ **保证上下文正确性**。token 的 KV 依赖它前面所有 token（因果注意力）。两个"局部 token 相同但前文不同"的块，KV 其实不同，绝不能复用。链式哈希让"哈希相同"严格等价于"从序列开头到这里完全一致"。

---

## 步骤 5：内存流程图（标注真实函数名）

把 M1 的内存流程图，对照源码填上真实函数名：

```
请求到达
  │
  ├─► get_computed_blocks()          [kv_cache_manager.py:194]
  │     └─ find_longest_cache_hit()  查 APC 缓存, 命中多少前缀块
  │
  ├─► allocate_slots()               [kv_cache_manager.py:236]
  │     ├─ if 块不够: return None ──► 上层抢占/排队
  │     ├─ touch(命中块)              [block_pool.py:402]  命中块复活, ref_cnt+1
  │     │    └─ free_block_queue.remove()  O(1) 从待淘汰队列抽出
  │     └─ get_new_blocks(新块数)     [block_pool.py:333]
  │          ├─ free_block_queue.popleft_n()   队头取(LRU)
  │          └─ _maybe_evict_cached_block()    取到的旧缓存块 → 驱逐+reset_hash
  │
  ├─► prefill + decode
  │     └─ 每满一块 → cache_full_blocks() → hash_block_tokens() 算哈希填入 block
  │                                          [kv_cache_utils.py:541]
  │
  └─► EOS, free()                    [kv_cache_manager.py:429]
        └─ free_blocks()             [block_pool.py:419]
             ├─ ref_cnt -= 1
             └─ ref_cnt==0 → free_block_queue.append_n()  放回队尾(哈希保留!)
```

---

## 设计哲学：PagedAttention 就是一个"操作系统页分配器"

读完代码，最大的顿悟是——**vLLM 团队把操作系统管物理内存的那套，原封不动搬来管 KV cache**。一一对应：

| 操作系统概念 | vLLM 对应 | 代码 |
|---|---|---|
| 物理页 (page frame) | KVCacheBlock (16 token) | `KVCacheBlock` |
| 空闲页链表 (free list) | FreeKVCacheBlockQueue | 手写双向链表 |
| 页表 (page table) | Block Table (逻辑块→物理块) | Request.block_hashes |
| 引用计数 (共享内存) | ref_cnt | `touch` / `free_blocks` |
| 页面置换 (LRU) | free list 队头淘汰 | `popleft_n` |
| 写时复制 (COW) | 命中块共享 + 满块才算哈希 | `_maybe_evict_cached_block` |
| 缺页中断后分配 | allocate_slots 返回 None → 抢占 | `return None` |

### 三个让我拍案的设计决策

**1. "释放"不等于"清除"——惰性失效（lazy invalidation）**
> free 只把 ref_cnt 减到 0、放回队尾，哈希原封不动。块在"空闲但仍可命中"的状态待着，直到被别人挤占才真正失效。这跟操作系统"进程退出后页内容还在，直到被重新分配才覆盖"一模一样。**好处：释放是 O(1) 的纯指针操作，缓存价值能榨到最后一刻。**

**2. 自己手搓双向链表，只为 O(1) 中间删除**
> 没有偷懒用 deque，因为"缓存命中复活"需要从队列中间 O(1) 抽块。为了这个操作，宁可手写带哨兵的链表、连 Python 对象都不额外分配（注释明说为了贴近 C++ deque 性能）。**这是"数据结构服从访问模式"的典范——先想清楚最高频的操作是什么，再选数据结构。**

**3. 链式哈希把"正确性"焊死在哈希里**
> 不是"先复用再校验对不对"，而是让哈希的定义本身保证"哈希相同 ⟺ 前缀完全一致"。正确性不靠运行时检查，靠数据结构的数学性质。**这是最高级的防御——让错误在物理上无法发生**（跟昨天 SGLang 约束解码"把非法 token 概率置 -∞"是同一种哲学：不靠约束，靠物理不可能）。

### 一句话总结

> **PagedAttention 不是什么新算法，而是把操作系统六十年沉淀的内存管理智慧，精准移植到 KV cache 上。** 它快，是因为站在了巨人的肩膀上——free list、引用计数、LRU、COW、惰性失效，每一个都是被验证过无数次的经典。读它的源码，像在读一本浓缩的 OS 教科书。

---

## 今日产出

- [x] **真实源码路径**：kv_cache_utils.py / block_pool.py / kv_cache_manager.py（含关键行号）
- [x] **Block 数据结构注释**：ref_cnt / _block_hash / prev-next_free_block，纠正了计划里 3 处字段名出入
- [x] **allocate/free 逐行注释**：先查缓存→补 free list→满块算哈希；free 只减 ref_cnt 不碰哈希
- [x] **verify_hash.py 运行结果**：改 1 token 两块哈希全变 ✓，部分命中 [True,True,False] ✓
- [x] **流程图标注函数名版本**（步骤 5）

## M2 下一站

- [ ] 05-07：SGLang radix_cache.py，对比"树"和"块"两种缓存数据结构
- [ ] 抢占（preemption）时 KV 是重算还是换出 CPU？→ sched/scheduler.py
- [ ] `get_num_blocks_to_allocate` 在多 KV cache group（混合注意力）下怎么算
