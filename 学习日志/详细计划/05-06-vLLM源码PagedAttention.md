# 05-06 详细计划：vLLM PagedAttention 源码精读

## 步骤 1：先动手定位真实文件（20 min）

不要靠记忆找路径，直接搜：

```bash
# vLLM 安装路径
python3 -c "import vllm; print(vllm.__file__)"
VLLM=$(python3 -c "import vllm; print(vllm.__file__.replace('/__init__.py',''))")

# 找 block/KV cache 核心文件
grep -rn "num_gpu_blocks" $VLLM/v1/core/ --include="*.py" -l
grep -rn "free_list"      $VLLM/v1/core/ --include="*.py" -l
grep -rn "block_hash"     $VLLM/v1/core/ --include="*.py" -l 2>/dev/null || \
grep -rn "prefix_hash"    $VLLM/v1/core/ --include="*.py" -l 2>/dev/null
```

预期找到的核心文件（路径因版本略有差异）：
- `vllm/v1/core/kv_cache_manager.py`  — 分配/释放/哈希主逻辑
- `vllm/v1/core/block_pool.py` 或 `kv_cache_utils.py` — free list、Block 数据结构

把实际路径记进笔记，后面所有步骤以你找到的真实文件为准。

## 步骤 2：读 Block 数据结构（20 min）

在找到的文件里搜 `class KVCacheBlock` 或 `class Block`，重点看这几个字段：

```
block_id      — 物理块编号（0 ~ num_gpu_blocks-1）
ref_count     — 引用计数（>0 表示有请求在用，不能淘汰）
block_hash    — APC 哈希值（None 表示未缓存/最后一个不满块）
prev_block    — 指向前一个块（构成哈希链）
token_ids     — 块内 token id 列表
```

在笔记里画出一个请求的 Block 链：

```
Block#7 ──prev──► Block#12 ──prev──► Block#3
hash=H0           hash=H1             hash=None（最后块未满，不缓存）
ref=0             ref=0               ref=1（请求仍在 decode）
```

## 步骤 3：读 allocate / free（40 min）

找 `allocate_slots` 或 `allocate` 方法，逐行读，在旁边注释每一步在做什么：

```python
# 伪代码结构，实际代码对照着读：
def allocate_slots(request, num_new_tokens):
    # 1. 先查 APC 缓存：已有的前缀块能命中多少？
    computed_blocks = self._get_computed_blocks(request)
    # 2. 剩余 token 需要几个新块？
    num_new_blocks = ceil((num_new_tokens - 命中长度) / block_size)
    # 3. 从 free_list 取块
    new_blocks = self.block_pool.allocate(num_new_blocks)
    # 4. 返回 (命中的块 + 新块) → 这就是 Block Table
    return computed_blocks + new_blocks
```

重点理解：**分配永远先查缓存，查完再从 free_list 补齐**。

找 `free_blocks` 方法：

```python
# 释放：引用计数 -1，到 0 时放回 free_list（如果有 APC 哈希则留在缓存区，等 LRU 淘汰）
def free_blocks(blocks):
    for b in blocks:
        b.ref_count -= 1
        if b.ref_count == 0:
            free_list.append(b)   # 有哈希的块保留哈希，可被下次命中
```

关键细节：**有哈希的块放回 free_list 后哈希不清除**，所以下次 allocate 时仍能命中——这就是 APC "缓存不立即丢弃"的代码实现。

## 步骤 4：读 APC 哈希计算（30 min）

搜 `_compute_hash` 或 `block_hash`，找到哈希是怎么算的：

```python
# 链式哈希的核心：
hash(block_i) = hash( (prev_block_hash, tuple(token_ids)) )
```

用 Python 在脚本里手动跑一次，验证链式性质：

```python
# verify_hash.py
if __name__ == "__main__":
    # 模拟 token ids（每块 16 个）
    block0_tokens = list(range(16))       # system prompt 第1块
    block1_tokens = list(range(16, 32))   # system prompt 第2块

    h0 = hash((None, tuple(block0_tokens)))
    h1 = hash((h0,   tuple(block1_tokens)))
    print(f"block0 hash: {h0}")
    print(f"block1 hash: {h1}")

    # 改动 block0 的第1个 token
    block0_modified = [99] + list(range(1, 16))
    h0_mod = hash((None, tuple(block0_modified)))
    h1_mod = hash((h0_mod, tuple(block1_tokens)))
    print(f"\nblock0改1字后 h0: {h0_mod} (变了: {h0_mod != h0})")
    print(f"block1 hash也变了: {h1_mod != h1}")
```

预期：改动第 1 块的 1 个 token，两块哈希全变——这正是"任何一个 token 不同，从那块起全作废"的代码层面解释。

## 步骤 5：和自己的流程图对照（20 min）

把 04-25 画的内存流程图拿出来，对照今天读的代码填入函数名：

```
请求到达
  ↓ allocate_slots()
逻辑块 → 物理块（Block Table）
  ↓ prefill + decode
每满一块 → _compute_hash() 填入 block.block_hash
  ↓ EOS
free_blocks() → ref_count=0 → 归还 free_list（哈希保留）
```

## 今日产出
- [x] 真实源码文件路径（kv_cache_utils.py / block_pool.py / kv_cache_manager.py + 行号）
- [x] Block 数据结构字段注释（纠正计划 3 处字段名：ref_cnt / _block_hash / prev-next_free_block）
- [x] allocate/free 逐行注释（先查缓存→补free list→满块算哈希；free 只减ref_cnt不碰哈希）
- [x] verify_hash.py 运行结果（改1token两块哈希全变✓，部分命中[True,True,False]✓）
- [x] 流程图标注函数名版本

> 完成于 2026-06-14。完整笔记：[[vLLM源码精读-PagedAttention内存管理-06-14]]
> 核心顿悟：PagedAttention = 操作系统页分配器（free list/引用计数/LRU/COW/惰性失效一一对应）。
