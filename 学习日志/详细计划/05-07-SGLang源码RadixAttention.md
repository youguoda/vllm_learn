# 05-07 详细计划：SGLang RadixAttention 源码精读

## 步骤 1：定位核心文件（20 min）

```bash
source ~/venv-sglang/bin/activate
SGLANG=$(python3 -c "import sglang; print(sglang.__file__.replace('/__init__.py',''))")

# 找树的本体
find $SGLANG -name "*.py" | xargs grep -l "class RadixCache" 2>/dev/null
find $SGLANG -name "*.py" | xargs grep -l "match_prefix" 2>/dev/null
# 预期：sglang/srt/mem_cache/radix_cache.py
```

把路径记进笔记。

## 步骤 2：读树节点数据结构（20 min）

搜 `class TreeNode` 或 `class RadixNode`，重点字段：

```
children    — dict: token_tuple → 子节点（边标签就是那段 token）
parent      — 父节点指针
lock_ref    — 引用计数（>0 时不可淘汰，等同于 vLLM 的 ref_count）
last_access_time — LRU 时间戳
value       — 指向 KV 池的索引（命中时直接取这个）
```

在笔记里画出 04-29 手画的演化图的代码对应版：
- 边 = children 字典的 key（一段 token tuple）
- 节点 = TreeNode 对象，含 value 和 lock_ref

## 步骤 3：读 match_prefix（40 min）

这是最重要的函数，逐行读：

```python
# 伪代码，对照真实代码读：
def match_prefix(token_ids):
    node = self.root
    matched_len = 0
    while True:
        # 找能和当前 token 序列开头匹配的子边
        prefix, child = _find_longest_prefix_child(node, token_ids[matched_len:])
        if prefix is None: break          # 没有匹配了，停止
        node = child
        node.last_access_time = now()     # 刷新 LRU
        node.lock_ref += 1                # 引用计数+1，防止淘汰
        matched_len += len(prefix)
    return node, matched_len              # 返回命中的最深节点 + 命中长度
```

重点理解：**match 是"走到走不动为止"**，不要求对齐到任何边界——这就是 radix 树比块哈希粒度细的原因。

## 步骤 4：读 insert（30 min）

insert 是最复杂的函数，核心是 split 操作：

```python
# 伪代码：
def insert(token_ids, value):
    node, matched_len = match_prefix(token_ids)
    if matched_len == len(token_ids): return  # 已存在，不插
    remain = token_ids[matched_len:]

    # 检查是否需要 split：
    # 如果 node 的某个子边只匹配了一半，要把那条边劈开
    for key, child in node.children.items():
        common = longest_common_prefix(key, remain)
        if 0 < len(common) < len(key):
            # split: 旧边 key → [common部分新节点] + [剩余部分child]
            mid = TreeNode()
            mid.children[key[len(common):]] = child
            node.children[common] = mid
            node = mid
            remain = remain[len(common):]
            break
    # 挂新叶子
    leaf = TreeNode(value=value)
    node.children[remain] = leaf
```

动手验证：在 Python 里构造一个极简 radix 树，手动 insert 04-29 的三个请求，打印树结构：

```python
# tiny_radix.py
class Node:
    def __init__(self): self.children = {}; self.val = None
    def __repr__(self): return f"Node(children={list(self.children.keys())})"

def insert(root, tokens, val):
    node = root
    remain = tokens
    while remain:
        matched_key = None
        for key in list(node.children):
            # 找公共前缀
            common = []
            for a, b in zip(key, remain):
                if a == b: common.append(a)
                else: break
            common = tuple(common)
            if not common: continue
            if len(common) == len(key):
                node = node.children[key]
                remain = remain[len(key):]
                matched_key = key; break
            else:  # split
                old_child = node.children.pop(key)
                mid = Node()
                mid.children[key[len(common):]] = old_child
                node.children[common] = mid
                node = mid
                remain = remain[len(common):]
                matched_key = common; break
        if matched_key is None:
            leaf = Node(); leaf.val = val
            node.children[remain] = leaf; break

if __name__ == "__main__":
    SYS = (1,2,3,4,5)   # 模拟 system prompt token ids
    root = Node()
    insert(root, SYS + (10,11), "KV_A")
    insert(root, SYS + (20,21), "KV_B")
    insert(root, SYS + (10,99), "KV_C")

    def show(node, indent=0):
        for k, c in node.children.items():
            print(" "*indent + str(k) + (" val="+str(c.val) if c.val else ""))
            show(c, indent+4)
    show(root)
```

输出应该显示 SYS 这条公共边只出现一次，三个请求的差异部分各自分叉。

## 步骤 5：读 evict（30 min）

搜 `evict` 函数，理解 LRU 淘汰：

```python
# 核心逻辑：
# 1. 按 last_access_time 排序所有叶子
# 2. 从最久未访问的叶子开始，跳过 lock_ref > 0 的
# 3. 删叶子 → 父节点可能变叶子 → 继续候选
# 4. 释放足够的 KV 内存后停止
```

关键点写进笔记：**只淘汰叶子，且必须 lock_ref == 0**。这保证了正在被请求用的前缀永远不会被淘汰——和 vLLM ref_count 机制完全相同，只是实现层不同。

## 步骤 6：对照 vLLM 写差异表（20 min）

| 对比点 | vLLM (kv_cache_manager) | SGLang (radix_cache) |
|---|---|---|
| 索引结构 | hash table (dict) | radix tree |
| 命中粒度 | 16-token 整块 | 任意 token 前缀 |
| split 操作 | 不需要 | 有（边可以被劈开） |
| 引用计数字段 | block.ref_count | node.lock_ref |
| 淘汰入口 | free_list LRU | evict() 遍历叶子 |
| 哈希计算 | 链式 hash() | 不需要（树结构本身保证唯一性） |

## 今日产出
- [ ] 真实源码路径（记进笔记）
- [ ] TreeNode 字段注释
- [ ] match_prefix / insert / evict 逐行注释
- [ ] tiny_radix.py 运行结果（树结构正确分叉）
- [ ] vLLM vs SGLang 差异对照表
