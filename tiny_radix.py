"""
tiny_radix.py — 极简 radix 树，复现 SGLang 的 split 操作

对照 SGLang 源码 (sglang/srt/mem_cache/radix_cache.py):
  - _match_prefix_helper / _insert_helper 里的核心:
    "沿树往下走, 遇到边只匹配一半就 _split_node 劈开"
  - 这就是 token 级粒度的本质: 不要求对齐任何边界, 任意位置可 split

模拟 04-29 手画的三请求演化:
  R1: SYS + (10,11)  → root 挂一条整边
  R2: SYS + (20,21)  → 与 R1 共享 SYS, 在 SYS 后 split 分叉
  R3: SYS + (10,99)  → 与 R1 共享 SYS+(10), 在 (10) 后再 split
"""


class Node:
    _id = 0

    def __init__(self):
        self.children = {}   # child_key(首token) → 子节点
        self.key = ()        # 边标签: 这一段 token
        self.val = None      # 模拟指向 KV 池的索引
        self.id = Node._id
        Node._id += 1

    def __repr__(self):
        return f"N{self.id}"


def lcp(a, b):
    """最长公共前缀"""
    out = []
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
        else:
            break
    return tuple(out)


def insert(root, tokens, val):
    """复刻 SGLang _insert_helper: 沿树下行, 匹配到边中间就 split"""
    node = root
    key = tuple(tokens)
    while key:
        first = key[0]
        if first not in node.children:
            break
        child = node.children[first]
        common = lcp(child.key, key)
        if len(common) < len(child.key):
            # === SPLIT: 边只匹配了一半, 在 len(common) 处劈开 ===
            # 对应 SGLang _split_node: 父=common, 子=剩余
            mid = Node()
            mid.key = common
            mid.val = ("split自", child.id)
            # 旧 child 变成 mid 的子节点, key 截掉公共部分
            child.key = child.key[len(common):]
            mid.children[child.key[0]] = child
            # mid 接到原位置
            node.children[first] = mid
            print(f"  [SPLIT] 边 {child.key if False else common+child.key} "
                  f"在第 {len(common)} 个 token 处劈开 → 父{mid} 子{child}")
            node = mid
            key = key[len(common):]
            break
        else:
            # 整条边匹配, 吃掉继续往下
            node = child
            key = key[len(common):]

    if key:
        leaf = Node()
        leaf.key = key
        leaf.val = val
        node.children[key[0]] = leaf


def show(node, indent=0):
    for k, c in node.children.items():
        tag = f"  val={c.val}" if c.val and not isinstance(c.val, tuple) else ""
        print(" " * indent + f"└─ {c.key}{tag}  [{c}]")
        show(c, indent + 3)


def main():
    SYS = (1, 2, 3, 4, 5)   # 模拟 system prompt 的 token ids
    root = Node()

    print("=" * 60)
    print("  极简 radix 树: 复现 SGLang 的 split")
    print("=" * 60)

    print("\n① insert R1 = SYS + (10,11):")
    insert(root, SYS + (10, 11), "KV_A")
    show(root)

    print("\n② insert R2 = SYS + (20,21)  (与 R1 共享 SYS, 应在 SYS 后分叉):")
    insert(root, SYS + (20, 21), "KV_B")
    show(root)

    print("\n③ insert R3 = SYS + (10,99)  (与 R1 共享 SYS+(10), 应再 split):")
    insert(root, SYS + (10, 99), "KV_C")
    show(root)

    print("\n" + "-" * 60)
    print("  观察:")
    print("  - SYS 这条公共边最终只出现一次 (被复用)")
    print("  - R1/R2 在 SYS 后分叉; R1/R3 在 SYS+(10) 后再分叉")
    print("  - split 发生在【任意 token 位置】, 不需要 16-token 块对齐")
    print("    → 这就是 radix 树比 vLLM 块哈希粒度细的根源")


if __name__ == "__main__":
    main()
