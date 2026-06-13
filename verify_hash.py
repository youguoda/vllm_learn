"""
verify_hash.py — 验证 vLLM APC 链式哈希的"差一 token 全失效"性质

模拟 vLLM 真实源码 (vllm/v1/core/kv_cache_utils.py:hash_block_tokens) 的核心:
    hash(块i) = hash_function( (parent_block_hash, tuple(本块token), extra_keys) )

链式性质: 每个块的哈希【包含前一块的哈希】, 所以前缀任何一个 token 变了,
          从那个块起所有后续块的哈希全部改变 → 整条缓存链作废。

对应 06-13 的 PrefixCaching 实验: "前缀加一个随机字符 → 命中率归零"。
"""

# vLLM 用一个固定哨兵代替"第一块没有父哈希"的情况 (源码里叫 NONE_HASH)
NONE_HASH = hash("__vllm_none_hash_sentinel__")
BLOCK_SIZE = 16


def hash_block_tokens(parent_block_hash, curr_block_token_ids, extra_keys=None):
    """复刻 vllm/v1/core/kv_cache_utils.py:541 的链式哈希"""
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    return hash((parent_block_hash, tuple(curr_block_token_ids), extra_keys))


def hash_chain(all_token_ids):
    """把一串 token 按 BLOCK_SIZE 切块, 逐块算链式哈希, 返回每块的哈希"""
    hashes = []
    parent = None
    for i in range(0, len(all_token_ids), BLOCK_SIZE):
        block = all_token_ids[i:i + BLOCK_SIZE]
        if len(block) < BLOCK_SIZE:
            break  # 最后不满一块的不缓存 (block_hash=None)
        h = hash_block_tokens(parent, block)
        hashes.append(h)
        parent = h  # 链式: 本块哈希成为下一块的父哈希
    return hashes


def main():
    print("=" * 64)
    print("  vLLM APC 链式哈希验证 (block_size=16)")
    print("=" * 64)

    # ===== 基准: 两块 system prompt (32 token) =====
    block0_tokens = list(range(16))        # 第1块: token 0~15
    block1_tokens = list(range(16, 32))    # 第2块: token 16~31

    h0 = hash_block_tokens(None, block0_tokens)
    h1 = hash_block_tokens(h0, block1_tokens)  # 注意: 父哈希是 h0
    print("\n[基准] 原始前缀:")
    print(f"  block0 hash = {h0}")
    print(f"  block1 hash = {h1}  (父哈希 = block0)")

    # ===== 破坏: 只改 block0 的第 1 个 token =====
    block0_modified = [99] + list(range(1, 16))  # 第1个 token: 0 → 99
    h0_mod = hash_block_tokens(None, block0_modified)
    h1_mod = hash_block_tokens(h0_mod, block1_tokens)  # block1 token 没变, 但父哈希变了

    print("\n[破坏] 只改 block0 的第 1 个 token (0 → 99):")
    print(f"  block0 hash = {h0_mod}  (变了? {h0_mod != h0})")
    print(f"  block1 hash = {h1_mod}  (变了? {h1_mod != h1})")
    print(f"             ↑ block1 的 token 一个没动, 但因父哈希变了, 哈希也变了!")

    # ===== 结论 =====
    print("\n" + "-" * 64)
    both_changed = (h0_mod != h0) and (h1_mod != h1)
    print(f"  两块哈希都变了吗? {'是 ✓' if both_changed else '否'}")
    print("  → 这就是【链式哈希】: 改前缀任意一个 token,")
    print("    从那块起所有后续块的哈希全部作废 → 缓存命中归零")
    print("  → 对应实验: 'B 共享截断前缀' 命中率掉到 82.1%,")
    print("    'system prompt 加随机字符' 命中率直接归零")

    # ===== 额外验证: 相同前缀 → 相同哈希 (可复用) =====
    print("\n[复用验证] 完全相同的前缀, 哈希应完全一致:")
    chain_a = hash_chain(list(range(48)))  # 48 token = 3 块
    chain_b = hash_chain(list(range(48)))  # 同样的 48 token
    print(f"  请求A 三块哈希 == 请求B 三块哈希? {chain_a == chain_b} ✓")
    print(f"  → 相同前缀算出相同哈希, 所以 APC 能命中复用")

    # 部分相同: 前 2 块相同, 第 3 块不同
    chain_c = hash_chain(list(range(32)) + list(range(900, 916)))
    same_prefix = [a == c for a, c in zip(chain_a, chain_c)]
    print(f"\n[部分命中] 前 2 块相同、第 3 块不同:")
    print(f"  逐块是否相同 = {same_prefix}")
    print(f"  → 前 2 块命中(True), 第 3 块失效(False), 命中到分歧点为止")


if __name__ == "__main__":
    main()
