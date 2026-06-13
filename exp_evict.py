"""
exp_evict.py — 观察 SGLang radix 树被迫淘汰(evict)的现场

设计:
  第一批: 20 个【不同】的长前缀(各500字), 填满 GPU 显存里的 radix 树缓存
  第二批: 重放第一批的前缀, 此时部分早期前缀可能已被 LRU 淘汰
          → 若被淘汰, 重放时无命中, TTFT 回到"冷"水平

核心问题: 被淘汰的 KV 去哪了?
  - 默认(无 HiCache): 【直接丢弃】, 下次重放还要从头 prefill
  - 开 HiCache: 先搬到 CPU 内存, 重放时从 CPU 搬回 GPU (比重算快)

用法:
  NO_PROXY="*" python3 exp_evict.py
对照:
  起 server 时加 --enable-hierarchical-cache 再跑一次, 对比第二批 TTFT
"""

import random
import string
import time

import httpx

BASE_URL = "http://127.0.0.1:30000"
MODEL = "Qwen/Qwen2.5-1.5B"

client = httpx.Client(trust_env=False, timeout=120)


def rand_prefix(n_char: int) -> str:
    # 混入中英文, 保证每个前缀都不同(无共享, 强制各占一片 KV)
    return "".join(random.choices(string.ascii_lowercase + "你我他的工业质检", k=n_char))


def ask(system_prompt: str) -> float:
    """发一条请求, 返回 TTFT(秒)。用 stream 测首 token。"""
    t0 = time.perf_counter()
    first = None
    with client.stream(
        "POST", f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "用一句话总结。"},
            ],
            "max_tokens": 32, "temperature": 0.0, "stream": True,
        },
    ) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                if first is None:
                    first = time.perf_counter() - t0
    return first if first is not None else (time.perf_counter() - t0)


def main():
    import sys
    n_prefix = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    char_len = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    random.seed(42)
    prefixes = [rand_prefix(char_len) for _ in range(n_prefix)]
    print(f"  (配置: {n_prefix} 个前缀 × {char_len} 字)")

    print("=" * 60)
    print("  SGLang radix 树淘汰观察 (exp_evict)")
    print("=" * 60)

    print("\n=== 第一批: 20 个不同长前缀, 填满缓存 (都是冷) ===")
    batch1 = []
    for i, p in enumerate(prefixes):
        t = ask(p)
        batch1.append(t)
        print(f"  请求{i:02d}  TTFT={t*1000:6.1f}ms")

    print("\n=== 第二批: 重放前 10 个前缀, 看命中/淘汰 ===")
    print("  (若前缀还在树上→命中→TTFT低; 若已被淘汰→无命中→TTFT回到冷水平)")
    batch2 = []
    for i, p in enumerate(prefixes[:10]):
        t = ask(p)
        batch2.append(t)
        print(f"  重放{i:02d}  TTFT={t*1000:6.1f}ms")

    # 统计
    b1_mean = sum(batch1) / len(batch1)
    b2_mean = sum(batch2) / len(batch2)
    print("\n  --- 统计 ---")
    print(f"  第一批(全冷)均值 TTFT: {b1_mean*1000:.1f}ms")
    print(f"  第二批(重放)均值 TTFT: {b2_mean*1000:.1f}ms")
    if b2_mean < b1_mean * 0.7:
        print(f"  → 第二批明显更快, 说明部分前缀仍在缓存命中")
    else:
        print(f"  → 第二批没快多少, 说明前缀大量被淘汰(evict), 重放要重算")
    print("\n  看 server 日志确认淘汰: grep -iE 'evict|cache hit|#cached-token' <日志>")


if __name__ == "__main__":
    main()
