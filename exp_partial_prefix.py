"""
exp_partial_prefix.py — 构造"半块对齐"前缀，暴露 vLLM APC 块粒度命中的局限

原理:
  vLLM 的 APC 以 block(16 token) 为粒度缓存 KV，只能命中【整块】。
  - 请求 A 重复 → 整条前缀块对齐 → 几乎全命中，TTFT 骤降
  - 请求 B 在【非块边界】处截断共享前缀再接不同内容
    → 共享部分的最后一个"不满块"无法复用，只省到最后一个对齐块为止

对照 SGLang RadixAttention: 它按 token 前缀为粒度，匹配到一半可 split 节点,
任意长度前缀都能命中 —— 这正是 vLLM 块哈希做不到的。

用法: NO_PROXY="*" python3 exp_partial_prefix.py
"""

import re
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen2.5-1.5B"

# 长共享前缀 (重复堆叠, 约几百 token)
BASE = "你是一位资深工业质检专家，精通缺陷检测、统计过程控制、六西格玛方法论。" * 10


def get_prefix_metrics(client):
    """抓 prefix cache 累计 queries/hits (单位: block)"""
    text = client.get(f"{BASE_URL}/metrics", timeout=5).text
    def grab(name):
        m = re.search(rf'{name}\{{[^}}]*\}}\s+([\d.eE+]+)', text)
        return float(m.group(1)) if m else 0.0
    return grab("vllm:prefix_cache_queries_total"), grab("vllm:prefix_cache_hits_total")


def ttft(client, prompt):
    """stream 模式测首 token 延迟, 同时返回本条命中的 block 数"""
    q0, h0 = get_prefix_metrics(client)
    t0 = time.perf_counter()
    first = None
    with client.stream(
        "POST", f"{BASE_URL}/v1/completions",
        json={"model": MODEL, "prompt": prompt, "max_tokens": 16,
              "temperature": 0.0, "stream": True},
        timeout=60,
    ) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:].strip() == "[DONE]":
                break
            if first is None:
                first = time.perf_counter() - t0
    q1, h1 = get_prefix_metrics(client)
    return first, int(q1 - q0), int(h1 - h0)


def tok_count(client, prompt):
    try:
        r = client.post(f"{BASE_URL}/tokenize",
                        json={"model": MODEL, "prompt": prompt}, timeout=10)
        return r.json().get("count", "?")
    except Exception:
        return "?"


def run(client, label, prompt):
    t, q, h = ttft(client, prompt)
    hit_rate = (h / q * 100) if q else 0.0
    n = tok_count(client, prompt)
    print(f"  {label:22s} TTFT={t*1000:7.1f}ms | prompt≈{n:>4} tok | "
          f"query {q:>3} blk, hit {h:>3} blk ({hit_rate:5.1f}%)")
    return t, q, h


def main():
    client = httpx.Client(trust_env=False)

    print("=" * 78)
    print("  vLLM APC 块粒度命中实验 (block_size=16)")
    print("=" * 78)

    pA = BASE + "请分析轴承故障。"
    # B: 把共享前缀截断到【非块边界】(BASE 中点 +3 字符), 再接不同内容
    pB = BASE[: len(BASE) // 2 + 3] + "请分析齿轮磨损。"

    print(f"\n  BASE 长度 = {len(BASE)} 字符 ≈ {tok_count(client, BASE)} token")
    print(f"  B 截断点 = len(BASE)//2 + 3 = {len(BASE)//2 + 3} 字符 (故意不对齐块边界)\n")

    run(client, "A 冷启动", pA)
    run(client, "A 重复(应命中)", pA)
    run(client, "B 共享截断前缀", pB)
    # 再重复 B 一次, 作为"B 自己也能整块命中"的对照
    run(client, "B 重复(应命中)", pB)

    print("\n  解读:")
    print("  - A 重复: 整条前缀块对齐 → 命中率应接近 100%, TTFT 骤降")
    print("  - B 首次: 与 A 共享前半段, 但截断点不在 16-token 块边界")
    print("    → 共享部分只能命中到最后一个【完整对齐块】, 尾部不满块要重算")
    print("    → 命中率 < A 重复, TTFT 介于冷启动和全命中之间")


if __name__ == "__main__":
    main()
