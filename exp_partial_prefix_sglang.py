"""
exp_partial_prefix_sglang.py — 半块对齐前缀实验的 SGLang 版本

与 exp_partial_prefix.py (vLLM 版) 用【完全相同】的 BASE 和截断逻辑,
跑在 SGLang (port 30000) 上, 对照两框架对"非块边界共享前缀"的命中差异。

核心对照:
  - vLLM (block_size=16, APC 块哈希): B 共享截断前缀只命中到最后一个对齐块
  - SGLang (page_size=1, RadixCache): 可在任意 token 位置 split, 命中全部共享前缀

用法: NO_PROXY="*" python3 exp_partial_prefix_sglang.py
"""

import json
import time

import httpx

BASE_URL = "http://127.0.0.1:30000"
MODEL = "Qwen/Qwen2.5-1.5B"

# 与 vLLM 版完全一致的前缀
BASE = "你是一位资深工业质检专家，精通缺陷检测、统计过程控制、六西格玛方法论。" * 10


def tok_count(client, text):
    try:
        r = client.post(f"{BASE_URL}/tokenize",
                        json={"model": MODEL, "prompt": text}, timeout=10)
        j = r.json()
        return j.get("count") or len(j.get("tokens", []))
    except Exception:
        return "?"


def completion_ttft(client, prompt):
    """用 /v1/completions (非 chat) 测 TTFT, 保持与 vLLM 版一致"""
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
    return first


def run(client, label, prompt):
    t = completion_ttft(client, prompt)
    n = tok_count(client, prompt)
    print(f"  {label:22s} TTFT={t*1000:7.1f}ms | prompt≈{n} tok")
    return t


def main():
    client = httpx.Client(trust_env=False)

    print("=" * 70)
    print("  SGLang 半块对齐前缀实验 (page_size=1, RadixCache)")
    print("=" * 70)

    pA = BASE + "请分析轴承故障。"
    pB = BASE[: len(BASE) // 2 + 3] + "请分析齿轮磨损。"

    print(f"\n  BASE = {len(BASE)} 字符 ≈ {tok_count(client, BASE)} token")
    print(f"  B 截断点 = len(BASE)//2 + 3 = {len(BASE)//2 + 3} 字符 (与 vLLM 版完全一致)\n")

    run(client, "A 冷启动", pA)
    run(client, "A 重复(应命中)", pA)
    run(client, "B 共享截断前缀", pB)
    run(client, "B 重复(应命中)", pB)

    print("\n  对照看 server 日志的 #cached-token 确认 SGLang 命中了多少:")
    print("  grep 'Prefill batch' /tmp/sglang_radix_exp.log | tail -4")


if __name__ == "__main__":
    main()
