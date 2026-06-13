"""
ngram 投机解码实验：普通 decode vs ngram 投机
指标：单请求 E2E 延迟、token/s。ngram 在重复性高内容(代码/模板)上有效。

运行(对应启动的 server):
  普通: vllm serve Qwen/Qwen2.5-1.5B --max-model-len 4096 --gpu-memory-utilization 0.85
  投机: 上面加 --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'

用法: NO_PROXY="*" python3 11_speculative_decode.py [tag]
"""
import json
import statistics
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen2.5-1.5B"
client = httpx.Client(trust_env=False, timeout=120)

with open("spec_prompts.json") as f:
    PROMPTS = json.load(f)


def bench(prompt, max_tokens=200, n_repeat=3):
    results = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        r = client.post(f"{BASE}/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True})
        dt = time.perf_counter() - t0
        usage = r.json().get("usage", {})
        n_tok = usage.get("completion_tokens", max_tokens)
        results.append((dt, n_tok))
    dt_med = statistics.median(d for d, _ in results)
    n_med = statistics.median(n for _, n in results)
    return dt_med, n_med, n_med / dt_med


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    print(f"=== 投机解码实验 [{tag}] ===")
    print(f"{'类型':<8} {'Prompt前16字':<20} {'E2E(s)':>8} {'tok/s':>8}")
    print("-" * 50)
    summary = {}
    for ptype, prompts in [("高重复", PROMPTS["high_rep"]), ("低重复", PROMPTS["low_rep"])]:
        tps_list = []
        for p in prompts:
            dt, n, tps = bench(p)
            tps_list.append(tps)
            print(f"{ptype:<8} {p[:16]:<20} {dt:>8.2f} {tps:>8.1f}")
        summary[ptype] = statistics.mean(tps_list)
    print(f"\n  高重复均值 tok/s: {summary['高重复']:.1f}")
    print(f"  低重复均值 tok/s: {summary['低重复']:.1f}")


if __name__ == "__main__":
    main()
