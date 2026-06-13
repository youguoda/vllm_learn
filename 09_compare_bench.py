"""
vLLM vs SGLang 对比压测（OpenAI 接口通用，框架无关）

只依赖 /v1/chat/completions 的 streaming 接口，vLLM 和 SGLang 都兼容。
同一脚本指向不同端口即可对比。

用法:
  NO_PROXY="*" python3 09_compare_bench.py --base-url http://127.0.0.1:8000  --tag vllm
  NO_PROXY="*" python3 09_compare_bench.py --base-url http://127.0.0.1:30000 --tag sglang

输出 bench_<tag>.csv，供 plot_compare.py 画图。

注意: 本负载【无共享前缀】(40 个不同主题), radix 树没有发挥空间,
      两者接近是正常的——这是基线流程验证, 不是最终结论。
"""

import argparse
import asyncio
import csv
import json
import time

import httpx

# 40 个【不同】主题，故意无共享前缀，避免缓存影响基线对比
PROMPTS = [
    f"请用约100字介绍主题{i}：工业传感器的一种典型应用（场景{i*7 % 13}）。"
    for i in range(40)
]


async def one(client, base, model, prompt):
    """单条请求，stream 模式测 TTFT / e2e / ITL / token 数"""
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": True,
    }
    async with client.stream("POST", f"{base}/v1/chat/completions", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                c = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = c.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok += 1
    e2e = time.perf_counter() - t0
    if ttft is None:
        ttft = e2e
    itl = (e2e - ttft) / max(ntok - 1, 1)
    return ttft, e2e, itl, ntok


async def bench(base, model, conc):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        async def guarded(p):
            async with sem:
                return await one(client, base, model, p)
        t0 = time.perf_counter()
        rs = await asyncio.gather(*[guarded(p) for p in PROMPTS])
    wall = time.perf_counter() - t0
    n = len(rs)
    ttfts = sorted(r[0] for r in rs)
    return {
        "concurrency": conc,
        "rps": round(n / wall, 3),
        "ttft_avg": round(sum(r[0] for r in rs) / n, 4),
        "ttft_p95": round(ttfts[int(n * 0.95)], 4),
        "itl_avg": round(sum(r[2] for r in rs) / n, 5),
        "tok_per_s": round(sum(r[3] for r in rs) / wall, 1),
        "wall": round(wall, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="如 http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--tag", required=True, help="vllm 或 sglang")
    a = ap.parse_args()

    base = a.base_url.rstrip("/").removesuffix("/v1")

    print(f"=== 压测 [{a.tag}] @ {base} ===")
    print("预热中...")
    asyncio.run(bench(base, a.model, 2))  # 预热，不计

    rows = []
    for c in (1, 5, 10, 20):
        r = asyncio.run(bench(base, a.model, c))
        r["framework"] = a.tag
        print(f"  并发{c:>2}: RPS={r['rps']:>6.2f}  吞吐={r['tok_per_s']:>7.1f}tok/s  "
              f"TTFT={r['ttft_avg']*1000:>6.1f}ms  ITL={r['itl_avg']*1000:>5.2f}ms")
        rows.append(r)

    fields = ["framework", "concurrency", "rps", "tok_per_s",
              "ttft_avg", "ttft_p95", "itl_avg", "wall"]
    out = f"bench_{a.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"  → 已保存 {out}")


if __name__ == "__main__":
    main()
