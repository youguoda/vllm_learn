"""
Week 6 统一压测框架（框架无关，httpx 直连）
用法:
  NO_PROXY="*" python3 13_bench_runner.py --framework vllm   --scenario fixed_output
  NO_PROXY="*" python3 13_bench_runner.py --framework sglang --scenario latency
场景: fixed_output / latency / multi_turn / streaming
"""
import argparse
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

import httpx

BASE = {"vllm": "http://127.0.0.1:8000", "sglang": "http://127.0.0.1:30000"}
MODEL = "Qwen/Qwen2.5-1.5B"
OUT = Path("bench_results"); OUT.mkdir(exist_ok=True)


async def timed(client, base, messages, max_tokens, stream=True):
    t0 = time.perf_counter()
    ttft = None; ntok = 0
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": stream, "ignore_eos": True}
    if stream:
        async with client.stream("POST", f"{base}/v1/chat/completions", json=body) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                p = line[6:].strip()
                if p == "[DONE]":
                    break
                try:
                    c = json.loads(p)
                except json.JSONDecodeError:
                    continue
                d = c.get("choices", [{}])[0].get("delta", {})
                if d.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    ntok += 1
    else:
        r = await client.post(f"{base}/v1/chat/completions", json=body)
        ttft = time.perf_counter() - t0
        ntok = r.json().get("usage", {}).get("completion_tokens", max_tokens)
    e2e = time.perf_counter() - t0
    if ttft is None:
        ttft = e2e
    itl = (e2e - ttft) / max(ntok - 1, 1)
    return {"ttft": ttft, "e2e": e2e, "itl": itl, "tokens": ntok}


def filler(n_char):
    return "工业传感器振动频谱故障诊断质检" * (n_char // 14 + 1)


async def bench_fixed(client, base, conc, n_req, in_len, out_len):
    sem = asyncio.Semaphore(conc)
    msg = [{"role": "user", "content": "分析数据并给建议：" + filler(in_len)}]
    async def one():
        async with sem:
            return await timed(client, base, msg, out_len, stream=True)
    t0 = time.perf_counter()
    rs = await asyncio.gather(*[one() for _ in range(n_req)])
    wall = time.perf_counter() - t0
    ttfts = sorted(r["ttft"] for r in rs)
    return {"concurrency": conc,
            "rps": round(n_req / wall, 2),
            "tok_per_s": round(sum(r["tokens"] for r in rs) / wall, 1),
            "ttft_median_ms": round(statistics.median(r["ttft"] for r in rs) * 1000, 1),
            "ttft_p95_ms": round(ttfts[int(n_req * 0.95)] * 1000, 1),
            "itl_median_ms": round(statistics.median(r["itl"] for r in rs) * 1000, 2),
            "e2e_median_ms": round(statistics.median(r["e2e"] for r in rs) * 1000, 1)}


async def bench_latency(client, base, in_len, out_len, n_req):
    msg = [{"role": "user", "content": "总结：" + filler(in_len)}]
    rs = [await timed(client, base, msg, out_len, stream=True) for _ in range(n_req)]
    return {"input_len": in_len,
            "ttft_median_ms": round(statistics.median(r["ttft"] for r in rs) * 1000, 1),
            "itl_median_ms": round(statistics.median(r["itl"] for r in rs) * 1000, 2),
            "e2e_median_ms": round(statistics.median(r["e2e"] for r in rs) * 1000, 1)}


def save(fw, scenario, rows):
    out = OUT / f"{fw}_{scenario}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["framework"] + list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({"framework": fw, **r})
    print(f"  → 保存 {out}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--scenario", default="fixed_output")
    a = ap.parse_args()
    base = BASE[a.framework]

    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        print(f"=== [{a.framework}] {a.scenario} 预热 ===")
        await bench_fixed(client, base, 2, 5, 256, 64)

        rows = []
        if a.scenario == "fixed_output":
            for conc in (1, 4, 8, 16, 32):
                r = await bench_fixed(client, base, conc, 40, 256, 128)
                print(f"  并发{conc:>2}: RPS={r['rps']:>6} tok/s={r['tok_per_s']:>7} "
                      f"TTFT={r['ttft_median_ms']:>6}ms")
                rows.append(r)
        elif a.scenario == "latency":
            for il in (64, 256, 512, 1024, 2048):
                r = await bench_latency(client, base, il, 64, 10)
                print(f"  输入{il:>5}: TTFT={r['ttft_median_ms']:>7}ms "
                      f"ITL={r['itl_median_ms']:>5}ms")
                rows.append(r)
        save(a.framework, a.scenario, rows)


if __name__ == "__main__":
    asyncio.run(main())
