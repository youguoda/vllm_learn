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


def unique_filler(n_char):
    # 用随机中文+数字避免重复触发 prefix cache，让 TTFT 真实反映 prefill
    import random
    pool = "工业传感器振动频谱故障诊断质检温度压力流量0123456789的是在和有"
    return "".join(random.choices(pool, k=n_char))


async def bench_latency(client, base, in_len, out_len, n_req):
    import random
    rs = []
    for _ in range(n_req):
        # 每次唯一输入, 禁止 prefix cache 命中, TTFT 反映真实 prefill
        # 字符数≈token数(中文), 留余量避免超 max_model_len(4096)
        msg = [{"role": "user", "content": f"[{random.randint(10**6,10**7)}]分析：" + unique_filler(min(in_len, 1500))}]
        rs.append(await timed(client, base, msg, out_len, stream=True))
    return {"input_len": in_len,
            "ttft_median_ms": round(statistics.median(r["ttft"] for r in rs) * 1000, 1),
            "itl_median_ms": round(statistics.median(r["itl"] for r in rs) * 1000, 2),
            "e2e_median_ms": round(statistics.median(r["e2e"] for r in rs) * 1000, 1)}


SYS_PROMPT = ("你是工业质检专家，精通振动分析、频谱分析、缺陷模式识别、"
              "六西格玛和统计过程控制。") * 8  # ~500 token
TURN_Q = ["轴承外圈剥落的振动特征是什么？", "如何用频谱分析定位故障位置？",
          "采样频率应该如何选择？", "报警阈值设定有什么原则？",
          "故障发展到什么程度需要立即停机？"]


async def one_dialog(client, base, sem):
    async with sem:
        msgs = [{"role": "system", "content": SYS_PROMPT}]
        ttfts = []
        for q in TURN_Q:
            msgs.append({"role": "user", "content": q})
            r = await timed(client, base, msgs, 100, stream=True)
            ttfts.append(r["ttft"])
            msgs.append({"role": "assistant", "content": "(回答)"})
        return ttfts


async def bench_multi_turn(client, base, conc, n_dialogs=20):
    sem = asyncio.Semaphore(conc)
    t0 = time.perf_counter()
    all_t = await asyncio.gather(*[one_dialog(client, base, sem) for _ in range(n_dialogs)])
    wall = time.perf_counter() - t0
    rows = []
    for ti in range(len(TURN_Q)):
        rows.append({"concurrency": conc, "turn": ti + 1,
                     "ttft_median_ms": round(statistics.median(d[ti] for d in all_t) * 1000, 1)})
    rows.append({"concurrency": conc, "turn": "rps",
                 "ttft_median_ms": round(n_dialogs * len(TURN_Q) / wall, 2)})
    return rows


async def bench_streaming(client, base, conc, n_req=20):
    sem = asyncio.Semaphore(conc)
    msg = [{"role": "user", "content": "详细分析：" + filler(256)}]
    async def one():
        async with sem:
            return await timed(client, base, msg, 256, stream=True)
    rs = await asyncio.gather(*[one() for _ in range(n_req)])
    itls = sorted(r["itl"] for r in rs)
    return {"concurrency": conc,
            "itl_p50_ms": round(itls[int(n_req * 0.50)] * 1000, 2),
            "itl_p95_ms": round(itls[int(n_req * 0.95)] * 1000, 2),
            "itl_p99_ms": round(itls[min(int(n_req * 0.99), n_req - 1)] * 1000, 2)}


def save(fw, scenario, rows, suffix=""):
    out = OUT / f"{fw}_{scenario}{suffix}.csv"
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
    ap.add_argument("--suffix", default="")
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
        elif a.scenario == "multi_turn":
            for conc in (1, 8, 16):
                sub = await bench_multi_turn(client, base, conc)
                for r in sub:
                    print(f"  并发{conc:>2} 轮{r['turn']}: {r['ttft_median_ms']}")
                rows.extend(sub)
        elif a.scenario == "streaming":
            for conc in (1, 4, 8):
                r = await bench_streaming(client, base, conc)
                print(f"  并发{conc}: ITL_p50={r['itl_p50_ms']}ms p95={r['itl_p95_ms']}ms")
                rows.append(r)
        save(a.framework, a.scenario, rows, a.suffix)


if __name__ == "__main__":
    asyncio.run(main())
