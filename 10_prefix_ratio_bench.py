"""
前缀复用率实验（全项目最重要数据）
变量: prefix_ratio = 共享前缀占总输入的比例
固定: 总输入 ~512 token, 输出 64 token, 20 请求, 并发 8

框架无关(OpenAI 接口), 同脚本指向不同端口对比 vLLM / SGLang。

用法:
  NO_PROXY="*" python3 10_prefix_ratio_bench.py --base-url http://127.0.0.1:8000  --tag vllm
  NO_PROXY="*" python3 10_prefix_ratio_bench.py --base-url http://127.0.0.1:30000 --tag sglang
"""
import argparse
import asyncio
import csv
import json
import time

import httpx

# 固定的共享 system prompt 素材(约 250 token)
BASE_SYS = ("你是工业质检专家，精通以下知识：振动分析、频谱特征、故障模式识别、"
            "统计过程控制、六西格玛方法论、预测性维护策略。") * 5

UNIQUE_QUESTIONS = [
    f"请分析场景{i}：传感器编号S{i:03d}的振动超标，频率为{300 + i * 7}Hz，给出诊断建议。"
    for i in range(20)
]


def build_messages(prefix_ratio: float, question: str):
    """
    prefix_ratio=1.0: 所有请求同一 system prompt (100% 可复用)
    prefix_ratio=0.0: system prompt 由问题内容填充 (0% 共享)
    """
    shared_len = int(len(BASE_SYS) * prefix_ratio)
    shared = BASE_SYS[:shared_len]
    # 剩余部分用问题内容填(每个请求不同 → 不可共享)
    fill = (question * 10)[: len(BASE_SYS) - shared_len]
    system = shared + fill
    return [{"role": "system", "content": system},
            {"role": "user", "content": question}]


async def one(client, base, model, msgs):
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    body = {"model": model, "messages": msgs, "max_tokens": 64,
            "temperature": 0.0, "stream": True}
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
    e2e = time.perf_counter() - t0
    return (ttft if ttft is not None else e2e), e2e, ntok


async def run_ratio(base, model, ratio, n_concur=8):
    sem = asyncio.Semaphore(n_concur)
    all_msgs = [build_messages(ratio, q) for q in UNIQUE_QUESTIONS]
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        async def guarded(msgs):
            async with sem:
                return await one(client, base, model, msgs)
        t0 = time.perf_counter()
        rs = await asyncio.gather(*[guarded(m) for m in all_msgs])
    wall = time.perf_counter() - t0
    n = len(rs)
    ttfts = sorted(r[0] for r in rs)
    return {
        "prefix_ratio": ratio,
        "ttft_avg_ms": round(sum(r[0] for r in rs) / n * 1000, 1),
        "ttft_p95_ms": round(ttfts[int(n * 0.95)] * 1000, 1),
        "rps": round(n / wall, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    base = a.base_url.rstrip("/").removesuffix("/v1")

    print(f"=== 前缀复用率实验 [{a.tag}] @ {base} ===")
    print("预热...")
    asyncio.run(run_ratio(base, a.model, 1.0, n_concur=4))

    rows = []
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = asyncio.run(run_ratio(base, a.model, ratio))
        r["framework"] = a.tag
        print(f"  ratio={ratio:.2f}  TTFT_avg={r['ttft_avg_ms']:>6.1f}ms  "
              f"TTFT_p95={r['ttft_p95_ms']:>6.1f}ms  RPS={r['rps']:.2f}")
        rows.append(r)

    out = f"prefix_ratio_{a.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["framework", "prefix_ratio",
                                          "ttft_avg_ms", "ttft_p95_ms", "rps"])
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in w.fieldnames})
    print(f"  → 保存 {out}")


if __name__ == "__main__":
    main()
