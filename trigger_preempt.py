"""
trigger_preempt.py — 故意逼出 vLLM 抢占(preemption)

同时提交多个长输出请求争夺有限 KV 块。配合小 max-model-len + 低显存启动:
  HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B \
    --max-model-len 2048 --gpu-memory-utilization 0.4 \
    --max-num-seqs 64 --port 8000

观察 server 日志的 "Preempted" 字样。
"""
import asyncio
import json
import time

import httpx

BASE = "http://127.0.0.1:8000"
LONG_PROMPT = ("请写一篇关于工业传感器发展历史的详细报告，要求包含："
               "起源、主要技术演进、代表性产品、未来趋势，每个章节不少于500字。")


async def req(client, i):
    t0 = time.perf_counter()
    body = {"model": "Qwen/Qwen2.5-1.5B",
            "messages": [{"role": "user", "content": LONG_PROMPT}],
            "max_tokens": 512, "temperature": 0.7}
    r = await client.post(f"{BASE}/v1/chat/completions", json=body)
    n = len(r.json()["choices"][0]["message"]["content"])
    print(f"  请求{i:02d} 完成, 输出 {n} 字, 耗时 {time.perf_counter()-t0:.1f}s")


async def main():
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        print("同时发出 24 个长输出请求(争夺 KV 块)...")
        await asyncio.gather(*[req(client, i) for i in range(24)])


if __name__ == "__main__":
    asyncio.run(main())
