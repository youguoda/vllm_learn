"""
sched_compare.py — 共享长前缀场景对比调度策略对 TTFT 的影响

用同一长 system prompt + 8 个不同问题。cache-aware 排序(LPM)应让共享前缀
的请求集中调度，命中率更高、TTFT 更低。

用法: NO_PROXY="*" python3 sched_compare.py [base_url]
"""
import asyncio
import sys
import time

import httpx

SYSTEM = ("你是一位专业的气象分析师。以下是过去30天的逐小时气象观测数据："
          + "温度22°C，湿度65%，气压1013hPa，风速3m/s；" * 40)

QUESTIONS = [
    "请分析最近一周的气温趋势并给出预测。",
    "湿度变化与降水概率的关系是什么？",
    "气压的波动规律说明了什么天气现象？",
    "请给出本月平均气象指标的统计摘要。",
    "风速数据是否有异常？",
    "综合以上数据，明天的天气预报是什么？",
    "气象数据中有无极端天气预警信号？",
    "请计算各指标的标准差。",
]


async def timed(client, base, i, q):
    t0 = time.perf_counter()
    first = None
    body = {"model": "Qwen/Qwen2.5-1.5B",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": q}],
            "max_tokens": 80, "temperature": 0.0, "stream": True}
    async with client.stream("POST", f"{base}/v1/chat/completions", json=body) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                if first is None:
                    first = time.perf_counter() - t0
    return first if first else (time.perf_counter() - t0)


async def run(base):
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        ttfts = await asyncio.gather(*[timed(client, base, i, q)
                                       for i, q in enumerate(QUESTIONS)])
    for i, t in enumerate(ttfts):
        print(f"  [req {i:02d}] TTFT={t*1000:.1f}ms")
    print(f"\n  平均 TTFT: {sum(ttfts)/len(ttfts)*1000:.1f}ms  最大: {max(ttfts)*1000:.1f}ms")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
    base = base.rstrip("/").removesuffix("/v1")
    asyncio.run(run(base))
