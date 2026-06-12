"""
vLLM Continuous Batching 实验

使用方法:
  1. 启动服务器:
     NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B \
         --max-model-len 4096 --gpu-memory-utilization 0.85 \
         [--max-num-seqs 16] [--max-num-batched-tokens 2048]

  2. 运行测试:
     NO_PROXY="*" python3 04_continuous_batching.py [--concurrency 1 5 10 20] [--num-requests 20] [--prompt-len 50]

原理:
  Continuous Batching (又称 iteration-level batching):
  - 每个推理 step, scheduler 从 waiting queue 取请求加入 batch
  - 已完成的序列立即从 batch 移除, 腾出位置给新请求
  - 对比 Static Batching: 等 batch 中所有序列完成后才处理下一批

  核心优势: 短请求不用等长请求完成, 显存利用率更高, 吞吐量大幅提升
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx

BASE_URL = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-1.5B"


# ========== 预设的测试 prompt (不同长度) ==========

PROMPTS_SHORT = [
    "What is 2+2?",
    "Name a color.",
    "Is the sky blue?",
    "Say hello.",
    "What day is it?",
    "Count to 5.",
    "Name a fruit.",
    "What is water?",
    "Say yes or no.",
    "Name an animal.",
]

PROMPTS_MEDIUM = [
    "Explain the concept of recursion in programming with a simple example.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis in simple terms.",
    "What is the difference between a process and a thread?",
    "Explain what an API is and give an example of how it's used.",
    "What are the key principles of object-oriented programming?",
    "Describe how a hash table works and its time complexity.",
    "What is the difference between SQL and NoSQL databases?",
    "Explain the concept of machine learning in simple terms.",
    "What are microservices and what problems do they solve?",
]


async def send_request(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> dict:
    """发送单个请求，记录 TTFT 和总延迟"""
    t_start = time.perf_counter()

    async with client.stream(
        "POST",
        f"{BASE_URL}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,  # 用 stream 模式测量 TTFT
        },
        timeout=60,
    ) as response:
        ttft = None
        chunks = []
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t_start
            try:
                chunk = json.loads(data_str)
                if chunk.get("choices") and chunk["choices"][0].get("text"):
                    chunks.append(chunk["choices"][0]["text"])
            except json.JSONDecodeError:
                pass

    t_end = time.perf_counter()
    total_latency = t_end - t_start
    output_text = "".join(chunks)
    output_tokens = len(output_text.split())  # 粗略估计

    return {
        "request_id": request_id,
        "ttft": ttft,
        "total_latency": total_latency,
        "output_len": len(output_text),
        "prompt_len": len(prompt),
    }


async def run_benchmark(
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    prompts: list[str],
) -> list[dict]:
    """用指定并发数运行基准测试"""
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(client, prompt, req_id):
        async with semaphore:
            return await send_request(client, prompt, max_tokens, req_id)

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            tasks.append(limited_request(client, prompt, i))

        results = await asyncio.gather(*tasks)

    return list(results)


def print_results(label: str, results: list[dict], concurrency: int):
    """打印统计结果"""
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    latencies = [r["total_latency"] for r in results]
    total_output_chars = sum(r["output_len"] for r in results)

    # 总实验时间 = 最晚完成 - 最早开始
    wall_time = max(latencies)  # 并发请求的 wall time 近似于最慢的那个

    # 吞吐量 = 总输出字符数 / wall time (粗略, 更准确需要用 token 数)
    throughput_chars = total_output_chars / wall_time if wall_time > 0 else 0
    # 请求数 / wall time
    rps = len(results) / wall_time if wall_time > 0 else 0

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  concurrency={concurrency}, {len(results)} requests")
    print(f"{'='*65}")
    print(f"  {'Metric':<28} {'Value':>12}")
    print(f"  {'-'*28} {'-'*12}")
    print(f"  {'Wall time':.<28} {wall_time:>10.2f} s")
    print(f"  {'Requests/sec':.<28} {rps:>10.2f} /s")
    print(f"  {'Output chars/sec':.<28} {throughput_chars:>10.1f} /s")

    if ttfts:
        print(f"  {'TTFT (mean)':.<28} {statistics.mean(ttfts):>10.3f} s")
        print(f"  {'TTFT (p50)':.<28} {sorted(ttfts)[len(ttfts)//2]:>10.3f} s")
        print(f"  {'TTFT (p95)':.<28} {sorted(ttfts)[int(len(ttfts)*0.95)]:>10.3f} s")
        print(f"  {'TTFT (max)':.<28} {max(ttfts):>10.3f} s")

    print(f"  {'Latency (mean)':.<28} {statistics.mean(latencies):>10.3f} s")
    print(f"  {'Latency (p50)':.<28} {sorted(latencies)[len(latencies)//2]:>10.3f} s")
    print(f"  {'Latency (max)':.<28} {max(latencies):>10.3f} s")


def main():
    parser = argparse.ArgumentParser(description="vLLM Continuous Batching Benchmark")
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
        help="并发数列表 (可多个, 如 --concurrency 1 5 10 20)",
    )
    parser.add_argument("--num-requests", type=int, default=20, help="总请求数")
    parser.add_argument("--max-tokens", type=int, default=32, help="每个请求最大输出 token 数")
    parser.add_argument(
        "--prompt-type",
        choices=["short", "medium", "mixed"],
        default="mixed",
        help="Prompt 类型: short(~10 tokens), medium(~30 tokens), mixed(混合)",
    )
    args = parser.parse_args()

    # 选择 prompts
    if args.prompt_type == "short":
        prompts = PROMPTS_SHORT
    elif args.prompt_type == "medium":
        prompts = PROMPTS_MEDIUM
    else:  # mixed
        prompts = PROMPTS_SHORT + PROMPTS_MEDIUM

    print(f"vLLM Continuous Batching Benchmark")
    print(f"  Model:        {MODEL}")
    print(f"  Concurrency:  {args.concurrency}")
    print(f"  Requests:     {args.num_requests}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Prompt type:  {args.prompt_type}")

    all_summary = []
    for c in args.concurrency:
        results = asyncio.run(
            run_benchmark(c, args.num_requests, args.max_tokens, prompts)
        )
        print_results(f"Concurrency = {c}", results, c)

        ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
        latencies = [r["total_latency"] for r in results]
        wall_time = max(latencies)
        all_summary.append({
            "concurrency": c,
            "wall_time": round(wall_time, 2),
            "rps": round(len(results) / wall_time, 2),
            "ttft_mean": round(statistics.mean(ttfts), 3) if ttfts else 0,
            "ttft_p95": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 3) if ttfts else 0,
            "latency_mean": round(statistics.mean(latencies), 3),
            "latency_max": round(max(latencies), 3),
        })

    # 汇总表
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Conc':>4} | {'RPS':>7} | {'TTFT mean':>9} | {'TTFT p95':>9} | {'Lat mean':>9} | {'Lat max':>9} | {'Wall(s)':>8}")
    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")
    for s in all_summary:
        print(
            f"  {s['concurrency']:>4} | {s['rps']:>7.2f} | {s['ttft_mean']:>8.3f}s | "
            f"{s['ttft_p95']:>8.3f}s | {s['latency_mean']:>8.3f}s | {s['latency_max']:>8.3f}s | {s['wall_time']:>7.2f}s"
        )


if __name__ == "__main__":
    main()
