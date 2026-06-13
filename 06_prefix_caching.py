"""
vLLM Automatic Prefix Caching (APC) 实验

原理:
  APC 把"已经算过的前缀 KV cache"按 block 缓存起来 (block hash 链式包含前缀)。
  下一个请求如果开头前缀相同, 直接复用这些 block, 跳过这部分 prefill。
  -> 第 2 条起 TTFT 骤降 (省掉长 system prompt 的 prefill 时间)
  -> 但只省 prefill(TTFT), 不省 decode(生成速度不变)

类比: 食堂预制菜——前半段菜(system prompt KV)提前做好, 谁来都直接端走,
      只现炒后半段(用户问题); 但上菜后吃饭速度(decode)不变。

用法:
  # 开 APC (默认)
  HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B --max-model-len 4096 --gpu-memory-utilization 0.85
  NO_PROXY="*" python3 06_prefix_caching.py --label "APC ON"

  # 关 APC 对照
  HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve ... --no-enable-prefix-caching
  NO_PROXY="*" python3 06_prefix_caching.py --label "APC OFF"

  # 变体: 前缀加倍 / 前缀加随机数字(破坏命中)
  NO_PROXY="*" python3 06_prefix_caching.py --label "2x prefix" --prefix-repeat 130
  NO_PROXY="*" python3 06_prefix_caching.py --label "random prefix" --random-prefix
"""

import argparse
import json
import random
import re
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen2.5-1.5B"

# 一段会被重复拼接的"素材", 用来堆出长 system prompt
PREFIX_UNIT = (
    "You are a knowledgeable and helpful AI assistant. "
    "You always answer concisely and accurately, citing facts when relevant. "
    "You are patient, thorough, and never make things up. "
)

# 10 个不同的短问题 (前缀相同, 后缀不同)
QUESTIONS = [
    "What is the capital of France?",
    "How many legs does a spider have?",
    "What is the boiling point of water?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "How many continents are there?",
    "What is the largest planet?",
    "What year did World War II end?",
    "What is the chemical symbol for gold?",
    "How many sides does a hexagon have?",
]


def build_system_prompt(repeat: int) -> str:
    """重复拼接出约 repeat*len 的长前缀。repeat=65 约 1000 token。"""
    return PREFIX_UNIT * repeat


def get_prefix_metrics(client: httpx.Client):
    """抓 prefix cache 累计 queries / hits (单位: block 数)"""
    text = client.get(f"{BASE_URL}/metrics", timeout=5).text
    def grab(name):
        m = re.search(rf'{name}\{{[^}}]*\}}\s+([\d.eE+]+)', text)
        return float(m.group(1)) if m else 0.0
    q = grab("vllm:prefix_cache_queries_total")
    h = grab("vllm:prefix_cache_hits_total")
    return q, h


def send_request(client: httpx.Client, system_prompt: str, question: str) -> dict:
    """stream 模式发请求, 测 TTFT 和总时长"""
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    with client.stream(
        "POST",
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "max_tokens": 32,
            "temperature": 0.0,  # 贪心, 让结果可复现
            "stream": True,
        },
        timeout=60,
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_chunks += 1
    total = time.perf_counter() - t0
    return {"ttft": ttft, "total": total, "decode_time": total - (ttft or 0), "chunks": n_chunks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="本次运行标签")
    parser.add_argument("--prefix-repeat", type=int, default=65,
                        help="PREFIX_UNIT 重复次数 (65≈1000token, 130≈2000token)")
    parser.add_argument("--random-prefix", action="store_true",
                        help="每条请求前缀开头加随机数字(破坏命中)")
    parser.add_argument("--salt", default="",
                        help="前缀开头加固定盐, 让不同长度实验的前缀互不重叠")
    args = parser.parse_args()

    client = httpx.Client(trust_env=False)  # 忽略 http_proxy

    base_prefix = build_system_prompt(args.prefix_repeat)
    if args.salt:
        base_prefix = f"[{args.salt}] " + base_prefix

    # 估算前缀 token 数 (用 /tokenize 端点)
    try:
        tok = client.post(f"{BASE_URL}/tokenize",
                          json={"model": MODEL, "prompt": base_prefix}, timeout=10)
        n_prefix_tokens = tok.json().get("count", "?")
    except Exception:
        n_prefix_tokens = "?"

    print(f"\n{'='*70}")
    print(f"  [{args.label}]  prefix≈{n_prefix_tokens} tokens, "
          f"repeat={args.prefix_repeat}, random_prefix={args.random_prefix}")
    print(f"{'='*70}")

    q0, h0 = get_prefix_metrics(client)

    print(f"  {'#':>2} | {'TTFT':>8} | {'total':>8} | {'decode':>8} | question")
    print(f"  {'-'*2}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*30}")

    results = []
    for i, q in enumerate(QUESTIONS):
        system_prompt = base_prefix
        if args.random_prefix:
            # 开头加随机数字 -> 第一个 block 的 hash 就变 -> 整条链命中失效
            system_prompt = f"[{random.randint(10**8, 10**9)}] " + base_prefix
        r = send_request(client, system_prompt, q)
        if r["ttft"] is None:
            # 请求被拒(通常 prompt 超 max_model_len)或没返回内容
            print(f"  {i:>2} | {'  --  ':>8} | {'  --  ':>8} | {'  --  ':>8} | "
                  f"{q[:30]}  [NO OUTPUT - prompt 可能超 max_model_len]")
            continue
        results.append(r)
        tag = "COLD" if len(results) == 1 else ""
        print(f"  {i:>2} | {r['ttft']*1000:>6.1f}ms | {r['total']*1000:>6.1f}ms | "
              f"{r['decode_time']*1000:>6.1f}ms | {q[:30]}  {tag}")

    if not results:
        print("\n  [!] 所有请求都无输出，检查 prefix 是否超过 max_model_len")
        return

    q1, h1 = get_prefix_metrics(client)

    # 统计
    ttfts = [r["ttft"] for r in results]
    cold_ttft = ttfts[0]
    warm_ttfts = ttfts[1:]
    warm_mean = sum(warm_ttfts) / len(warm_ttfts)
    decode_mean = sum(r["decode_time"] for r in results) / len(results)

    dq = q1 - q0
    dh = h1 - h0
    hit_rate = (dh / dq * 100) if dq > 0 else 0.0

    print(f"\n  --- 统计 [{args.label}] ---")
    print(f"  第 0 条 (COLD)  TTFT : {cold_ttft*1000:.1f} ms")
    print(f"  第 1+ 条均值    TTFT : {warm_mean*1000:.1f} ms")
    print(f"  warm/cold 比值       : {warm_mean/cold_ttft:.2f}  (越小=命中省得越多)")
    print(f"  平均 decode 时间     : {decode_mean*1000:.1f} ms  (APC 不影响这个)")
    print(f"  prefix cache 本轮    : queries +{dq:.0f} blocks, hits +{dh:.0f} blocks")
    print(f"  本轮命中率           : {hit_rate:.1f}%")


if __name__ == "__main__":
    main()
