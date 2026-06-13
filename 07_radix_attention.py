"""
RadixAttention 多轮对话实验 (SGLang)

场景：同一 system prompt，多轮追问（每轮携带全部历史）→ 前缀逐轮变长。
观察：每轮 TTFT。前缀（system prompt + 历史对话）已在 radix 树上，
      轮 2 起只需 prefill 新增的"上一轮回答 + 新问题"，TTFT 应大降。

对照三档 system prompt 长度 (500/1000/2000 token)，验证"前缀越长，命中省得越多"。

启动 SGLang:
  source ~/venv-sglang/bin/activate
  NO_PROXY="*" HF_HUB_OFFLINE=1 python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B --context-length 4096 \
    --mem-fraction-static 0.8 --port 30000

跑实验:
  NO_PROXY="*" python3 07_radix_attention.py
"""

import time

import httpx

BASE_URL = "http://127.0.0.1:30000"
MODEL = "Qwen/Qwen2.5-1.5B"

QUESTIONS = [
    "轴承故障有哪些类型？",
    "如何用振动信号检测？",
    "采样频率怎么选？",
    "频谱分析关注哪些特征？",
    "如何区分内圈和外圈故障？",
]


def make_sys(n_tokens_approx: int) -> str:
    base = "你是一位资深工业质检专家，精通缺陷检测、统计过程控制与六西格玛方法论。"
    return base * max(1, n_tokens_approx // 30)


def tok_count(client, text):
    """用 SGLang 的 tokenize 估算 token 数（无该端点则用字符数粗估）"""
    try:
        r = client.post(f"{BASE_URL}/tokenize",
                        json={"model": MODEL, "prompt": text}, timeout=10)
        j = r.json()
        return j.get("count") or len(j.get("tokens", []))
    except Exception:
        return None


def chat_ttft(client, messages):
    """stream 模式发一轮对话，返回 (TTFT 秒, 完整回答文本)"""
    t0 = time.perf_counter()
    first = None
    chunks = []
    with client.stream(
        "POST", f"{BASE_URL}/v1/chat/completions",
        json={"model": MODEL, "messages": messages,
              "max_tokens": 100, "temperature": 0.0, "stream": True},
        timeout=120,
    ) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            import json
            try:
                c = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = c.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if first is None:
                    first = time.perf_counter() - t0
                chunks.append(delta["content"])
    return first, "".join(chunks)


def run_dialog(client, sys_prompt, tag):
    msgs = [{"role": "system", "content": sys_prompt}]
    print(f"\n=== {tag} ===")
    sys_tok = tok_count(client, sys_prompt)
    print(f"  system prompt 实测 ≈ {sys_tok} token, {len(sys_prompt)} 字")
    print(f"  {'轮':>3} | {'TTFT':>9} | {'历史长度(字)':>12}")
    print(f"  {'-'*3}-+-{'-'*9}-+-{'-'*12}")

    ttfts = []
    for i, q in enumerate(QUESTIONS):
        msgs.append({"role": "user", "content": q})
        first, answer = chat_ttft(client, msgs)
        ttfts.append(first)
        hist_len = sum(len(m["content"]) for m in msgs)
        print(f"  {i+1:>3} | {first*1000:>7.1f}ms | {hist_len:>12}")
        msgs.append({"role": "assistant", "content": answer})

    return sys_tok, ttfts


def main():
    client = httpx.Client(trust_env=False)  # 忽略 http_proxy

    print("=" * 70)
    print("  RadixAttention 多轮对话实验 (SGLang, port 30000)")
    print("=" * 70)

    summary = []
    for n in (500, 1000, 2000):
        sys_tok, ttfts = run_dialog(client, make_sys(n), f"system prompt≈{n} token")
        t1 = ttfts[0]
        warm = ttfts[1:]
        warm_mean = sum(warm) / len(warm)
        drop = (1 - warm_mean / t1) * 100 if t1 else 0
        summary.append({
            "target": n, "sys_tok": sys_tok,
            "t1": t1, "warm_mean": warm_mean, "drop": drop,
        })

    # 汇总表
    print("\n" + "=" * 70)
    print("  三档前缀长度汇总")
    print("=" * 70)
    print(f"  {'前缀目标':>8} | {'实测token':>9} | {'轮1 TTFT':>9} | {'轮2-5均值':>9} | {'降幅':>7}")
    print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}")
    for s in summary:
        print(f"  {s['target']:>8} | {str(s['sys_tok']):>9} | "
              f"{s['t1']*1000:>7.1f}ms | {s['warm_mean']*1000:>7.1f}ms | {s['drop']:>5.1f}%")

    print("\n  预期：前缀越长，轮1越慢、降幅越大——命中省的就是前缀的 prefill。")


if __name__ == "__main__":
    main()
