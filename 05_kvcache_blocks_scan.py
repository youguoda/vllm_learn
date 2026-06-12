"""
PagedAttention 显存→块→并发 链路验证脚本

依次用不同的 (max_model_len, gpu_memory_utilization) 启动 vLLM server,
抓取每种配置下的:
  - num_gpu_blocks      (来自 /metrics 的 vllm:cache_config_info)
  - block_size          (同上)
  - GPU KV cache tokens (来自启动日志)
  - Maximum concurrency (来自启动日志)

验证两条公式:
  total_tokens = num_gpu_blocks * block_size
  max_concurrency = total_tokens / max_model_len

用法:
  HF_HUB_OFFLINE=1 NO_PROXY="*" python3 05_kvcache_blocks_scan.py
"""

import os
import re
import signal
import subprocess
import time

import httpx

MODEL = "Qwen/Qwen2.5-1.5B"
BASE_URL = "http://127.0.0.1:8000"

# (max_model_len, gpu_memory_utilization)
CONFIGS = [
    (2048, 0.85),
    (4096, 0.85),
    (8192, 0.85),
    (4096, 0.70),
]


def kill_server():
    """杀掉占用 8000 端口的 server 和残留 EngineCore"""
    subprocess.run("fuser -k 8000/tcp", shell=True,
                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    time.sleep(2)
    subprocess.run(
        "ps aux | grep EngineCore | grep -v grep | awk '{print $2}' | xargs -r kill -9",
        shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
    )
    time.sleep(3)


def wait_ready(timeout=240):
    start = time.time()
    client = httpx.Client(trust_env=False)  # 忽略 http_proxy 环境变量
    while time.time() - start < timeout:
        try:
            r = client.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def get_metrics():
    """从 /metrics 抓 num_gpu_blocks 和 block_size"""
    client = httpx.Client(trust_env=False)
    text = client.get(f"{BASE_URL}/metrics", timeout=5).text
    info_line = next(
        (l for l in text.splitlines() if l.startswith("vllm:cache_config_info")),
        "",
    )
    num_blocks = re.search(r'num_gpu_blocks="(\d+)"', info_line)
    block_size = re.search(r'block_size="(\d+)"', info_line)
    return (
        int(num_blocks.group(1)) if num_blocks else None,
        int(block_size.group(1)) if block_size else None,
    )


def parse_log(log_path):
    """从启动日志抓 KV cache tokens 和 Maximum concurrency"""
    with open(log_path) as f:
        text = f.read()
    tokens = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    conc = re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", text)
    return (
        int(tokens.group(1).replace(",", "")) if tokens else None,
        float(conc.group(1)) if conc else None,
    )


def run_one(max_len, gpu_util):
    kill_server()
    log_path = f"/tmp/vllm_scan_{max_len}_{gpu_util}.log"
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            ["vllm", "serve", MODEL,
             "--max-model-len", str(max_len),
             "--gpu-memory-utilization", str(gpu_util)],
            stdout=logf, stderr=subprocess.STDOUT, env=env,
            preexec_fn=os.setsid,
        )

    try:
        if not wait_ready():
            print(f"  [!] config ({max_len}, {gpu_util}) failed to start, check {log_path}")
            return None
        num_blocks, block_size = get_metrics()
        kv_tokens, max_conc = parse_log(log_path)
        return {
            "max_len": max_len, "gpu_util": gpu_util,
            "num_blocks": num_blocks, "block_size": block_size,
            "kv_tokens": kv_tokens, "max_conc": max_conc,
        }
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    results = []
    for max_len, gpu_util in CONFIGS:
        print(f">>> Testing max_model_len={max_len}, gpu_util={gpu_util} ...")
        r = run_one(max_len, gpu_util)
        if r:
            results.append(r)
            print(f"    num_gpu_blocks={r['num_blocks']}, "
                  f"block_size={r['block_size']}, "
                  f"kv_tokens={r['kv_tokens']}, max_conc={r['max_conc']}x")
        kill_server()
        time.sleep(2)

    # 汇总表
    print("\n" + "=" * 90)
    print("  汇总：显存预算 → GPU blocks → 并发容量")
    print("=" * 90)
    header = (f"  {'max_len':>8} | {'gpu_util':>8} | {'blocks':>7} | {'blk_sz':>6} | "
              f"{'KV tokens':>10} | {'tokens=blk*sz':>13} | {'max_conc':>8} | {'tok/len':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        calc_tokens = r["num_blocks"] * r["block_size"]
        calc_conc = calc_tokens / r["max_len"]
        print(f"  {r['max_len']:>8} | {r['gpu_util']:>8} | {r['num_blocks']:>7} | "
              f"{r['block_size']:>6} | {r['kv_tokens']:>10,} | {calc_tokens:>13,} | "
              f"{r['max_conc']:>7.2f}x | {calc_conc:>7.2f}x")

    print("\n  观察要点:")
    print("  1. 固定 gpu_util, 改 max_len: blocks 数几乎不变(块由剩余显存决定),")
    print("     变的是每个请求要占多少块 -> 所以并发数随 max_len 反比变化")
    print("  2. 固定 max_len, 调低 gpu_util: blocks 数下降(显存预算变小)")
    print("  3. KV tokens(日志) == blocks * block_size(metrics) 恒成立")


if __name__ == "__main__":
    main()
