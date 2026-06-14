"""baseline 数据合理性核查"""
import csv
from pathlib import Path


def load(p):
    with open(p) as f:
        return list(csv.DictReader(f))


print("=== Fixed Output 核查 ===")
for fw in ("vllm", "sglang"):
    p = Path(f"bench_results/{fw}_fixed_output.csv")
    if not p.exists():
        print(f"  {fw}: 缺失"); continue
    print(f"  {fw}:")
    for r in load(p):
        rps, ttft = float(r["rps"]), float(r["ttft_median_ms"])
        ok = (0.1 < rps < 100) and (10 < ttft < 5000)
        print(f"    并发{r['concurrency']:>2}: RPS={rps:>6.2f} tok/s={r['tok_per_s']:>7} "
              f"TTFT={ttft:>6.0f}ms  {'OK' if ok else '异常'}")

print("\n=== Latency 核查（TTFT 应随输入增长，但小文本+prefix cache 可能平坦）===")
for fw in ("vllm", "sglang"):
    p = Path(f"bench_results/{fw}_latency.csv")
    if not p.exists():
        print(f"  {fw}: 缺失"); continue
    print(f"  {fw}:")
    for r in load(p):
        print(f"    input≈{r['input_len']:>5}: TTFT={float(r['ttft_median_ms']):>6.0f}ms "
              f"ITL={r['itl_median_ms']}ms")
