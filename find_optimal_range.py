"""找吞吐/延迟性价比最高的并发数"""
import json
D = json.load(open("bench_results/all_data.json"))
print("=== 最优负载区间分析 (效率 = tok/s ÷ TTFT秒) ===\n")
for fw in ("vllm", "sglang"):
    rows = D.get("fixed_output", {}).get(fw, [])
    print(f"--- {fw} ---")
    for r in rows:
        tps = r.get("tok_per_s", 0)
        ttft = r.get("ttft_median_ms", 1) / 1000
        eff = tps / ttft if ttft > 0 else 0
        print(f"  并发{int(r['concurrency']):>2}: 吞吐={tps:>7.1f} tok/s "
              f"TTFT={r.get('ttft_median_ms', 0):>6.0f}ms 效率={eff:>8.0f}")
    best = max(rows, key=lambda r: r.get("tok_per_s", 0) / max(r.get("ttft_median_ms", 1) / 1000, 0.01))
    print(f"  → 性价比最优并发: {int(best['concurrency'])}\n")
