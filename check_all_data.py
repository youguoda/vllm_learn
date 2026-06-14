from pathlib import Path
import csv
REQ = ["vllm_fixed_output.csv","sglang_fixed_output.csv","vllm_latency.csv","sglang_latency.csv",
       "vllm_multi_turn.csv","sglang_multi_turn.csv","vllm_streaming.csv","sglang_streaming.csv"]
OUT = Path("bench_results")
print("=== Week 6 数据完整性检查 ===")
ok = True
for f in REQ:
    p = OUT / f
    if p.exists():
        n = len(list(csv.DictReader(open(p))))
        print(f"  OK {f}: {n} 行")
    else:
        print(f"  缺失 {f}"); ok = False
print("\n全部就绪" if ok else "\n有缺失")
