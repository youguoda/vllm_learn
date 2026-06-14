"""合并所有 CSV → bench_results/all_data.json"""
import csv, json
from pathlib import Path

SCENARIOS = {
    "fixed_output": ["concurrency", "rps", "tok_per_s", "ttft_median_ms", "ttft_p95_ms", "itl_median_ms", "e2e_median_ms"],
    "latency": ["input_len", "ttft_median_ms", "itl_median_ms", "e2e_median_ms"],
    "multi_turn": ["concurrency", "turn", "ttft_median_ms"],
    "streaming": ["concurrency", "itl_p50_ms", "itl_p95_ms", "itl_p99_ms"],
}

all_data = {}
for sc, fields in SCENARIOS.items():
    all_data[sc] = {}
    for fw in ("vllm", "sglang"):
        p = Path(f"bench_results/{fw}_{sc}.csv")
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p)))
        cleaned = []
        for r in rows:
            cr = {"framework": fw}
            for k in fields:
                if k in r:
                    try:
                        cr[k] = float(r[k])
                    except ValueError:
                        cr[k] = r[k]
            cleaned.append(cr)
        all_data[sc][fw] = cleaned
        print(f"  {fw} {sc}: {len(cleaned)} 行")

json.dump(all_data, open("bench_results/all_data.json", "w"), ensure_ascii=False, indent=2)
print("all_data.json 已保存")
