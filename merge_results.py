"""三轮 CSV 取中位数合并"""
import csv
import statistics
import sys
from pathlib import Path

OUT = Path("bench_results")


def merge(fw, scenario):
    rounds = []
    for r in (1, 2, 3):
        p = OUT / f"{fw}_{scenario}_r{r}.csv"
        if p.exists():
            with open(p) as f:
                rounds.append(list(csv.DictReader(f)))
    if not rounds:
        print("无数据"); return
    keys = list(rounds[0][0].keys())
    merged = []
    for i, row0 in enumerate(rounds[0]):
        m = {}
        for k in keys:
            try:
                vals = [float(rd[i][k]) for rd in rounds if i < len(rd)]
                m[k] = round(statistics.median(vals), 2)
            except (ValueError, KeyError):
                m[k] = row0.get(k, "")
        merged.append(m)
    out = OUT / f"{fw}_{scenario}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=merged[0].keys())
        w.writeheader(); w.writerows(merged)
    print(f"合并(3轮中位数) → {out}")


if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2])
