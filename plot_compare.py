"""
画 vLLM vs SGLang 对比图。
用基础环境 python: /home/guoda/python/bin/python3 plot_compare.py
(中文字体在 WSL 常缺失, 用英文标签避免乱码)
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(tag):
    with open(f"bench_{tag}.csv") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs("assets", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (a1, a2), (a3, a4) = axes

    styles = {"vllm": ("o-", "#1f77b4"), "sglang": ("s--", "#ff7f0e")}
    for tag, (style, color) in styles.items():
        rows = load(tag)
        x = [int(r["concurrency"]) for r in rows]
        a1.plot(x, [float(r["tok_per_s"]) for r in rows], style, color=color, label=tag, linewidth=2, markersize=7)
        a2.plot(x, [float(r["ttft_avg"]) * 1000 for r in rows], style, color=color, label=tag, linewidth=2, markersize=7)
        a3.plot(x, [float(r["rps"]) for r in rows], style, color=color, label=tag, linewidth=2, markersize=7)
        a4.plot(x, [float(r["itl_avg"]) * 1000 for r in rows], style, color=color, label=tag, linewidth=2, markersize=7)

    a1.set_title("Output Throughput (token/s)  - higher better", fontsize=12, fontweight="bold")
    a2.set_title("TTFT mean (ms)  - lower better", fontsize=12, fontweight="bold")
    a3.set_title("Request throughput RPS (req/s)  - higher better", fontsize=12, fontweight="bold")
    a4.set_title("ITL Inter-token latency (ms)  - lower better", fontsize=12, fontweight="bold")
    for a in (a1, a2, a3, a4):
        a.set_xlabel("Concurrency")
        a.set_xticks([1, 5, 10, 20])
        a.grid(True, alpha=0.3)
        a.legend()

    fig.suptitle("vLLM vs SGLang  |  Qwen2.5-1.5B, RTX 3060 12GB, ctx=4096, mem=0.85\n"
                 "(no shared prefix workload - radix tree has no room to shine)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "assets/compare_v1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
