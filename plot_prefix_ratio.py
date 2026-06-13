"""画前缀复用率对比图。用基础环境 python: /home/guoda/python/bin/python3 plot_prefix_ratio.py"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False


def load(tag):
    with open(f"prefix_ratio_{tag}.csv") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs("assets", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"vllm": ("#4C72B0", "o-"), "sglang": ("#55A868", "s--")}
    for tag, (c, sty) in colors.items():
        rows = load(tag)
        x = [float(r["prefix_ratio"]) for r in rows]
        ax1.plot(x, [float(r["ttft_avg_ms"]) for r in rows], sty, color=c, label=tag, ms=8, lw=2)
        ax2.plot(x, [float(r["rps"]) for r in rows], sty, color=c, label=tag, ms=8, lw=2)

    ax1.set_title("前缀复用率 vs 平均 TTFT (越低越好)")
    ax1.set_xlabel("前缀复用率"); ax1.set_ylabel("TTFT (ms)")
    ax2.set_title("前缀复用率 vs 吞吐 RPS (越高越好)")
    ax2.set_xlabel("前缀复用率"); ax2.set_ylabel("RPS")
    for a in (ax1, ax2):
        a.legend(); a.set_xticks([0, .25, .5, .75, 1.0]); a.grid(True, alpha=0.3)

    # 标注 SGLang 在 100% 复用率的吞吐飞跃
    ax2.annotate("SGLang 18.95\n(vLLM 8.88, 2.1x)", xy=(1.0, 18.95),
                 xytext=(0.55, 15), fontsize=9, color="#2d6a3e",
                 arrowprops=dict(arrowstyle="->", color="#55A868"))

    fig.suptitle("vLLM vs SGLang：前缀复用率对性能的影响 (Qwen2.5-1.5B, RTX3060)", fontsize=13)
    fig.tight_layout()
    fig.savefig("assets/prefix_ratio_comparison.png", dpi=150, bbox_inches="tight")
    print("saved assets/prefix_ratio_comparison.png")


if __name__ == "__main__":
    main()
