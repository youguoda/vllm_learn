import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("assets", exist_ok=True)


def load(fw, sc):
    with open(f"bench_results/{fw}_{sc}.csv") as f:
        return list(csv.DictReader(f))


# 图1: TTFT vs 输入长度
fig, ax = plt.subplots(figsize=(9, 5))
for fw, c, s in (("vllm", "#4C72B0", "o-"), ("sglang", "#55A868", "s--")):
    rows = load(fw, "latency")
    x = [float(r["input_len"]) for r in rows]
    y = [float(r["ttft_median_ms"]) for r in rows]
    ax.plot(x, y, s, color=c, label=fw, ms=8, lw=2)
ax.set_xlabel("输入长度 (近似 token)"); ax.set_ylabel("TTFT (ms)")
ax.set_title("TTFT vs 输入长度 (低并发, prefill 主导, 近似线性)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("assets/latency_ttft.png", dpi=150, bbox_inches="tight")
print("saved assets/latency_ttft.png")
plt.close(fig)

# 图2: ITL 分布
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ci, conc in enumerate((1, 4, 8)):
    ax = axes[ci]; ax.set_title(f"并发={conc}")
    for xi, (fw, col) in zip([0.7, 1.3], (("vllm", "#4C72B0"), ("sglang", "#55A868"))):
        rows = load(fw, "streaming")
        row = next(r for r in rows if int(float(r["concurrency"])) == conc)
        p50, p95, p99 = float(row["itl_p50_ms"]), float(row["itl_p95_ms"]), float(row["itl_p99_ms"])
        ax.bar(xi, p99, width=0.4, color=col, alpha=0.4)
        ax.bar(xi, p95, width=0.4, color=col, alpha=0.7)
        ax.bar(xi, p50, width=0.4, color=col)
        ax.text(xi, p50 + 0.1, f"{p50:.1f}", ha="center", fontsize=8)
    ax.set_xticks([0.7, 1.3]); ax.set_xticklabels(["vllm", "sglang"])
    ax.set_ylim(0, 14)
    if ci == 0: ax.set_ylabel("ITL (ms/token, 越低越流畅)")
fig.suptitle("ITL 分布 (token 间延迟, 深→浅 = P50/P95/P99)", fontsize=13)
fig.tight_layout(); fig.savefig("assets/itl_distribution.png", dpi=150, bbox_inches="tight")
print("saved assets/itl_distribution.png")
