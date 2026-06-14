import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("assets", exist_ok=True)


def load(fw):
    with open(f"bench_results/{fw}_multi_turn.csv") as f:
        return list(csv.DictReader(f))


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ci, conc in enumerate((1, 8, 16)):
    ax = axes[ci]; ax.set_title(f"并发={conc}")
    for fw, c, s in (("vllm", "#4C72B0", "o-"), ("sglang", "#55A868", "s--")):
        rows = load(fw)
        data = [r for r in rows if int(float(r["concurrency"])) == conc and r["turn"] != "rps"]
        x = [int(float(r["turn"])) for r in data]
        y = [float(r["ttft_median_ms"]) for r in data]
        ax.plot(x, y, s, color=c, label=fw, ms=8, lw=2)
    ax.set_xlabel("对话轮次"); ax.set_xticks([1, 2, 3, 4, 5])
    if ci == 0: ax.set_ylabel("TTFT (ms)")
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("多轮对话 TTFT 变化 (共享 system prompt，两框架都有前缀缓存)", fontsize=13)
fig.tight_layout()
fig.savefig("assets/multi_turn_ttft.png", dpi=150, bbox_inches="tight")
print("saved assets/multi_turn_ttft.png")
