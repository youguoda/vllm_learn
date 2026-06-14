import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("assets", exist_ok=True)


def parse(path):
    rows = []
    for line in open(path):
        parts = line.split(",")
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            pass
    return rows


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for fw, c in (("vllm", "#4C72B0"), ("sglang", "#55A868")):
    rows = parse(f"bench_results/gpu_{fw}.csv")
    t = list(range(len(rows)))
    ax1.plot(t, [r[0] for r in rows], color=c, label=fw, alpha=0.8, lw=1.2)
    ax2.plot(t, [r[1] for r in rows], color=c, label=fw, alpha=0.8, lw=1.2)
ax1.set_xlabel("时间 (s)"); ax1.set_ylabel("GPU 利用率 (%)")
ax1.set_title("GPU 计算利用率 (压测全程)"); ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim(0, 105)
ax2.set_xlabel("时间 (s)"); ax2.set_ylabel("显存使用 (MiB)")
ax2.set_title("显存占用 (压测全程)"); ax2.legend(); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("assets/p5_gpu_utilization.png", dpi=150, bbox_inches="tight")
print("saved assets/p5_gpu_utilization.png")
