import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False


def load(fw):
    with open(f"bench_results/{fw}_fixed_output.csv") as f:
        return list(csv.DictReader(f))


os.makedirs("assets", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for fw, c, s in (("vllm", "#4C72B0", "o-"), ("sglang", "#55A868", "s--")):
    rows = load(fw)
    x = [int(float(r["concurrency"])) for r in rows]
    y1 = [float(r["tok_per_s"]) for r in rows]
    y2 = [float(r["ttft_median_ms"]) for r in rows]
    ax1.plot(x, y1, s, color=c, label=fw, ms=8, lw=2)
    ax2.plot(x, y2, s, color=c, label=fw, ms=8, lw=2)
    # 拐点: 吞吐增速<30%(归一化到并发翻倍)
    for i in range(1, len(y1)):
        growth = (y1[i] - y1[i - 1]) / y1[i - 1]
        ratio = x[i] / x[i - 1]
        if growth / ratio < 0.5:  # 增速远低于并发增速 = 开始饱和
            ax1.axvline(x[i], color=c, ls=":", lw=1)
            break

ax1.set_title("吞吐量 (token/s, 越高越好)"); ax1.set_xlabel("并发数")
ax1.set_ylabel("tok/s"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.set_title("TTFT 中位数 (ms, 越低越好)"); ax2.set_xlabel("并发数")
ax2.set_ylabel("ms"); ax2.legend(); ax2.grid(alpha=0.3)
fig.suptitle("vLLM vs SGLang 吞吐/延迟 (固定输出, 无共享前缀, 3轮中位数)", fontsize=13)
fig.tight_layout()
fig.savefig("assets/throughput_v1.png", dpi=150, bbox_inches="tight")
print("保存 assets/throughput_v1.png")

# 打印吞吐增速分析(拐点)
print("\n吞吐增速分析(并发翻倍时吞吐增长比):")
for fw in ("vllm", "sglang"):
    rows = load(fw)
    y = [float(r["tok_per_s"]) for r in rows]
    x = [int(float(r["concurrency"])) for r in rows]
    print(f"  {fw}: ", end="")
    for i in range(1, len(y)):
        print(f"{x[i-1]}->{x[i]}: {y[i]/y[i-1]:.2f}x ", end="")
    print()
