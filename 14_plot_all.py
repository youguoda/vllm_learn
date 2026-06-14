"""一键生成 4 张主分析图。/home/guoda/python/bin/python3 14_plot_all.py"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("assets", exist_ok=True)

D = json.load(open("bench_results/all_data.json"))
COLORS = {"vllm": "#4C72B0", "sglang": "#55A868"}
STYLES = {"vllm": "o-", "sglang": "s--"}


def getx(sc, fw, xk, yk, filt=None):
    rows = D.get(sc, {}).get(fw, [])
    if filt:
        rows = [r for r in rows if filt(r)]
    xs = [r[xk] for r in rows if xk in r and yk in r]
    ys = [r[yk] for r in rows if xk in r and yk in r]
    return xs, ys


def line_plot(fname, sc, xk, yk, xlabel, ylabel, title, filt=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for fw in ("vllm", "sglang"):
        x, y = getx(sc, fw, xk, yk, filt)
        if x:
            ax.plot(x, y, STYLES[fw], color=COLORS[fw], label=fw, ms=8, lw=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"assets/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved assets/{fname}")


line_plot("p1_throughput.png", "fixed_output", "concurrency", "tok_per_s",
          "并发请求数", "吞吐量 (token/s)", "吞吐量对比 (固定输出 128 token, 无共享前缀)")
line_plot("p2_ttft_vs_conc.png", "fixed_output", "concurrency", "ttft_median_ms",
          "并发请求数", "TTFT 中位数 (ms)", "首 Token 延迟 vs 并发数")
line_plot("p3_ttft_vs_input.png", "latency", "input_len", "ttft_median_ms",
          "输入长度 (近似 token)", "TTFT 中位数 (ms)", "首 Token 延迟 vs 输入长度 (prefill 主导)")
line_plot("p4_multi_turn.png", "multi_turn", "turn", "ttft_median_ms",
          "对话轮次", "TTFT 中位数 (ms)", "多轮对话 TTFT (并发=8, 共享 system prompt)",
          filt=lambda r: r.get("concurrency") == 8.0 and str(r.get("turn")) != "rps")
print("四张主图完成")
