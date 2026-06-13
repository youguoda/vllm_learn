"""PP 时序甘特图。/home/guoda/python/bin/python3 draw_pp.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    os.makedirs("assets", exist_ok=True)
    p, m = 4, 4
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for stage in range(p):
        for mb in range(m):
            t_start = stage + mb
            ax.barh(stage, 1, left=t_start, color=colors[mb], edgecolor="white", height=0.7)
            ax.text(t_start + 0.5, stage, f"m{mb+1}", ha="center", va="center",
                    fontsize=9, color="white")
        bubble_start = m + stage
        bubble_end = m + p - 1
        if bubble_start < bubble_end:
            ax.barh(stage, bubble_end - bubble_start, left=bubble_start,
                    color="#cccccc", edgecolor="white", height=0.7, alpha=0.5)
            ax.text(bubble_start + (bubble_end - bubble_start) / 2, stage, "气泡",
                    ha="center", va="center", fontsize=8, color="gray")
        # 起始气泡(stage 等待前序)
        if stage > 0:
            ax.barh(stage, stage, left=0, color="#eeeeee", edgecolor="white", height=0.7, alpha=0.5)

    ax.set_yticks(range(p))
    ax.set_yticklabels([f"卡{i}（段{i}）" for i in range(p)])
    ax.set_xlabel("时间单位")
    ax.set_title(f"Pipeline Parallelism 时序（p={p}段，m={m}微批次）"
                 f"  气泡比 = {p-1}/{m+p-1} ≈ {(p-1)/(m+p-1):.0%}")
    legend = [mpatches.Patch(color=colors[i], label=f"微批次 m{i+1}") for i in range(m)]
    legend.append(mpatches.Patch(color="#cccccc", alpha=0.5, label="气泡（空等）"))
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_xlim(0, m + p)
    fig.tight_layout()
    fig.savefig("assets/pp_schedule.png", dpi=150, bbox_inches="tight")
    print(f"气泡比: {(p-1)/(m+p-1):.1%}")


if __name__ == "__main__":
    main()
