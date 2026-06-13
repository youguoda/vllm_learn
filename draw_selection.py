"""选型决策树。/home/guoda/python/bin/python3 draw_selection.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    os.makedirs("assets", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("vLLM vs SGLang 选型决策（KV Cache 视角）", fontsize=13)

    def dbox(x, y, w, h, t, c):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.1", fc=c, ec="gray", lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)

    def arr(x1, y1, x2, y2, t=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.3))
        if t:
            ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, t, fontsize=8, color="dimgray")

    dbox(3.5, 6.0, 3.0, 0.7, "前缀复用率 > 75%？", "#fff3cd")
    arr(5.0, 6.0, 3.5, 5.0, "是")
    dbox(2.0, 4.3, 3.0, 0.7, "多轮对话/Agent/RAG？", "#fff3cd")
    arr(5.0, 6.0, 6.5, 5.0, "否")
    dbox(5.5, 4.3, 3.0, 0.7, "通用/低复用负载", "#fff3cd")
    arr(3.5, 4.3, 2.5, 3.3, "是")
    dbox(0.9, 2.6, 3.0, 0.7, "SGLang\n(吞吐可达 2.1x)", "#d4edda")
    arr(4.6, 4.3, 4.6, 3.3, "否但有共享")
    dbox(3.2, 2.6, 3.0, 0.7, "vLLM APC 已够用\n(整块命中)", "#dde8f5")
    arr(7.0, 4.3, 7.0, 3.3, "")
    dbox(5.6, 2.6, 3.2, 0.7, "vLLM\n(简单可靠,生态完整)", "#dde8f5")
    arr(2.4, 2.6, 2.4, 1.7, "显存不够?")
    dbox(0.9, 0.9, 3.0, 0.7, "+ HiCache\n(SGLang 三级,加 direct backend)", "#fde8d8")

    fig.tight_layout()
    fig.savefig("assets/selection_guide.png", dpi=150, bbox_inches="tight")
    print("saved assets/selection_guide.png")


if __name__ == "__main__":
    main()
