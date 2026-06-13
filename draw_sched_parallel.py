"""05-18 PPT 素材：调度对比图 + 并行表 + 决策图。/home/guoda/python/bin/python3 draw_sched_parallel.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, txt, color, fs=9):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                         fc=color, ec="gray", lw=1.2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)


def arr(ax, x1, y1, x2, y2, lbl="", color="steelblue"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4))
    if lbl:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, lbl, fontsize=8, color=color, ha="center")


def draw_scheduler_compare():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("调度策略对比：vLLM vs SGLang", fontsize=14)
    for ax, title in [(ax1, "vLLM 调度器（FCFS + 抢占）"), (ax2, "SGLang 调度器（缓存感知）")]:
        ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.axis("off")
        ax.set_title(title, fontsize=12)

    # vLLM
    box(ax1, 2.5, 6.8, 3.0, 0.7, "新请求", "#dde8f5")
    box(ax1, 2.5, 5.6, 3.0, 0.7, "WAITING 队列 (FCFS)", "#fff3cd")
    arr(ax1, 4.0, 6.8, 4.0, 6.3)
    box(ax1, 2.3, 4.3, 3.4, 0.7, "token_budget 检查\n(max_num_batched_tokens)", "#ffe0b2", 8)
    arr(ax1, 4.0, 5.6, 4.0, 5.0)
    box(ax1, 2.5, 3.0, 3.0, 0.7, "RUNNING (Continuous Batching)", "#d4edda", 8)
    arr(ax1, 4.0, 4.3, 4.0, 3.7)
    box(ax1, 0.3, 1.5, 2.8, 0.9, "PREEMPTED\nfree KV+重算(RECOMPUTE)\n→ 回 WAITING 队首", "#f8d7da", 7.5)
    box(ax1, 4.9, 1.6, 2.8, 0.7, "FINISHED", "#e2d9f3")
    arr(ax1, 3.3, 3.0, 1.9, 2.4, "KV不足", "crimson")
    arr(ax1, 4.7, 3.0, 6.3, 2.3, "EOS", "seagreen")
    ax1.text(0.2, 0.5, "特点：FCFS 公平调度，被动查 APC，抢占=重算(v1 无 swap)",
             fontsize=8, style="italic", color="dimgray")

    # SGLang
    box(ax2, 2.5, 6.8, 3.0, 0.7, "新请求", "#dde8f5")
    box(ax2, 2.3, 5.6, 3.4, 0.7, "calc_priority\nRadix 树 match_prefix", "#fff3cd", 8)
    arr(ax2, 4.0, 6.8, 4.0, 6.3)
    box(ax2, 0.2, 4.2, 3.0, 1.0, "命中 → cache-aware 排序\n(LPM/DFS_WEIGHT)\n同前缀集中调度", "#d4edda", 7.5)
    box(ax2, 4.8, 4.3, 3.0, 0.9, "未命中 → 普通排队\n分配 KV Pool", "#fff3cd", 8)
    arr(ax2, 3.5, 5.6, 1.7, 5.3, "命中", "seagreen")
    arr(ax2, 4.5, 5.6, 6.3, 5.3, "未命中", "darkorange")
    box(ax2, 2.5, 2.9, 3.0, 0.7, "RUNNING (Continuous Batching)", "#d4edda", 8)
    arr(ax2, 1.7, 4.2, 3.4, 3.6)
    arr(ax2, 6.3, 4.3, 4.6, 3.6)
    box(ax2, 2.3, 1.5, 3.4, 0.7, "EOS → insert 写回 radix 树", "#e2d9f3", 8)
    arr(ax2, 4.0, 2.9, 4.0, 2.2)
    ax2.text(0.2, 0.5, "特点：缓存感知调度(主动把同前缀集中)，淘汰靠 evict 堆 LRU",
             fontsize=8, style="italic", color="dimgray")

    fig.tight_layout()
    fig.savefig("assets/scheduler_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved assets/scheduler_compare.png")


def draw_parallelism_table():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    headers = ["并行方式", "切分对象", "通信原语", "通信量(相对)", "显存节省", "典型场景"]
    rows = [
        ["TP\nTensor Parallel", "单层权重\n(行/列切)", "All-Reduce", "高\n(每层2次)", "权重 ÷N", "单机多卡\n7B~70B"],
        ["PP\nPipeline Parallel", "层\n(按深度切)", "P2P send/recv", "低\n(只传激活)", "权重 ÷p", "跨机\n超大模型"],
        ["EP\nExpert Parallel", "MoE 专家\n(按编号)", "All-to-All", "中\n(定向路由)", "专家权重 ÷N", "MoE 模型\nDeepSeek/Mixtral"],
        ["DP\nData Parallel", "请求/数据\n(按样本)", "外部 LB / DP-Attn", "最低\n(无KV共享)", "无", "推理扩容\n多实例"],
    ]
    colors_row = ["#dde8f5", "#d4edda", "#fff3cd", "#fde8d8"]
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontsize=10)
        else:
            cell.set_facecolor(colors_row[r - 1])
        cell.set_edgecolor("white"); cell.set_height(0.18)
    ax.set_title("推理框架并行策略速查表（vLLM & SGLang 共享）", fontsize=13, pad=15)
    fig.tight_layout()
    fig.savefig("assets/parallelism_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved assets/parallelism_table.png")


def draw_parallelism_decision():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("并行策略决策图：什么情况用什么并行", fontsize=13)
    box(ax, 4.0, 7.0, 3.0, 0.7, "单卡放得下模型？", "#fff3cd")
    arr(ax, 4.0, 7.0, 2.0, 6.2, "是")
    box(ax, 0.7, 5.5, 2.8, 0.7, "不用并行(单卡)", "#d4edda")
    arr(ax, 7.0, 7.0, 8.5, 6.2, "否")
    box(ax, 7.2, 5.5, 3.2, 0.7, "MoE 模型？", "#fff3cd")
    arr(ax, 8.8, 5.5, 7.0, 4.7, "是")
    box(ax, 5.0, 3.9, 3.6, 0.8, "EP(专家) + DP-Attn\n+ 必要时 TP/PP", "#fde8d8", 8.5)
    arr(ax, 8.8, 5.5, 9.6, 4.7, "否(Dense)")
    box(ax, 8.8, 3.9, 2.0, 0.8, "跨机?", "#fff3cd")
    arr(ax, 9.4, 3.9, 8.4, 3.0, "否")
    box(ax, 6.7, 2.2, 2.4, 0.7, "TP(单机多卡)", "#dde8f5", 8.5)
    arr(ax, 10.0, 3.9, 10.0, 3.0, "是")
    box(ax, 9.2, 2.2, 2.2, 0.7, "TP × PP 组合", "#dde8f5", 8.5)
    ax.text(0.5, 0.6, "第一问永远是'单卡放不放得下'，不是'哪个更快'。\n"
            "并行是'能不能跑大模型'的问题，与 vLLM/SGLang 选型(KV/调度)正交。",
            fontsize=9, style="italic", color="dimgray")
    fig.tight_layout()
    fig.savefig("assets/parallelism_decision.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved assets/parallelism_decision.png")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    draw_scheduler_compare()
    draw_parallelism_table()
    draw_parallelism_decision()
