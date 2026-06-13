"""
画 KV Cache 架构对比图（三张）。用基础环境 python:
  /home/guoda/python/bin/python3 draw_arch.py

箭头标注的函数名均来自 05-06/05-07 源码精读笔记（真实源码）。
差异图数字均来自 06-13 实测（PrefixCaching / RadixAttention 多轮对话）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# WSL 实测可用中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, txt, color, fs=9):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1", fc=color, ec="gray", lw=1.2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, lbl="", color="steelblue"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    if lbl:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, lbl,
                fontsize=7.5, color=color, ha="center")


# ============ 图1：vLLM PagedAttention 流程 ============
def draw_paged():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("vLLM PagedAttention：KV Cache 管理全流程（标注真实源码函数）",
                 fontsize=14, pad=12)

    box(ax, 0.3, 5.2, 2.2, 0.8, "新请求\n(token ids)", "#dde8f5")
    box(ax, 3.0, 5.1, 2.7, 1.0,
        "get_computed_blocks()\nallocate_slots()\n查 APC 哈希表", "#fff3cd", 8.5)
    arrow(ax, 2.5, 5.6, 3.0, 5.6)

    # 命中
    box(ax, 3.0, 3.7, 2.7, 1.0,
        "命中 → touch()\n复活物理块, ref_cnt+1\n跳过 prefill", "#d4edda", 8.5)
    ax.text(4.35, 4.9, "命中", fontsize=8, color="green", ha="center")
    arrow(ax, 4.35, 5.1, 4.35, 4.7, color="green")

    # 未命中
    box(ax, 6.2, 3.7, 2.7, 1.0,
        "未命中 → get_new_blocks()\nfree_list 队头取(LRU)\n_maybe_evict_cached_block", "#f8d7da", 8)
    ax.text(7.55, 4.9, "未命中", fontsize=8, color="red", ha="center")
    arrow(ax, 5.7, 5.5, 6.4, 4.7, color="red")

    # Block Table
    box(ax, 4.7, 2.2, 3.0, 0.9, "Block Table\n[逻辑块 → 物理块 id]\n(间接寻址)", "#e2d9f3", 8.5)
    arrow(ax, 4.35, 3.7, 5.4, 3.1)
    arrow(ax, 7.55, 3.7, 6.9, 3.1)

    # Prefill/Decode
    box(ax, 9.4, 3.7, 2.9, 1.0,
        "Prefill + Decode\nattention kernel\n按 Block Table 读非连续KV", "#cfe2ff", 8)
    arrow(ax, 7.7, 2.65, 9.4, 4.0, "Block Table 传入")

    # 满块算哈希
    box(ax, 9.4, 2.2, 2.9, 0.9,
        "块满 → hash_block_tokens()\nhash((父块hash, 本块token))\n链式哈希, 写入 _block_hash", "#fff3cd", 7.5)
    arrow(ax, 10.85, 3.7, 10.85, 3.1)

    # EOS 释放
    box(ax, 9.4, 0.6, 2.9, 1.0,
        "EOS → free_blocks()\nref_cnt-1, 归还 free_list 队尾\n★哈希保留, 可被下次命中", "#d4edda", 7.5)
    arrow(ax, 10.85, 2.2, 10.85, 1.6)

    # 物理块池
    box(ax, 0.3, 2.2, 2.3, 1.5,
        "GPU 物理块池\nFreeKVCacheBlockQueue\n(手写双向链表)\nnum_gpu_blocks=14268", "#f5f5f5", 7.8)
    arrow(ax, 2.6, 2.95, 6.2, 4.0, "popleft 取块")
    arrow(ax, 9.4, 1.1, 2.6, 2.6, "append 归还", color="seagreen")

    ax.text(0.3, 0.25,
            "关键性质：①逻辑连续/物理离散(间接寻址)  ②按需分配(碎片98.8%→21.9%)  "
            "③链式哈希保证APC正确性  ④释放只减ref_cnt不清哈希(惰性失效)",
            fontsize=8.5, style="italic", color="dimgray")

    fig.tight_layout()
    fig.savefig("assets/pagedattention_arch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存 assets/pagedattention_arch.png")


# ============ 图2：SGLang RadixAttention 流程 ============
def draw_radix():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("SGLang RadixAttention：Radix 树 KV Cache 管理（标注真实源码函数）",
                 fontsize=14, pad=12)

    box(ax, 0.3, 5.2, 2.2, 0.8, "新请求\n(token ids)", "#dde8f5")
    box(ax, 3.0, 5.1, 2.9, 1.0,
        "match_prefix()\n从 root 逐边匹配\n_match_prefix_helper", "#fff3cd", 8.5)
    arrow(ax, 2.5, 5.6, 3.0, 5.6)

    # 命中
    box(ax, 3.0, 3.6, 2.9, 1.1,
        "命中 N token\n★任意长度, 不限块边界\n边匹配一半→_split_node\ninc_lock_ref (锁到root)", "#d4edda", 7.8)
    ax.text(4.45, 4.9, "命中", fontsize=8, color="green", ha="center")
    arrow(ax, 4.45, 5.1, 4.45, 4.7, color="green")

    # 未命中
    box(ax, 6.4, 3.7, 2.9, 1.0,
        "剩余 token\nKV Pool 分配新内存\n做 prefill", "#f8d7da", 8.5)
    ax.text(7.85, 4.9, "未命中", fontsize=8, color="red", ha="center")
    arrow(ax, 5.9, 5.5, 6.6, 4.7, color="red")

    # KV Pool
    box(ax, 0.3, 2.2, 2.4, 1.0,
        "KV Pool\ntoken_to_kv_pool\n(page_size=1)", "#f5f5f5", 8)
    arrow(ax, 2.7, 2.7, 6.4, 4.0, "取内存")

    # decode
    box(ax, 9.7, 3.7, 2.6, 1.0, "Decode\n(生成 token)", "#cfe2ff", 9)
    arrow(ax, 9.3, 4.2, 9.7, 4.2)

    # insert
    box(ax, 6.4, 2.2, 2.9, 0.9,
        "insert() / _insert_helper\n写回 radix 树\n必要时 _split_node 劈边", "#fff3cd", 8)
    arrow(ax, 7.85, 3.7, 7.85, 3.1)
    arrow(ax, 11.0, 3.7, 9.3, 2.65, "EOS后写回")

    # evict
    box(ax, 6.4, 0.6, 2.9, 1.0,
        "evict()\n最小堆 LRU, 叶子优先\nlock_ref==0 才可淘汰\n父变叶子→重新入堆", "#fde8d8", 7.8)
    arrow(ax, 7.85, 2.2, 7.85, 1.6)

    # 树结构示意（不用 monospace，避免中文缺字）
    ax.text(0.4, 1.6, "Radix 树示意 (page_size=1, 任意位置 split)：",
            fontsize=8.5, color="dimgray")
    ax.text(0.4, 1.05, "root ─[ 系统提示 S ]─┬─[ 问题A ]  val→KV",
            fontsize=8, color="dimgray")
    ax.text(0.4, 0.7, "                        └─[ 问题B ]  val→KV",
            fontsize=8, color="dimgray")

    ax.text(0.3, 0.12,
            "关键性质：①任意长度前缀命中(page_size=1)  ②split劈边→无冗余存储  "
            "③lock_ref沿路径锁到root  ④树结构天然保证前缀唯一(无需哈希)",
            fontsize=8.5, style="italic", color="dimgray")

    fig.tight_layout()
    fig.savefig("assets/radixattention_arch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存 assets/radixattention_arch.png")


# ============ 图3：三维差异（真实实测数据）============
def draw_diff():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("PagedAttention vs RadixAttention：三维差异（数据均为本机实测）",
                 fontsize=13)

    # 子图1：命中粒度
    ax = axes[0]; ax.set_title("① 命中粒度（最小复用单位）")
    bars = ax.barh(["vLLM\n(块哈希)", "SGLang\n(radix树)"], [16, 1],
                   color=["#f8d7da", "#d4edda"])
    ax.set_xlabel("最小命中单位 (token)")
    ax.bar_label(bars, labels=["block_size=16", "page_size=1"], padding=4, fontsize=9)
    ax.set_xlim(0, 20)

    # 子图2：共享前缀 warm TTFT（06-13 实测, 726 token 前缀）
    ax = axes[1]; ax.set_title("② 共享前缀 warm TTFT\n(≈726 token 前缀, 第2轮起)")
    vals = [33.3, 29.3]   # vLLM APC / SGLang Radix, 均为实测
    bars = ax.bar(["vLLM\nAPC", "SGLang\nRadix"], vals, color=["#dde8f5", "#d4edda"])
    ax.set_ylabel("TTFT (ms, 越低越好)"); ax.set_ylim(0, 50)
    ax.bar_label(bars, labels=[f"{v}ms" for v in vals], padding=3, fontsize=9)

    # 子图3：半块对齐前缀（06-13 实测, 117 token 共享前缀）
    ax = axes[2]; ax.set_title("③ 半块对齐前缀命中\n(117 token, 截断点不对齐块边界)")
    bars = ax.bar(["vLLM\n(块哈希)", "SGLang\n(radix树)"], [96, 111],
                  color=["#f8d7da", "#d4edda"])
    ax.set_ylabel("实际命中 token 数 (满分117)"); ax.set_ylim(0, 130)
    ax.axhline(117, color="gray", ls="--", lw=0.8)
    ax.text(1.0, 119, "共享上限117", fontsize=7.5, color="gray", ha="center")
    ax.bar_label(bars, labels=["96 (82.1%)", "111 (94.9%)"], padding=3, fontsize=9)
    ax.text(0.5, 60, "SGLang 多救回 15 token\n= 一个 block 的对齐损失",
            fontsize=8, ha="center", color="dimgray",
            bbox=dict(boxstyle="round", fc="#fffbe6", ec="gray", lw=0.6))

    fig.tight_layout()
    fig.savefig("assets/kvcache_diff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存 assets/kvcache_diff.png")


if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)
    draw_paged()
    draw_radix()
    draw_diff()
