# 05-11 详细计划：KV Cache 专题笔记（Week 3 交付）

## 步骤 1：汇总整理（90 min）

新建 `学习日志/KVCache专题对比.md`，按以下结构撰写，每节必须配源码引用或实验数据：

---

```markdown
# KV Cache 管理机制专题对比
## 1. 背景：为什么 KV Cache 是推理的核心瓶颈
（一段话：decode 访存密集 → KV 吃满显存 → batch 开不大 → GPU 摸鱼）

## 2. vLLM PagedAttention

### 2.1 数据结构
- KVCacheBlock: block_id / ref_count / block_hash / prev_block（来自 05-06 笔记）
- Block Table：逻辑 → 物理的映射数组

### 2.2 分配与释放
（allocate_slots / free_blocks 关键逻辑，配 05-06 的流程图）

### 2.3 APC（Automatic Prefix Caching）
- 链式哈希：hash(i) = f(hash(i-1), tokens_i)
- 命中粒度：16 token 整块
- 实验证据：APC ON vs OFF 对比表（04-26 数据）

## 3. SGLang RadixAttention

### 3.1 数据结构
- TreeNode: children / lock_ref / last_access_time / value（来自 05-07 笔记）

### 3.2 match_prefix / insert / evict
（配 tiny_radix.py 输出的树结构图）

### 3.3 HiCache（多级扩展）
- 三级结构与搬移时机（05-09 笔记）

## 4. 核心差异对照表

| 维度 | vLLM | SGLang |
|---|---|---|
| 索引结构 | 哈希表 | Radix 树 |
| 命中粒度 | 16 token（块对齐） | 任意长度 |
| split 操作 | 无 | 有（边可劈开） |
| 引用计数 | block.ref_count | node.lock_ref |
| 淘汰 | LRU free_list | evict() LRU 叶子 |
| 扩展 | CPU swap（重算恢复） | HiCache 三级 |

## 5. 实验数据：前缀复用率决定选型

（贴 assets/prefix_ratio_comparison.png）

**结论**（填入 05-10 的结论句）：
> 复用率 ___% 以上 SGLang 优势显著；___% 以下两者持平。
> 客服/agent/RAG 等高复用场景 → SGLang；通用负载 → vLLM

## 6. 常见误区
- ❌ "SGLang KV Cache 机制整体更好" → 只在高复用负载下优势明显
- ❌ "vLLM APC 没用" → 命中整块时效果接近 radix 树，实现简单可靠
- ❌ "HiCache 没延迟" → CPU 命中需要 PCIe 传输（ms 级），比 GPU 命中慢
```

---

## 步骤 2：选型决策树（30 min）

画一张"我该选哪个框架"的决策流程图，存 `assets/selection_guide.png`：

```python
# draw_selection.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0,10); ax.set_ylim(0,7); ax.axis('off')
    ax.set_title("vLLM vs SGLang 选型决策（KV Cache 视角）", fontsize=13)

    def dbox(x,y,w,h,t,c): 
        ax.add_patch(mpatches.FancyBboxPatch((x,y),w,h,
            boxstyle="round,pad=0.1",fc=c,ec="gray",lw=1.2))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9)
    def arr(x1,y1,x2,y2,t=""):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
            arrowprops=dict(arrowstyle="->",color="gray",lw=1.3))
        if t: ax.text((x1+x2)/2+0.1,(y1+y2)/2,t,fontsize=8,color="dimgray")

    dbox(3.5,6.0,3.0,0.7,"前缀复用率 > 50%？","#fff3cd")
    arr(5.0,6.0,3.5,5.0,"是"); dbox(2.0,4.3,3.0,0.7,"多轮对话/Agent/RAG？","#fff3cd")
    arr(5.0,6.0,6.5,5.0,"否"); dbox(5.5,4.3,3.0,0.7,"通用/低复用负载","#fff3cd")
    arr(3.5,4.3,2.5,3.3,"是"); dbox(1.0,2.6,2.8,0.7,"SGLang\n(RadixAttention 优势最大)","#d4edda")
    arr(3.5,4.3,4.5,3.3,"否，但有"); dbox(3.2,2.6,3.0,0.7,"vLLM APC 已够用\n（整块命中）","#dde8f5")
    arr(7.0,4.3,7.0,3.3,""); dbox(5.5,2.6,3.0,0.7,"vLLM\n(简单可靠，生态更完整)","#dde8f5")
    arr(2.5,2.6,2.5,1.7,"显存不够?")
    dbox(1.0,0.9,2.8,0.7,"+ HiCache\n(SGLang 三级缓存)","#fde8d8")

    fig.tight_layout()
    fig.savefig("assets/selection_guide.png", dpi=150, bbox_inches="tight")
```

## 步骤 3：自测（10 min）

回答 05-07 遗留的那道题：**为什么块哈希要链式包含前缀哈希？**

答案要点：Attention 是全局的——"今天适合出门"这句话对应的 KV 值，取决于前面所有 token 的上下文。只看本块内容相同不够，必须保证前缀完全一致才能复用。链式哈希用 O(1) 的代价保证了这个正确性（改动第 i 块，i 块之后的所有块哈希全变，绝对查不到错误缓存）。

## 今日产出
- [x] KVCache专题对比.md（完整，带源码引用+实验数字）→ [[KVCache专题对比]]
- [x] assets/selection_guide.png（选型决策树）
- [x] 自测题书面回答（链式哈希为何必要）

> 完成于 2026-06-14。Week 3 KV Cache 专题汇总，整合 M2/M3 全部源码与实验成果。
