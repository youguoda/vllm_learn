# 05-17 详细计划：Pipeline Parallelism + Week 4 总结

## 步骤 1：理解 PP 气泡问题（50 min）

PP = 把 Transformer 的层按顺序切到多卡，卡 0 算第 1-N 层，卡 1 算第 N+1-2N 层……

**气泡问题**：单个 batch 时，卡 1 要等卡 0 算完才能开始，卡 0 算完后却在等卡 1（空等），大量时间浪费在等待上。

**解决方案：微批次（micro-batch）**——把一个大 batch 切成 m 个小批次，流水线化：

```
            时间 →
卡0(层1-N):  [m1][m2][m3][m4]   等  等  等  等
卡1(层N+1):       [m1][m2][m3][m4]  等  等  等
卡2(层2N+1):           [m1][m2][m3][m4]  等  等
```

气泡比 = (p-1)/(m+p-1)，其中 p=流水线段数，m=微批次数。

手算：p=4，m=8 时，气泡比 = 3/11 ≈ 27%；m=16 时 = 3/19 ≈ 16%。**微批次越多，气泡越小，但内存消耗越大（要同时存 m 个微批次的激活值）**。

## 步骤 2：画 PP 时序图（30 min）

画 4 段流水线 × 4 微批次的甘特图，存 `assets/pp_schedule.png`：

```python
# draw_pp.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

if __name__ == "__main__":
    p, m = 4, 4   # 4段流水线，4个微批次
    colors = ["#4C72B0","#55A868","#C44E52","#8172B2"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    
    # 前向传播（每个微批次经过每段耗时 1 单位）
    for stage in range(p):
        for mb in range(m):
            t_start = stage + mb          # 前向：依次推迟 1 单位
            ax.barh(stage, 1, left=t_start, color=colors[mb],
                    edgecolor='white', height=0.7)
            ax.text(t_start+0.5, stage, f"m{mb+1}",
                    ha='center', va='center', fontsize=9, color='white')
        # 气泡（等待）
        bubble_start = m + stage
        bubble_end   = m + p - 1
        if bubble_start < bubble_end:
            ax.barh(stage, bubble_end-bubble_start, left=bubble_start,
                    color='#cccccc', edgecolor='white', height=0.7, alpha=0.5)
            ax.text(bubble_start+(bubble_end-bubble_start)/2, stage,
                    "气泡", ha='center', va='center', fontsize=8, color='gray')

    ax.set_yticks(range(p))
    ax.set_yticklabels([f"卡{i}（段{i}）" for i in range(p)])
    ax.set_xlabel("时间单位")
    ax.set_title(f"Pipeline Parallelism 时序（p={p}段，m={m}微批次）\n气泡比 = {p-1}/{m+p-1} ≈ {(p-1)/(m+p-1):.0%}")
    
    legend = [mpatches.Patch(color=colors[i], label=f"微批次 m{i+1}") for i in range(m)]
    legend.append(mpatches.Patch(color='#cccccc', alpha=0.5, label='气泡（空等）'))
    ax.legend(handles=legend, loc='upper right', fontsize=8)
    ax.set_xlim(0, m+p)
    fig.tight_layout()
    fig.savefig("assets/pp_schedule.png", dpi=150, bbox_inches="tight")
    print(f"气泡比: {(p-1)/(m+p-1):.1%}")
```

## 步骤 3：读 vLLM + SGLang PP 实现要点（30 min）

```bash
# vLLM PP 入口
grep -rn "pipeline_parallel\|pp_rank\|is_last_pipeline_stage" $VLLM --include="*.py" -l | head -5
# SGLang PP 入口
grep -rn "pipeline_parallel\|pp_rank" $SGLANG --include="*.py" -l | head -5
```

不需要精读全部，只找两个问题的答案：
1. **层如何分配到 stage**：`num_hidden_layers // pp_size`，余数分给前几个 stage
2. **stage 间通信**：卡 i 把最后一层的隐状态 send 给卡 i+1，用 `torch.distributed.send/recv`（点对点，不是广播）

写进笔记：两框架 PP 实现基本一致，都源自 Megatron 范式，差异主要在虚拟流水线（vPP，把每个 stage 再细分成多个 chunk，进一步减少气泡）的支持程度。

## 步骤 4：Week 4 总结表（40 min）

新建 `学习日志/并行策略对比.md`：

```markdown
# 并行策略四维对比

## 速查表

| 并行方式 | 切分对象 | 通信原语 | 通信量 | 显存减少 | 适用场景 |
|---|---|---|---|---|---|
| TP | 单层权重（按行/列） | All-Reduce | ∝ batch×hidden | 1/N | 单机多卡，延迟敏感 |
| PP | 层（按深度） | P2P send/recv | ∝ micro_batch×hidden | 1/p（权重） | 跨机，超大模型 |
| EP | MoE 专家（按专家编号） | All-to-All | ∝ tokens×hidden×激活率 | 1/N（专家权重） | MoE 模型 |
| DP | 数据（请求级别） | All-Reduce（梯度） | 训练时大 | 不减少 | 推理时靠外部LB |

## 组合使用

- Llama 7B，单机 8 卡：TP=8（最常见）
- Llama 70B，2机16卡：TP=8 + PP=2
- DeepSeek-V3，多机：TP=8 + EP=64 + PP=N（按需）
- SGLang 推理多实例：EP=N + DP Attention

## 单卡实验的局限与外推

- 本实验（RTX 3060 单卡）只能单卡运行，TP/PP/EP 均无法直接实测
- 结论来源：源码阅读 + 数学推导 + 社区 benchmark（标注版本和硬件）
- 可外推：通信量公式、气泡比公式（与硬件无关的数学结论）
- 不可外推：绝对延迟、最优 TP 数（依赖 NVLink 带宽 vs 计算比）

## 自测题答案

Q: 为什么单机内用 TP，跨机用 PP？
A: 单机 NVLink 带宽 ~600 GB/s，All-Reduce 延迟 <0.1ms，TP 通信开销可接受；
   跨机 InfiniBand ~200 GB/s，延迟 ~10-100 倍，All-Reduce 变成瓶颈；
   PP 的 P2P 通信量小（只传 micro_batch 的激活，不传权重梯度），跨机也可接受。
```

## 今日产出
- [x] assets/pp_schedule.png（PP 时序甘特图）
- [x] 气泡比公式手算（p=4: m=4→43%, m=8→27%, m=16→16%）
- [x] 并行策略对比.md（四维表+组合场景+外推边界）→ [[并行策略对比]]

> 完成于 2026-06-14。Week4 总结：TP切权重/PP切层/EP分专家/DP复制，正交可组合。
> 都是 Megatron 底座，非两框架差异点。
