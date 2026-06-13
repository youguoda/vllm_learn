---
title: Tensor Parallelism 对比 —— vLLM / SGLang 的 TP 实现与数学等价性
date: 2026-06-14
tags:
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/核心算法/分布式推理
  - output/active
related:
  - "[[vLLM源码精读-PagedAttention内存管理-06-14]]"
---

# Tensor Parallelism 对比

> 单卡环境（RTX 3060）无法真跑多卡 TP，但可以**读源码 + PyTorch 手写验证数学等价性**。今天搞懂：TP 怎么切权重、哪一步通信、为什么数学无损。

---

## 步骤 1：vLLM TP 源码

源码：`vllm/model_executor/layers/linear.py`。两个核心类：

| 问题 | ColumnParallelLinear | RowParallelLinear |
|---|---|---|
| 权重怎么切 | 按**列**切（输出维度 out_features） | 按**行**切（输入维度 in_features） |
| 前向输入 | 每卡拿完整输入，各算部分输出 | 输入已按列切（上层输出） |
| 哪步通信 | **不通信**（输出 concat） | **All-Reduce**（各卡输出相加） |
| 数学等价 | concat = 完整矩阵乘 | All-Reduce = 完整矩阵乘 |

**All-Reduce 调用位置**（linear.py:1558-1559）：

```python
def forward(self, input_):                          # RowParallelLinear
    ...
    output_parallel = self.quant_method.apply(self, input_parallel, bias_)
    if self.reduce_results and self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output_parallel)   # ← 唯一通信点
    else:
        output = output_parallel
```

> 只有 `tp_size > 1` 时才 all_reduce。注意 bias 只在 rank 0 加（`tp_rank > 0` 时 bias_=None），避免 TP>1 时 bias 被加多次——这是个容易忽略的正确性细节。

---

## 步骤 2：PyTorch 验证数学无损

[verify_tp.py](../verify_tp.py) 单卡模拟 tp=2，结果：

```
ColumnParallelLinear 最大误差: 9.54e-07
RowParallelLinear    最大误差: 4.77e-07
```

> 误差 ≈ 浮点精度（1e-6/1e-7），**TP 数学上完全无损**。这是 TP 能把大模型拆到多卡的理论保证——切开算再合并，结果和单卡完整算一模一样。

### 原理图解

```
ColumnParallel (按列切, 不通信):
  W[6,8] → 卡0:W[:3] 卡1:W[3:]
  x@W0.T=[4,3]  x@W1.T=[4,3]  → concat → [4,6]   (= x@W.T)

RowParallel (按行切, All-Reduce):
  W[6,8] → 卡0:W[:,:4] 卡1:W[:,4:]
  x[:,:4]@Wr0.T=[4,6]  x[:,4:]@Wr1.T=[4,6]  → 相加(All-Reduce) → [4,6]
  (分块矩阵乘: A@B = A[:,:k]@B[:k] + A[:,k:]@B[k:])
```

> **巧妙之处**：Transformer 层把 ColumnParallel 和 RowParallel **配对**用——QKV/gate 用 Column（切输出，不通信），out_proj/down 用 Row（切输入，All-Reduce）。Column 的输出正好是 Row 的"已切分输入"，所以中间**不需要额外通信**，一个 attention 或 MLP 块只在末尾 All-Reduce 一次。

---

## 步骤 3：SGLang TP 对比

SGLang 的 TP 实现同样源自 **Megatron-LM 范式**（`ColumnParallelLinear`/`RowParallelLinear`），与 vLLM 几乎一致。

| 对比点 | vLLM | SGLang |
|---|---|---|
| 范式来源 | Megatron-LM | Megatron-LM（相同） |
| All-Reduce | tensor_model_parallel_all_reduce | 同（NCCL 后端） |
| 通信时机 | RowParallel forward 末尾 | 同 |
| QKV 优化 | QKVParallelLinear（合并 QKV 投影） | 类似 |
| 主要差异 | — | 通信与计算 overlap 的调度细节 |

> **结论**：TP 实现两者**基本一致**（都是 Megatron 范式），不是核心区别点。差异集中在通信-计算 overlap 优化，对最终用户基本透明。这也是为什么 TP 不在我的"vLLM vs SGLang 选型"考量里——它是两者共享的底座能力。

---

## 步骤 4：通信量推导

一个 Transformer 层 TP=N 的 All-Reduce 通信量：

```
每层 2 次 All-Reduce (attention out_proj 1 次 + MLP down_proj 1 次)
每次 All-Reduce 通信量 ≈ batch × hidden × dtype_bytes × 2(N-1)/N
  (Ring All-Reduce: 每卡收发 2(N-1)/N × 数据量)

以 Qwen2.5-7B (hidden=3584, batch=16, fp16=2字节, TP=4) 为例:
  单次数据量 = 16 × 3584 × 2 = 114.7 KB
  All-Reduce 实际传输 ≈ 114.7 × 2×3/4 ≈ 172 KB
  每层 2 次 ≈ 344 KB, 28 层 ≈ 9.6 MB/step
```

> **自测：为什么 TP 数 N 越大，通信开销越大但也越必要？**
>
> - 显存：模型权重切 N 份，每卡只存 1/N → **N 越大越省显存**。
> - 通信：All-Reduce 量 ∝ 2(N-1)/N，N 从 2→4→8 时是 1→1.5→1.75，**趋近常数 2**（不是线性爆炸）。
> - 必要性：当模型超过单卡显存（如 70B fp16 = 140GB > 单卡 80GB），**不得不用 TP** 拆分——这时通信开销是"能跑起来"的必要代价。
> - 代价：TP 需要卡间高速互联（NVLink），跨节点（只有 PCIe/IB）时 All-Reduce 会成为瓶颈，所以 TP 一般限在单节点内（≤8 卡），跨节点用 PP。

---

## 今日产出

- [x] Column/RowParallelLinear 代码注释（all_reduce 在 linear.py:1559）
- [x] verify_tp.py 运行结果（误差 9.5e-7 / 4.8e-7，数学无损）
- [x] vLLM vs SGLang TP 对比表（结论：Megatron 范式一致，非区别点）
- [x] 通信量推导（Qwen2.5-7B TP=4 ≈ 9.6 MB/step）

## 一句话

> **TP 是 vLLM 和 SGLang 共享的 Megatron 底座**：按列/行切权重，配对使用让中间零通信，每层末尾 2 次 All-Reduce。数学无损（误差 1e-6）。它解决的是"单卡装不下大模型"，不是两框架的差异点。
