# 05-15 详细计划：Tensor Parallelism 对比（单卡：读码 + 手写验证）

## 步骤 1：读 vLLM TP 实现（60 min）

```bash
VLLM=$(python3 -c "import vllm; print(vllm.__file__.replace('/__init__.py',''))")
# 找并行 Linear 层
grep -rn "class ColumnParallelLinear\|class RowParallelLinear" $VLLM --include="*.py" -l
# 预期: vllm/model_executor/layers/linear.py
```

精读两个类，回答：

| 问题 | ColumnParallelLinear | RowParallelLinear |
|---|---|---|
| 权重怎么切？ | 按**列**切（输出维度切分） | 按**行**切（输入维度切分） |
| 前向输入怎么处理？ | 每卡拿完整输入，各自算部分输出 | 输入已被按列切（上一层的输出） |
| 哪一步通信？ | 不通信（输出 concat） | **All-Reduce**（各卡输出相加） |
| 数学等价性？ | 是，concat 结果 = 完整矩阵乘结果 | 是，All-Reduce = 完整矩阵乘结果 |

找到代码里 `all_reduce` 的调用位置，记录行号。

## 步骤 2：20 行 PyTorch 验证 TP 数学等价性（40 min）

单卡模拟两张卡的计算，验证切分 + 合并的结果与原始矩阵乘完全相同：

```python
# verify_tp.py — 手动模拟 Tensor Parallelism，验证数学无损
import torch
torch.manual_seed(42)

if __name__ == "__main__":
    # 假设 hidden_size=8, out_features=6, tp=2（模拟2卡）
    x = torch.randn(4, 8)          # batch=4, input=8
    W = torch.randn(6, 8)          # 完整权重 [out=6, in=8]
    b = torch.randn(6)

    # === 完整计算（参考值）===
    ref = x @ W.T + b              # [4, 6]

    # === TP=2 模拟：ColumnParallelLinear ===
    # 按列切权重（按输出维度切）
    W0 = W[:3, :]                  # 卡0: [3, 8]
    W1 = W[3:, :]                  # 卡1: [3, 8]
    b0, b1 = b[:3], b[3:]

    out0 = x @ W0.T + b0           # 卡0输出 [4, 3]
    out1 = x @ W1.T + b1           # 卡1输出 [4, 3]
    col_tp = torch.cat([out0, out1], dim=-1)   # concat → [4, 6]

    # === TP=2 模拟：RowParallelLinear ===
    # 接上一层 ColumnParallel 的输出（每卡拿一半）
    # 按行切权重（按输入维度切）
    W_row = torch.randn(6, 8)
    Wr0 = W_row[:, :4]             # 卡0: [6, 4]，处理输入前半
    Wr1 = W_row[:, 4:]             # 卡1: [6, 4]，处理输入后半

    # 输入也按对应方式切（来自 ColumnParallel 的各卡输出）
    x_split0 = x[:, :4]
    x_split1 = x[:, 4:]

    partial0 = x_split0 @ Wr0.T   # 卡0: [4, 6]（partial sum）
    partial1 = x_split1 @ Wr1.T   # 卡1: [4, 6]（partial sum）
    row_tp = partial0 + partial1   # All-Reduce（相加）→ [4, 6]

    ref_row = x @ W_row.T

    print("=== ColumnParallelLinear 验证 ===")
    print(f"最大误差: {(col_tp - ref).abs().max().item():.2e}")   # 应 ≈ 1e-6

    print("\n=== RowParallelLinear 验证 ===")
    print(f"最大误差: {(row_tp - ref_row).abs().max().item():.2e}")  # 应 ≈ 1e-6

    print("\n通信发生位置：")
    print("  ColumnParallel: 不通信（各卡独立，最后 concat）")
    print("  RowParallel:    All-Reduce（各卡 partial sum 相加）")
    print("  → 一个 Transformer 层 = 1次 ColumnParallel + 1次 RowParallel = 1次 All-Reduce")
```

运行后把误差数字记进笔记：**TP 数学上无损（浮点误差 ≈ 1e-6），这是 TP 能拆分模型的理论保证。**

## 步骤 3：读 SGLang TP 实现 + 对比（30 min）

```bash
source ~/venv-sglang/bin/activate
SGLANG=$(python3 -c "import sglang; print(sglang.__file__.replace('/__init__.py',''))")
grep -rn "ColumnParallelLinear\|RowParallelLinear" $SGLANG --include="*.py" -l
```

两者都源自 Megatron-LM 的并行范式，找到 SGLang 对应实现后对比：

| 对比点 | vLLM | SGLang |
|---|---|---|
| All-Reduce 库 | torch.distributed / 自定义 | 同，或 NCCL 直调 |
| 通信时机 | RowParallel forward() 末尾 | 同 |
| QKV 专项优化 | QKVParallelLinear | 类似 |
| 主要差异 | | |

**结论**：TP 实现两者基本一致（都来自 Megatron 范式），差异集中在通信调度和 overlap（计算与通信重叠），不是核心区别点。

## 步骤 4：通信量推导（20 min）

手算一个 Transformer 层的 TP=N 通信量，写进笔记：

```
MLP 部分（two-linear）：
  - ColumnParallel（gate proj）：不通信
  - RowParallel（down proj）：All-Reduce，每卡发送 batch × hidden_size × 2字节
Attention 部分（QKV + out proj）：
  - 同理，1次 All-Reduce

每层共 2次 All-Reduce，每次通信量 = batch × hidden × dtype_bytes × (N-1)/N

以 Qwen2.5-7B（hidden=3584, batch=16, fp16, TP=4）为例：
  每次 All-Reduce ≈ 16 × 3584 × 2 × (3/4) ≈ 172 KB
  每层 2次 ≈ 344 KB，28层 ≈ 9.7 MB/step
```

自测：为什么 TP 数 N 越大，通信开销越大但也越必要？（显存减少 N 倍，但通信量 ∝ (N-1)/N 接近常数——超过单卡显存限制时不得不用 TP）

## 今日产出
- [ ] Column/RowParallelLinear 代码注释（含 all_reduce 行号）
- [ ] verify_tp.py 运行结果（误差 ≈ 1e-6）
- [ ] vLLM vs SGLang TP 对比表
- [ ] 通信量推导（手算）
