"""
verify_tp.py — 单卡模拟 Tensor Parallelism，验证切分+合并数学无损

对照 vLLM 源码 (model_executor/layers/linear.py):
  - ColumnParallelLinear: 按列(输出维度)切权重, 各卡算部分输出, concat, 不通信
  - RowParallelLinear:    按行(输入维度)切权重, 各卡算 partial sum,
                          All-Reduce 相加 (linear.py:1559 tensor_model_parallel_all_reduce)
"""
import torch

torch.manual_seed(42)


def main():
    # hidden=8, out=6, tp=2 (模拟 2 卡)
    x = torch.randn(4, 8)   # batch=4, input=8
    W = torch.randn(6, 8)   # 完整权重 [out=6, in=8]
    b = torch.randn(6)

    # === 参考: 完整计算 ===
    ref = x @ W.T + b       # [4, 6]

    # === ColumnParallelLinear: 按列切(输出维度) ===
    W0, W1 = W[:3, :], W[3:, :]    # 卡0/卡1 各 [3,8]
    b0, b1 = b[:3], b[3:]
    out0 = x @ W0.T + b0          # 卡0 [4,3]
    out1 = x @ W1.T + b1          # 卡1 [4,3]
    col_tp = torch.cat([out0, out1], dim=-1)   # concat → [4,6], 不通信

    # === RowParallelLinear: 按行切(输入维度) + All-Reduce ===
    W_row = torch.randn(6, 8)
    Wr0, Wr1 = W_row[:, :4], W_row[:, 4:]      # 卡0/卡1 各 [6,4]
    x0, x1 = x[:, :4], x[:, 4:]                # 输入按列切(来自上层 ColumnParallel)
    partial0 = x0 @ Wr0.T          # 卡0 partial sum [4,6]
    partial1 = x1 @ Wr1.T          # 卡1 partial sum [4,6]
    row_tp = partial0 + partial1   # All-Reduce(相加) → [4,6]
    ref_row = x @ W_row.T

    print("=" * 56)
    print("  Tensor Parallelism 数学等价性验证 (单卡模拟 tp=2)")
    print("=" * 56)
    print(f"\n  ColumnParallelLinear 最大误差: {(col_tp - ref).abs().max().item():.2e}")
    print(f"  RowParallelLinear    最大误差: {(row_tp - ref_row).abs().max().item():.2e}")
    print(f"\n  → TP 数学无损(误差 ≈ 浮点精度 1e-6), 这是 TP 能拆分模型的理论保证")
    print("\n  通信发生位置:")
    print("    ColumnParallel: 不通信(各卡独立算, 最后 concat)")
    print("    RowParallel:    All-Reduce(各卡 partial sum 相加)")
    print("    → 一个 Transformer 层 = ColumnParallel(QKV/gate) + RowParallel(out/down)")
    print("      = 每层 2 次 All-Reduce (attention 1 次 + MLP 1 次)")


if __name__ == "__main__":
    main()
