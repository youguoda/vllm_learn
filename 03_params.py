"""
vLLM 关键参数探索
在 RTX 3060 12GB 上测试不同参数对性能和内存的影响

关键参数:
- --max-model-len: 最大上下文长度，直接影响 KV cache 大小
- --gpu-memory-utilization: GPU 显存利用率 (0-1)
- --tensor-parallel-size: 张量并行数 (需要多卡)
- --enforce-eager: 禁用 CUDA Graph，减少显存占用但降低性能
"""

import time
from vllm import LLM, SamplingParams


def benchmark(label, llm, prompts, sampling_params):
    """简单基准测试"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_in_tokens = sum(len(o.prompt_token_ids) for o in outputs)

    print(f"  Total input tokens:  {total_in_tokens}")
    print(f"  Total output tokens: {total_out_tokens}")
    print(f"  Time:                {elapsed:.2f}s")
    print(f"  Output throughput:   {total_out_tokens / elapsed:.1f} tok/s")

    # 打印第一个输出作为示例
    print(f"\n  Sample output:")
    print(f"    Prompt:    {outputs[0].prompt!r}")
    text = outputs[0].outputs[0].text[:120]
    print(f"    Generated: {text!r}...")

    return elapsed, total_out_tokens


def main():
    prompts = [
        "The meaning of life is",
        "In the year 2050,",
        "The most important invention in history is",
    ]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=64)

    # --- 参数组合 1: 短上下文 ---
    print("\n\n>>> Test 1: max_model_len=2048, gpu_mem=0.85")
    llm1 = LLM(
        model="Qwen/Qwen2.5-1.5B",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    benchmark("max_model_len=2048, gpu_mem=0.85", llm1, prompts, sampling_params)
    del llm1
    time.sleep(2)

    # --- 参数组合 2: 长上下文 ---
    print("\n\n>>> Test 2: max_model_len=4096, gpu_mem=0.90")
    llm2 = LLM(
        model="Qwen/Qwen2.5-1.5B",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    benchmark("max_model_len=4096, gpu_mem=0.90", llm2, prompts, sampling_params)
    del llm2
    time.sleep(2)

    # --- 参数组合 3: enforce_eager (无 CUDA Graph) ---
    print("\n\n>>> Test 3: max_model_len=4096, gpu_mem=0.85, enforce_eager=True")
    llm3 = LLM(
        model="Qwen/Qwen2.5-1.5B",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,  # 禁用 CUDA Graph, 省显存但略慢
    )
    benchmark("enforce_eager=True (no CUDA Graphs)", llm3, prompts, sampling_params)
    del llm3

    print("\n\nDone! Key takeaways:")
    print("  - max_model_len 越大, KV cache 越大, 能处理的上下文越长")
    print("  - gpu_memory_utilization 控制显存预留比例")
    print("  - enforce_eager=True 跳过 CUDA Graph 捕获, 启动快但推理慢")
    print("  - tensor-parallel-size 用于多卡, 单卡=1")


if __name__ == "__main__":
    main()
