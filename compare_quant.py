"""
compare_quant.py — FP16 vs AWQ INT4 对比(显存/速度/质量)
单卡顺序加载两个模型(每次 del 释放)。
用法: HF_HUB_OFFLINE=1 NO_PROXY="*" python3 compare_quant.py
"""
import gc
import time

import torch
from vllm import LLM, SamplingParams

PROMPT = ("请详细解释工业传感器中压电式加速度计的工作原理，"
          "包括物理原理、结构设计、信号处理流程和典型应用场景。")
PARAMS = SamplingParams(temperature=0.0, max_tokens=300)


def bench(model_path, label, n=3):
    print(f"\n加载 {label} ({model_path}) ...")
    llm = LLM(model=model_path, max_model_len=4096,
              gpu_memory_utilization=0.85, enforce_eager=True)
    # 模型权重显存(从 vLLM 日志拿不到, 用 torch 估)
    torch.cuda.synchronize()
    mem = torch.cuda.memory_allocated() / 1e9

    llm.generate([PROMPT], PARAMS, use_tqdm=False)  # 预热
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = llm.generate([PROMPT], PARAMS, use_tqdm=False)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / n
    ntok = len(out[0].outputs[0].token_ids)
    tps = ntok / avg
    preview = out[0].outputs[0].text[:80].replace("\n", " ")
    print(f"  权重显存(torch): {mem:.2f} GB")
    print(f"  平均: {avg:.2f}s, {ntok} token, {tps:.1f} tok/s")
    print(f"  输出预览: {preview}...")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    return mem, tps


def main():
    results = {}
    results["fp16"] = bench("Qwen/Qwen2.5-1.5B", "FP16 (base)")
    results["awq"] = bench("Qwen/Qwen2.5-1.5B-Instruct-AWQ", "AWQ INT4")

    print("\n" + "=" * 45)
    print(f"  {'版本':<10} {'权重显存(GB)':>14} {'tok/s':>10}")
    for k, (mem, tps) in results.items():
        print(f"  {k:<10} {mem:>14.2f} {tps:>10.1f}")
    fm, ft = results["fp16"]
    am, at = results["awq"]
    print(f"\n  显存节省: {(1 - am / fm) * 100:.0f}%  (AWQ 权重 4bit vs fp16 16bit)")
    print(f"  速度变化: {at / ft:.2f}x  (>1 更快, <1 更慢)")


if __name__ == "__main__":
    main()
