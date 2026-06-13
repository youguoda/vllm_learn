"""check_hw.py — 确认 GPU 量化支持范围"""
import torch

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {cap}")
    print("量化支持:")
    print(f"  FP8 W8A8 推理: {'OK' if cap >= (9,0) else 'NO 需 Hopper sm_90+'}")
    print(f"  INT8 计算: {'OK' if cap >= (8,0) else 'NO'}")
    print(f"  INT4 AWQ/GPTQ 权重: OK (dequant to fp16)")
    print(f"  FP8 KV cache 存储(e5m2): OK")
