"""用实测吞吐算每百万 token 成本"""
import json
D = json.load(open("bench_results/all_data.json"))

# 云 GPU 参考价(2026 按需价区间, RMB/小时)
GPUS = {"RTX 3060级(消费)": 1.5, "RTX 4090级": 3.5, "A10": 5.0, "A100 80G": 18.0}

print("=== 每百万 token 成本估算 ===\n")
for fw in ("vllm", "sglang"):
    best = max(D["fixed_output"][fw], key=lambda r: r["tok_per_s"])
    tps = best["tok_per_s"]
    tph = tps * 3600  # token/小时
    print(f"{fw}: 最高吞吐 {tps:.0f} tok/s = {tph/1e6:.2f} M token/小时")
    for gpu, price in GPUS.items():
        cost_per_m = price / (tph / 1e6)
        print(f"  {gpu} (¥{price}/h): ¥{cost_per_m:.3f}/M token")
    print()
print("说明: 实测是 1.5B 小模型单卡满载吞吐, 大模型成本更高(吞吐低)。")
print("两框架成本差异≈吞吐差异(8%), 高复用场景 SGLang 成本可降一半(吞吐2.1x)。")
