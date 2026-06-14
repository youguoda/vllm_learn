"""按对外讲的场景重组数据，提炼 PPT 论点骨架"""
import json
D = json.load(open("bench_results/all_data.json"))


def fo(fw, key, conc):
    for r in D.get("fixed_output", {}).get(fw, []):
        if int(r["concurrency"]) == conc:
            return r.get(key, 0)
    return 0


print("=" * 60)
print("场景1：通用批量推理（无共享前缀）")
print("=" * 60)
for fw in ("vllm", "sglang"):
    best = max(r["tok_per_s"] for r in D["fixed_output"][fw])
    print(f"  {fw}: 最高吞吐={best:.0f} tok/s, 单并发TTFT={fo(fw,'ttft_median_ms',1):.0f}ms")
print(f"  → 论点: 两者持平(差{(1443-1342)/1342*100:.0f}%)，vLLM 略优+生态成熟")

print("\n" + "=" * 60)
print("场景2：高前缀复用（统一 system prompt / RAG / Agent）")
print("=" * 60)
print("  vLLM RPS=8.88 vs SGLang RPS=18.95 (100%复用率)")
print("  → 论点: SGLang RadixAttention 吞吐 2.1x")

print("\n" + "=" * 60)
print("场景3：延迟敏感（交互式）")
print("=" * 60)
for fw in ("vllm", "sglang"):
    print(f"  {fw}: 单并发TTFT={fo(fw,'ttft_median_ms',1):.0f}ms")
print("  → 论点: 延迟持平，看吞吐和复用率决定")

print("\n" + "=" * 60)
print("PPT 三句话骨架:")
print("  1. KV Cache 管理(块 vs token)是两框架最大差异")
print("  2. 复用率是分水岭: 无复用持平, 高复用 SGLang 2.1x")
print("  3. 选型: 通用→vLLM(生态), 高复用→SGLang(性能)")
