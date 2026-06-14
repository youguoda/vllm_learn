from pathlib import Path
REQ = {
    "bench_results/vllm_fixed_output.csv": "场景A vLLM",
    "bench_results/sglang_fixed_output.csv": "场景A SGLang",
    "bench_results/vllm_latency.csv": "场景B vLLM",
    "bench_results/sglang_latency.csv": "场景B SGLang",
    "bench_results/vllm_multi_turn.csv": "场景C vLLM",
    "bench_results/sglang_multi_turn.csv": "场景C SGLang",
    "bench_results/vllm_streaming.csv": "场景D vLLM",
    "bench_results/sglang_streaming.csv": "场景D SGLang",
    "bench_results/all_data.json": "合并数据集",
    "assets/p1_throughput.png": "吞吐图",
    "assets/p2_ttft_vs_conc.png": "TTFT-并发图",
    "assets/p3_ttft_vs_input.png": "TTFT-输入图",
    "assets/p4_multi_turn.png": "多轮图",
    "assets/p5_gpu_utilization.png": "GPU利用率图",
    "assets/p6_cost_comparison.png": "成本图",
    "assets/pagedattention_arch.png": "PagedAttention架构",
    "assets/radixattention_arch.png": "RadixAttention架构",
    "assets/kvcache_diff.png": "KVCache差异",
    "assets/prefix_ratio_comparison.png": "复用率图",
    "assets/scheduler_compare.png": "调度对比",
    "assets/parallelism_table.png": "并行表",
    "assets/selection_guide.png": "选型指南",
    "学习日志/实验报告.md": "实验报告",
    "学习日志/选型结论.md": "选型结论",
    "学习日志/社区数据交叉验证-Week7.md": "社区验证",
}
print("=== M3 交付前完整性核查 ===")
missing = []
for path, desc in REQ.items():
    p = Path(path)
    if p.exists():
        print(f"  OK {desc}: {p.stat().st_size//1024}KB")
    else:
        print(f"  缺少 {desc}: {path}"); missing.append(path)
print(f"\n{'全部就绪!' if not missing else f'缺 {len(missing)} 个'}")
