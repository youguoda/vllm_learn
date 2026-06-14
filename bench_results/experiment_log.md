# 实验记录日志（Week 6）

## 环境快照
- vLLM: 0.22.1（基础环境 /home/guoda/python，端口 8000）
- SGLang: 0.5.13（venv-sglang，端口 30000）
- GPU: RTX 3060 12GB (WSL2)
- 对齐参数: max_model_len=4096, 显存比例=0.85

## 启动命令
### vLLM
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000
### SGLang
source ~/venv-sglang/bin/activate
NO_PROXY="*" HF_HUB_OFFLINE=1 python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B --context-length 4096 --mem-fraction-static 0.85 --port 30000

## 实验执行记录
| 日期 | 框架 | 场景 | 结果文件 |
|---|---|---|---|
| 06-14 | vllm | fixed_output | bench_results/vllm_fixed_output.csv |
| 06-14 | sglang | fixed_output | bench_results/sglang_fixed_output.csv |
