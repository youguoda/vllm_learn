# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM learning/exploration project. Goal: build deep technical understanding of LLM inference frameworks (vLLM + SGLang) for a July 2026 tech talk. Includes hands-on experiments, paper notes, and daily learning logs.

## Environment

- **GPU**: NVIDIA RTX 3060 12GB (WSL2)
- **CUDA**: Driver 13.1, Toolkit 12.4
- **Python**: 3.12
- **vLLM 0.22.1** — base env `/home/guoda/python` (port 8000)
- **SGLang 0.5.13** — isolated venv `~/venv-sglang` (port 30000); `source ~/venv-sglang/bin/activate` first. Kept separate to avoid torch/flashinfer/kernel conflicts. Single 12GB card runs one at a time.

## Running Scripts

All scripts require `if __name__ == "__main__"` guard (WSL2 forces `spawn` multiprocessing).

```bash
# Offline inference
python3 01_basic_inference.py

# API server test (start server first, then run client)
NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B --max-model-len 4096 --gpu-memory-utilization 0.85
NO_PROXY="*" python3 02_api_server.py

# Parameter exploration benchmarks
python3 03_params.py

# Continuous batching benchmark (server must be running)
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B --max-model-len 4096 --gpu-memory-utilization 0.85
NO_PROXY="*" python3 04_continuous_batching.py --concurrency 1 5 10 20 --num-requests 40

# Official benchmark (vllm bench serve) — saves JSON to bench_results/
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm bench serve --backend openai --model Qwen/Qwen2.5-1.5B \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --max-concurrency 8 --num-prompts 64 --seed 42 \
  --save-result --result-dir bench_results --result-filename bench_c8.json
```

## WSL2 Caveats

- vLLM forces `spawn` multiprocessing — all scripts MUST wrap code in `if __name__ == "__main__"`
- HTTP proxy (`http_proxy`) intercepts localhost — use `NO_PROXY="*"` with curl and python
- `vllm serve` startup takes ~60-90s (model load + CUDA graph capture), health check blocks until ready
- Use `HF_HUB_OFFLINE=1` to skip HuggingFace network calls (avoids proxy timeouts when model is cached)
- After killing vLLM server, check for orphaned EngineCore processes: `ps aux | grep EngineCore | grep -v grep | awk '{print $2}' | xargs kill -9`

## Key Parameters (RTX 3060 12GB)

| Parameter | Effect | Recommended |
|---|---|---|
| `--max-model-len` | Max context length → KV cache size | `4096` (default 131072 OOMs) |
| `--gpu-memory-utilization` | Fraction of VRAM for vLLM | `0.85` (desktop uses ~2.7GB) |
| `--enforce-eager` | Skip CUDA Graphs + torch.compile | Faster startup, ~20% slower inference |
| `--tensor-parallel-size` | Multi-GPU tensor parallelism | `1` (single GPU) |

Model (Qwen2.5-1.5B): ~3 GiB VRAM. With `max_model_len=4096`: KV cache ~6 GiB, CUDA graphs ~0.5 GiB.

## Repo Structure

- `01_basic_inference.py` — Offline batch inference with `vllm.LLM`
- `02_api_server.py` — OpenAI-compatible API server test client
- `03_params.py` — Parameter exploration benchmarks
- `04_continuous_batching.py` — Continuous batching benchmark (concurrent API requests, measures RPS/TTFT/latency)
- `05_kvcache_blocks_scan.py` — PagedAttention block scan: starts server across (max_len, gpu_util) configs, scrapes `num_gpu_blocks` from `/metrics`, verifies 显存→block→并发 chain
- `06_prefix_caching.py` — APC (Automatic Prefix Caching) experiment: shared long system prompt + varied questions, measures TTFT cold vs warm, scrapes `vllm:prefix_cache_{queries,hits}_total` for hit rate. Flags: `--prefix-repeat`, `--random-prefix`, `--salt`
- `07_radix_attention.py` — SGLang multi-turn dialog experiment (port 30000): same system prompt + growing history, measures per-turn TTFT across 3 prefix lengths. Needs SGLang server running.
- `exp_partial_prefix.py` / `exp_partial_prefix_sglang.py` — half-block-aligned prefix experiment, vLLM (8000) vs SGLang (30000). Shows token-level (page_size=1) vs block-level (block_size=16) cache granularity. Read `#cached-token` from SGLang log for hit evidence.
- `08_structured_output.py` — SGLang JSON schema constrained decoding (port 30000): unconstrained vs schema (0%→100% valid), nested schema + enum, adversarial "write poetry" prompt. Uses OpenAI `response_format` json_schema. xgrammar backend compiles schema→FSM (~0.28s first call).
- `09_compare_bench.py` — framework-agnostic vLLM↔SGLang benchmark (OpenAI streaming API, `--base-url`/`--tag`). Outputs `bench_<tag>.csv`. Run one framework at a time (GPU exclusive). `plot_compare.py` renders `assets/compare_v1.png`. NOTE: tok/s not directly comparable (gen length differs); RPS is fairer.
- `verify_hash.py` — replicates vLLM chained block hash (`hash_block_tokens`) to demonstrate "change 1 token → all downstream block hashes invalidate". Pure CPU, no server needed. Companion to the M2 source-reading note.
- `tiny_radix.py` — minimal radix tree replicating SGLang's `_split_node`. Inserts 3 requests, shows two splits at arbitrary token positions (proving token-level vs vLLM's 16-token-block granularity). Pure CPU. Companion to the SGLang source-reading note.
- `draw_arch.py` — renders 3 PPT diagrams to `assets/`: pagedattention_arch.png, radixattention_arch.png, kvcache_diff.png. Arrows labelled with real source fn names; diff chart uses measured data. Chinese via WenQuanYi Zen Hei font. Run with base-env python (has matplotlib).
- `论文笔记-PagedAttention-vLLM.md` — PagedAttention paper notes (vLLM's core innovation)
- `SGLang与vLLM推理框架对比分享.md` — Tech talk outline: SGLang vs vLLM comparison (July 2026)
- `学习日志/` — Daily learning logs (04-22 through 06-09, covering both SGLang and vLLM)
- `AGENTS.md` — Stale; still says "currently empty", needs updating

## References

- vLLM source: https://github.com/vllm-project/vllm
- vLLM docs: https://docs.vllm.ai
- PagedAttention paper: https://arxiv.org/abs/2309.06180
