# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM learning/exploration project. Goal: build deep technical understanding of LLM inference frameworks (vLLM + SGLang) for a July 2026 tech talk. Includes hands-on experiments, paper notes, and daily learning logs.

## Environment

- **GPU**: NVIDIA RTX 3060 12GB (WSL2)
- **CUDA**: Driver 13.1, Toolkit 12.4
- **Python**: 3.12, vLLM 0.22.1

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
```

## WSL2 Caveats

- vLLM forces `spawn` multiprocessing — all scripts MUST wrap code in `if __name__ == "__main__"`
- HTTP proxy (`http_proxy`) intercepts localhost — use `NO_PROXY="*"` with curl and python
- `vllm serve` startup takes ~60-90s (model load + CUDA graph capture), health check blocks until ready

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
- `论文笔记-PagedAttention-vLLM.md` — PagedAttention paper notes (vLLM's core innovation)
- `SGLang与vLLM推理框架对比分享.md` — Tech talk outline: SGLang vs vLLM comparison (July 2026)
- `学习日志/` — Daily learning logs (04-22 through 06-09, covering both SGLang and vLLM)
- `AGENTS.md` — Stale; still says "currently empty", needs updating

## References

- vLLM source: https://github.com/vllm-project/vllm
- vLLM docs: https://docs.vllm.ai
- PagedAttention paper: https://arxiv.org/abs/2309.06180
