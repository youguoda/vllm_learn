---
title: 实测页：vllm bench serve 在 1/8/32 并发下的吞吐与 TTFT
date: 2026-06-13
tags:
  - R/技术框架/vLLM
  - R/核心算法/LLM推理
  - output/active
related:
  - "[[vLLM-Continuous-Batching实验-06-11]]"
---

# 实测页：vllm bench serve 在 1/8/32 并发下的吞吐与 TTFT

> 用 vLLM **官方** benchmark 工具 `vllm bench serve` 在自己的卡上跑出来的真实数字。自己跑的数据最有说服力——这一页可直接放进分享 PPT。

## 测试条件

| 项 | 值 |
|---|---|
| GPU | RTX 3060 12GB (WSL2) |
| 模型 | Qwen/Qwen2.5-1.5B (bf16) |
| Server | `--max-model-len 4096 --gpu-memory-utilization 0.85`（默认开 APC + chunked prefill） |
| 数据集 | `random`，输入 **512** token / 输出 **128** token（固定，可复现） |
| 并发档位 | `--max-concurrency` = 1 / 8 / 32 |
| 请求数 | 16 / 64 / 256（≈ 并发 × 8，保证测量稳定） |
| seed | 42 |

复现命令：

```bash
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B \
  --max-model-len 4096 --gpu-memory-utilization 0.85

HF_HUB_OFFLINE=1 NO_PROXY="*" vllm bench serve \
  --backend openai --model Qwen/Qwen2.5-1.5B \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --max-concurrency 8 --num-prompts 64 --seed 42 \
  --save-result --result-dir bench_results --result-filename bench_c8.json
```

---

## 核心结果表

| 并发 | 输出吞吐<br>(tok/s) | 总吞吐<br>(tok/s) | RPS<br>(req/s) | Mean TTFT<br>(ms) | P99 TTFT<br>(ms) | Mean TPOT<br>(ms/tok) |
|---:|---:|---:|---:|---:|---:|---:|
| **1**  | 88.85   | 444   | 0.69 | **94.5**  | 271   | 10.6 |
| **8**  | 530.52  | 2,653 | 4.14 | **247.3** | 465   | 13.2 |
| **32** | 1114.79 | 5,574 | 8.71 | **568.7** | 1,685 | 24.4 |

> - **吞吐 (throughput)** = 每秒生成多少 token，越高越好。
> - **TTFT (Time To First Token)** = 首 token 延迟，越低越好。
> - **TPOT (Time Per Output Token)** = 出第一个 token 后，每个后续 token 的平均间隔（= decode 速度）。
> - **P99** = 99% 的请求都比这个值快（衡量"最差体验"）。

---

## 一图看懂：吞吐 vs 延迟的权衡

### 输出吞吐量（并发越高，吞吐越高）

```
tok/s
1200 |                                        ┌──────┐
     |                                        │ 1115 │ c=32
1000 |                                        └──────┘
 800 |
 600 |                      ┌──────┐
     |                      │ 531  │ c=8
 400 |                      └──────┘
 200 |  ┌──────┐
     |  │  89  │ c=1
   0 +──┴──────┴──────────┴──────┴──────────┴──────┴──
        并发1               并发8              并发32

  并发 1→8 :  吞吐 ×6.0   (88.85 → 530.52)
  并发 8→32:  吞吐 ×2.1   (530.52 → 1114.79)
  并发 1→32:  吞吐 ×12.5  ← Continuous Batching 把闲置算力喂饱
```

### 首 token 延迟 TTFT（并发越高，延迟越大）

```
ms
1800 |                                     P99 ┌──────┐
     |                                         │ 1685 │ c=32
1400 |                                         │      │
     |                                         │      │
1000 |                                         │      │
     |                              P99 ┌─────┐│ mean │
 600 |                                  │ 465 ││ 569  │
     |              P99 ┌─────┐  mean   │     ││      │
 200 |  P99┌────┐ mean  │     │  │247 │  │     ││      │
     |  │271│ 94 │      │     │  └─────┘  └─────┘└──────┘
   0 +──┴────┴───┴──────┴─────┴──────────────────────────
          并发1            并发8              并发32

  Mean TTFT:  94.5ms → 247.3ms → 568.7ms   (并发越高，排队越久)
  P99  TTFT: 271ms  → 465ms   → 1685ms      (尾延迟恶化更快)
```

### 解码速度 TPOT（基本稳定，略升）

```
ms/tok
 30 |                                   ┌──────┐
    |                                   │ 24.4 │ c=32
 20 |                                   └──────┘
    |                ┌──────┐
 10 |  ┌──────┐      │ 13.2 │ c=8
    |  │ 10.6 │      └──────┘
    |  └──────┘ c=1
  0 +──┴──────┴──────┴──────┴───────────┴──────┴──
       并发1           并发8              并发32

  出字速度从 10.6 → 24.4 ms/token，仍然流畅（>40 tok/s）
```

---

## 怎么读这三张图（结论）

1. **吞吐随并发大涨，但收益递减**
   并发 1→8 吞吐涨 6 倍，8→32 只再涨 2.1 倍。因为 GPU 算力在高并发时逐渐被喂满，接近 RTX 3060 跑 1.5B 的上限（~1100 tok/s）。

2. **TTFT 是高吞吐的代价**
   并发 32 时 Mean TTFT 升到 569ms、**P99 飙到 1685ms**。请求多了要排队等 prefill，尾延迟恶化比均值更剧烈。

3. **TPOT（出字速度）相对稳**
   即使并发 32，每 token 也才 24ms（约 41 tok/s），用户读起来仍然流畅。**吞吐的提升主要靠"同时服务更多人"，而非牺牲单人的阅读流畅度。**

4. **怎么选并发？**
   - 在意**单用户体验**（低延迟）→ 并发压在 8 以内，TTFT < 250ms。
   - 在意**整体成本/吞吐**（多用户）→ 并发拉到 32，单卡吞吐 ×12.5，但要接受 P99 TTFT 上秒。
   - 这正是 [[vLLM-Continuous-Batching实验-06-11]] 里"吞吐 vs 延迟权衡"的官方工具复现版。

---

## 附：原始数据

JSON 结果存于 `bench_results/bench_c{1,8,32}.json`。关键字段：

| 字段 | c=1 | c=8 | c=32 |
|---|---:|---:|---:|
| duration (s) | 23.05 | 15.44 | 29.39 |
| request_throughput | 0.69 | 4.14 | 8.71 |
| output_throughput | 88.85 | 530.52 | 1114.79 |
| total_token_throughput | 444.26 | 2652.58 | 5573.93 |
| mean_ttft_ms | 94.48 | 247.29 | 568.72 |
| median_ttft_ms | 80.34 | 290.88 | 585.62 |
| p99_ttft_ms | 271.17 | 464.84 | 1684.85 |
| mean_tpot_ms | 10.60 | 13.22 | 24.38 |
| p99_tpot_ms | 10.64 | 15.06 | 31.01 |
