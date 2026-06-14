---
title: 社区数据交叉验证（Week 7）—— 验证报告结论
date: 2026-06-14
tags:
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - output/active
related:
  - "[[实验报告]]"
  - "[[社区数据交叉验证]]"
  - "[[吞吐量实验-06-14]]"
---

# 社区数据交叉验证（Week 7）

> 针对报告里的核心结论，再找社区数据做最终验证（含主动找矛盾证据）。承接 05-12 的初步验证，聚焦 Week 6 新增的实验结论。

---

## 验证结论1：无共享前缀两框架持平（我的核心发现）

**我的数据**：无前缀固定输出，vLLM 1443 vs SGLang 1342 tok/s（差 8%，持平）。

**社区验证**（2026 多源）：
> "On unique-prompt workloads with no prefix reuse, the two engines are **roughly at parity**."
> "one benchmark used unique prompts throughout, so you are seeing the **baseline behavior**."

→ **完全一致** ✓。社区明确：unique prompts 时两框架持平，这正是我无前缀实验的结论。

---

## 验证结论2：SGLang 优势只在共享前缀场景（复用率分水岭）

**我的数据**：100% 复用率 SGLang 2.1x；0-75% 持平。

**社区验证**：
> "SGLang's main throughput advantage comes from RadixAttention, which **only helps when requests share content**."
> "29% higher throughput on H100s (ShareGPT) and **up to 6.4x on prefix-heavy** workloads."

→ **一致** ✓。我的 2.1x（100% 极端复用）落在社区 29%（混合）~6.4x（极端前缀重）区间。**复用率决定选型**这一核心论点被反复印证。

---

## 验证结论3：RadixAttention 的内存代价（我没测到的补充）

**社区补充**（我的实验未覆盖）：
> "The tradeoff is memory. RadixAttention's LRU cache consumes GPU memory... When GPU memory is tight and prefix overlap is low, **the cache overhead works against you**."

→ 这解释了我 05-29 观察到的"无前缀时 vLLM 略快 8%"——**SGLang 的 radix 树维护在无复用时是纯开销**。补全了我的认知：不只是"持平"，低复用时 SGLang 还略有缓存开销劣势。

---

## 验证结论4：vLLM V1 引擎改进（印证我的源码发现）

**我的源码发现**（05-13）：vLLM v1 调度"无 prefill/decode 之分，统一 num_computed_tokens"。

**社区验证**：
> "vLLM shipped a major **V1 engine rewrite that removes the distinction between prefill and decode phases**, treating prompt and generated tokens uniformly... lower scheduling overhead at high concurrency."

→ **完全一致** ✓。我读源码得出的"统一抽象"正是 V1 重写的核心，且社区指出这降低了高并发调度开销——解释了为什么我的无前缀高并发 vLLM 略优。

---

## 主动找矛盾证据

| 我搜的"反方"观点 | 实际结论 |
|---|---|
| "SGLang 不比 vLLM 快" | 成立但**仅限无复用**——和我结论一致（unique prompts 持平） |
| "vLLM 多轮对话更好" | 没找到；社区也认同 vLLM APC 缓存历史，与我 4.4 节认知修正一致 |
| "2.1x 普遍成立" | **被否定**——社区强调倍数高度依赖复用率，ShareGPT 仅 29% |

> **无强矛盾证据**。所有"反方"观点细查后都是"特定条件下成立"，且条件与我的实验场景对应。我的结论稳健。

---

## 交叉验证总表

| 我的结论 | 社区数据 | 一致性 |
|---|---|---|
| 无前缀持平(差8%) | unique prompts 持平 | ✓ 完全一致 |
| 100%复用 2.1x | 29%~6.4x 区间 | ✓ 落在区间 |
| 多轮 vLLM 不差 | APC 缓存历史 | ✓ 一致 |
| 复用率决定选型 | "only helps when share content" | ✓ 核心印证 |
| V1 统一调度 | V1 rewrite 移除 prefill/decode 之分 | ✓ 源码印证 |

---

## 今日产出

- [x] 社区数据交叉验证-Week7.md（5 条结论验证 + 矛盾证据排查）
- [x] 所有核心结论获社区数据支撑，无强矛盾

## 来源

- [vLLM vs SGLang 2026 (Morph)](https://www.morphllm.com/comparisons/vllm-vs-sglang)
- [H100 Benchmarks (Spheron)](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
- [SGLang 高并发 benchmark #21061](https://github.com/sgl-project/sglang/issues/21061)
- [Picking an engine (Jarvislabs)](https://jarvislabs.ai/blog/vllm-sglang-trtllm-comparison)

## 一句话

> **报告所有核心结论都获社区数据印证**：无前缀持平、高复用 SGLang 占优、复用率决定选型、V1 统一调度。主动找矛盾证据无果——"反方"观点细查后都与我的场景结论一致。结论稳健可对外分享。
