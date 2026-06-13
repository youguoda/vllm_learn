---
title: 投机解码实验（ngram）—— 重复内容 4.5x 加速
date: 2026-06-14
tags:
  - R/技术框架/vLLM
  - R/核心算法/投机解码
  - output/active
related:
  - "[[投机解码原理-06-14]]"
  - "[[投机解码框架对比-06-14]]"
---

# 投机解码实验（ngram）

> 实测 vLLM 的 ngram 投机解码——无需额外模型，从上下文找重复片段当 draft。对比普通 decode，看不同内容类型的加速差异。

环境：Qwen2.5-1.5B，RTX 3060 12GB，vLLM 0.22.1，单请求，max_tokens=200，ignore_eos。

---

## 实验配置

```bash
# 普通
vllm serve Qwen/Qwen2.5-1.5B --max-model-len 4096 --gpu-memory-utilization 0.85
# ngram 投机
vllm serve ... --speculative-config \
  '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
```

vLLM 接受配置：`speculative_config=SpeculativeConfig(method='ngram', num_spec_tokens=5)`。脚本 [11_speculative_decode.py](../11_speculative_decode.py)。

---

## 实测数据

| Prompt | 普通 tok/s | ngram tok/s | 加速比 |
|---|---:|---:|---:|
| Python 冒泡排序 | 89.6 | 78.7 | 0.88x ⚠️ |
| 工业传感器 10 类型 | 90.5 | 120.7 | 1.33x |
| 5 种轴承故障（固定格式） | 90.2 | **399.5** | **4.43x** |
| JSON 配置 20 节点 | 90.8 | **405.5** | **4.46x** |
| 科幻故事 | 90.3 | 82.3 | 0.91x ⚠️ |
| AI 影响分析 | 90.1 | 390.9 | 4.34x |
| 工业事故调查 | 90.8 | 107.3 | 1.18x |
| 质检工作日记 | 90.5 | 391.6 | 4.33x |

**汇总**：普通 decode 全部 ~90 tok/s（单请求访存瓶颈）；ngram 投机**高度重复内容飙到 ~400 tok/s（4.5x）**，自由内容 78-107 tok/s（基本无效甚至略慢）。

---

## 核心发现

### 1. 加速比与内容重复度强相关（高达 4.5x）

最快的两个 prompt——**JSON 配置（405 tok/s）和固定格式轴承故障（399 tok/s）**——都生成大量重复结构（`{"id":..,"type":..}` 反复出现、`故障名称：\n产生原因：\n...` 模板重复）。ngram 从已生成部分找到这些重复 pattern，一次猜中 5 个 token，接受率极高 → 4.5x 加速。

### 2. 自由创作几乎无效甚至略慢（0.88-0.91x）

冒泡排序代码（0.88x）和科幻故事（0.91x）**比普通还慢**——这些内容每个 token 都新，ngram 命中率极低，但每步还要**付验证开销**（多算 K+1 个位置），收益为负。这正好印证 05-19 的预测："接受率太低时验证开销抵消收益"。

### 3. 一个诚实的观察

温度=0 + ignore_eos 让一些"低重复"prompt（AI 分析、工作日记）也产生了重复填充内容，导致它们也被加速。**真实自由生成（不强制 ignore_eos、温度>0）的加速会更接近 1.0x**。这是实验设计的副作用，要诚实标注。

---

## 与官方 EAGLE 数据对比

ngram 是"零成本"投机（无需训练/额外模型），但只对重复内容有效。EAGLE 系列普适性更强（需训练 draft head）：

| 方案 | 接受率 | 加速比 | 来源 |
|---|---:|---:|---|
| **我的 ngram（重复内容）** | 高(估>0.8) | **4.5x** | 本机实测 |
| 我的 ngram（自由内容） | 低(<0.2) | ~0.9x | 本机实测 |
| vLLM ngram（官方回归测试） | match_rate 0.78 | — | vLLM CI |
| EAGLE（Llama-3.1-8B） | 0.681 | 1.89x | vLLM 回归测试 |
| EAGLE3（Llama-3.1-8B） | 0.742 | 2.15x | vLLM 回归测试 |
| EAGLE3（Red Hat 集成） | — | 最高 2.5x | Red Hat 2025 |

> **对比解读**：我的 ngram 在重复内容上 4.5x 看似比 EAGLE 的 2x 高，但**不可直接比**——ngram 只在高重复内容有效（自由内容 0.9x），EAGLE 是**全场景稳定 2-2.5x**（接受率 0.68-0.74）。EAGLE 用训练好的 draft head 复用大模型特征，普适性强；ngram 零成本但挑场景。
>
> 注意：EAGLE 加速比"高度依赖请求率和并发"——同步/低并发收益最大，高并发时验证额外 token 的开销会吃掉收益（这也是 vLLM 不支持 tree decoding 的原因）。

---

## 今日产出

- [x] 11_speculative_decode.py
- [x] result_normal.txt + result_spec.txt
- [x] 对比表（8 prompt × 两模式，含加速比）
- [x] EAGLE 官方数据引用（vLLM 回归测试 / Red Hat，注明模型）

## 一句话

> **ngram 投机零成本但挑内容**：重复结构（JSON/模板/列表）4.5x 加速，自由创作 ~0.9x（验证开销倒拖）。EAGLE 系列需训练但全场景稳定 2-2.5x。**选型**：输出格式化/重复多（代码生成、数据抽取、报表）→ ngram 白嫖；通用对话 → EAGLE（如有 draft head）或不开。

## 来源

- [vLLM 投机解码回归测试 #28135](https://github.com/vllm-project/vllm/issues/28135)
- [Red Hat: EAGLE3 in vLLM (2025)](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding)
- [Speculative Decoding 2-3x 指南](https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025)
