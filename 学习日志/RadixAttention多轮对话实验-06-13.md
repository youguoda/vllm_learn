---
title: RadixAttention 多轮对话实验：缓存粒度才是 SGLang 的本质优势
date: 2026-06-13
tags:
  - R/技术框架/SGLang
  - R/技术框架/vLLM
  - R/核心算法/LLM推理
  - R/核心算法/推理优化
  - output/active
related:
  - "[[SGLang论文-RadixAttention精读-06-13]]"
  - "[[SGLang安装笔记]]"
  - "[[vLLM-PrefixCaching实测-06-13]]"
---

# RadixAttention 多轮对话实验

> 完成 05-02 计划。在 SGLang 上实测多轮对话的前缀复用，并与 vLLM 做"半块对齐前缀"对照——拿到两框架缓存机制差异最直接的实验证据。
>
> **一句话结论**：RadixAttention 的本质优势不是更快的 kernel，而是**更细的缓存粒度**（page_size=1 vs vLLM block_size=16）。

环境：SGLang 0.5.13 @ `~/venv-sglang`，Qwen2.5-1.5B，RTX 3060 12GB，context=4096，mem-fraction=0.8。

---

## 步骤 1+2：多轮对话 + 前缀长度变体

脚本 [07_radix_attention.py](../07_radix_attention.py)。场景：同一 system prompt，连续 5 轮追问，每轮携带全部历史 → 前缀逐轮变长。三档 system prompt 长度对照。

### 每轮 TTFT（2000 token 档为例）

| 轮次 | TTFT | 历史长度(字) | 说明 |
|---:|---:|---:|---|
| 1 | 110.1ms | 2320 | 冷，要 prefill 整个 system prompt |
| 2 | 29.2ms | 2575 | **骤降**，历史进树，只算新增 |
| 3 | 27.7ms | 2806 | 稳定 |
| 4 | 30.9ms | 3023 | 稳定 |
| 5 | 31.2ms | 3245 | 稳定 |

### 三档前缀长度汇总（warmup 后干净数据）

| 前缀≈token（实测） | 轮1 TTFT | 轮2-5 均值 | 降幅 |
|---:|---:|---:|---:|
| 352 | 59.3ms | 28.2ms | 52.3% |
| 726 | 72.7ms | 29.3ms | 59.7% |
| 1452 | 110.1ms | 29.7ms | **73.0%** |

**规律验证**：前缀越长 → 轮1越慢（59→73→110ms）→ 降幅越大（52%→60%→73%）。命中省的就是前缀的 prefill，前缀越长省得越多。轮2-5 一律稳定在 ~29ms。

### 命中的铁证（SGLang 日志 #cached-token）

2000 档对话的 Prefill batch 日志：

```
轮1: #new-token: 741,  #cached-token: 729    (system prompt 大部分命中)
轮2: #new-token: 16,   #cached-token: 1570   ← 只算 16 个新 token!
轮3: #new-token: 16,   #cached-token: 1686
轮4: #new-token: 17,   #cached-token: 1802
轮5: #new-token: 19,   #cached-token: 1919
```

> 这就是 RadixAttention 多轮对话的核心：**历史（system prompt + 前几轮问答）全在 radix 树上，每轮只 prefill 新增的十几个 token**（上一轮回答的尾巴 + 新问题）。cached-token 随轮数稳步增长（1570→1919），新 token 始终只有 16-19 个。

---

## 步骤 3：关键对照 —— 任意长度前缀命中（vs vLLM）

用**完全相同**的 BASE 前缀和截断逻辑（`exp_partial_prefix_sglang.py` vs 昨天的 `exp_partial_prefix.py`），构造"半块对齐"场景：请求 B 把共享前缀截断到**非 16-token 块边界**处。

### 半块对齐前缀对照表（PPT 必用）

| 框架 | 机制 | B 命中 token | 命中率 | 要重算 | B 请求 TTFT |
|---|---|---:|---:|---:|---:|
| **vLLM** | APC 块哈希 (block_size=16) | 96/117 | 82.1% | 21 | 32.1ms |
| **SGLang** | RadixCache (page_size=1) | **111/117** | **94.9%** | **6** | 28.3ms |

> 注：B 请求共 117 token，与 A 共享前半段（截断点在 178 字符处，故意不对齐块边界）。

### 差异分析

```
SGLang 多命中 = 111 - 96 = 15 token
              ≈ 一整个 block_size(16) 的损失

vLLM:   截断点不在 16-token 块边界
        → 共享部分只能命中到【最后一个完整对齐块】
        → 尾部那个不满块 (~15 token) 无法复用, 必须重算
        → 命中 96/117 = 82.1%

SGLang: page_size=1, radix 树可在【任意 token 位置 split】
        → 共享前缀逐 token 精确匹配, 几乎全部命中
        → 命中 111/117 = 94.9%
```

这正好实证了 [[SGLang论文-RadixAttention精读-06-13]] 的核心论点，也是 [[vLLM-PrefixCaching实测-06-13]] 里"B 命中率只有 82.1%"的另一面：**同样的 B 请求，SGLang 多救回了 15 个被 vLLM 块边界浪费掉的 token**。

---

## 步骤 4：结论

### 一句话

> **RadixAttention 的本质优势不是更快的 kernel，而是更细的缓存粒度。**

- vLLM block_size=16：缓存以 16-token 块为单位，**非对齐的共享前缀尾部最多浪费一整块**。
- SGLang page_size=1：radix 树以 token 为单位，**任意位置可 split，共享前缀逐 token 精确复用**。

### 为什么这在真实场景很重要

多轮对话、RAG、few-shot 这些场景，**共享前缀往往不会恰好落在 16-token 边界上**（用户输入长度随机）。block 粒度每次都会在边界损失最多一块；token 粒度则能榨干每一个可复用的 token。前缀越长、轮次越多，这个差距累积得越明显——这正是 SGLang 论文宣称多轮/RAG 场景最高 6.4× 提升的微观来源。

### 代价（不能只说好处）

SGLang 的 token 级 radix 树也有代价：树的维护开销、并发请求的锁竞争、引用计数管理。block 粒度更粗但**数据结构更简单、并发更友好**。这是工程权衡，不是单方面碾压。

---

## 今日产出

- [x] **07_radix_attention.py** + 三档前缀数据表（352/726/1452 token，降幅 52%/60%/73%）
- [x] **半块前缀对照表**（SGLang 94.9% vs vLLM 82.1%，多命中 15 token = 一个 block 的损失）
- [x] 多轮对话命中铁证（每轮只 prefill 16-19 新 token，cached 1570→1919）

## 复现命令

```bash
# 启动 SGLang
source ~/venv-sglang/bin/activate
NO_PROXY="*" HF_HUB_OFFLINE=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B --context-length 4096 \
  --mem-fraction-static 0.8 --port 30000

# 多轮对话实验
NO_PROXY="*" python3 07_radix_attention.py

# 半块对齐前缀对照 (先 flush 保证 A 真冷)
NO_PROXY="*" curl --noproxy '*' -s -X POST http://127.0.0.1:30000/flush_cache
NO_PROXY="*" python3 exp_partial_prefix_sglang.py
# 命中证据: grep 'Prefill batch' <sglang日志> | grep cached-token
```
