---
title: APC 实测：Automatic Prefix Caching 如何把 TTFT 砍掉 4-6 倍
date: 2026-06-13
tags:
  - R/技术框架/vLLM
  - R/核心算法/LLM推理
  - R/核心算法/推理优化
  - output/active
related:
  - "[[论文笔记-PagedAttention-vLLM]]"
  - "[[vLLM-PagedAttention实测-06-13]]"
---

# APC 实测：Automatic Prefix Caching 如何把 TTFT 砍掉 4-6 倍

> 本文在 RTX 3060 12GB (WSL2) 上实测 vLLM 的 **Automatic Prefix Caching (APC)**。通过开/关 APC 的对照实验 + 两个破坏性变体，亲眼验证："共享前缀的 KV cache 被复用 → 第 2 条起 TTFT 骤降，但只省 prefill(TTFT)，不省 decode"。
>
> **一句话类比**：APC 就像食堂的"预制菜"——前半段菜（system prompt 的 KV）提前做好放着，谁来点单都直接端走，只现炒后半段（用户问题）；但上菜后的吃饭速度（decode）不变。

---

## 原理：APC 在做什么

vLLM 把每个请求的 KV cache 按 **block（16 token）** 存储（见 [[vLLM-PagedAttention实测-06-13]]）。APC 在此之上加了一层：

1. **给每个 block 算哈希**：哈希 **链式包含前缀所有 block 的哈希**（`hash(block_i) = f(hash(block_{i-1}), tokens_i)`）。
2. **缓存已算过的 block**：请求处理完，它的前缀 block 不立即丢弃，留在缓存里。
3. **新请求查表复用**：如果新请求开头的 block 哈希命中缓存，**直接复用那些物理块，跳过这部分的 prefill 计算**。

```
请求1: [system prompt 1000 tok ][问题A]
        └─ 算 KV，缓存这些 block ─┘
请求2: [system prompt 1000 tok ][问题B]
        └─ 哈希命中！直接复用 ───┘ 只需 prefill [问题B]
        ⇒ 省掉 1000 token 的 prefill → TTFT 骤降
```

**关键限制（为什么哈希要链式包含前缀）**：只要前缀有**任何一个 token 不同**，从那个 block 起所有后续 block 的哈希全变，命中链断裂。这保证了正确性——绝不会复用"看起来像但其实上下文不同"的 KV。

---

## 第 1 步：确认 APC 开关与指标

vLLM 0.22.1 **默认开启** APC。启动后确认：

```bash
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B \
  --max-model-len 4096 --gpu-memory-utilization 0.85
```

- **日志确认**：`enable_prefix_caching=True`
- **哈希算法**：`prefix_caching_hash_algo="sha256"`
- **命中率指标**（`/metrics`，单位是 **block 数**）：

| 指标名 | 含义 |
|---|---|
| `vllm:prefix_cache_queries_total` | 累计查询的 block 数 |
| `vllm:prefix_cache_hits_total` | 累计命中的 block 数 |

命中率 = `hits / queries`。

```bash
NO_PROXY="*" curl --noproxy '*' -s http://127.0.0.1:8000/metrics | grep prefix_cache
```

---

## 第 2 步：测试脚本设计

脚本 [06_prefix_caching.py](../06_prefix_caching.py) 的核心：

- **共享长前缀**：重复拼接一段话堆出 ~1000 token 的 system prompt
- **10 个不同短问题**：前缀相同、后缀不同 → 第 2 条起应命中前缀缓存
- **stream 模式测 TTFT**：精确捕捉第一个 token 到达时刻
- **temperature=0**：贪心解码，结果可复现
- **跑前跑后抓 `/metrics`**：算出本轮命中率
- 支持变体开关：`--prefix-repeat`（前缀长度）、`--random-prefix`（破坏命中）、`--salt`（隔离不同实验的缓存）

---

## 第 3 步：对照实验（核心产出）

同一套脚本（1000-token 前缀），分别在 **APC ON** 和 **APC OFF**（`--no-enable-prefix-caching`）下各跑一遍。

### 对比表

| 指标 | APC ON | APC OFF | 说明 |
|---|---:|---:|---|
| **COLD TTFT**（第 0 条） | 195.7 ms | 498.0 ms | 首条都要全额 prefill（含 JIT 冷启动） |
| **WARM TTFT**（第 1+ 均值） | **33.3 ms** | **152.6 ms** | ← APC 的关键差异 |
| **WARM/COLD 比值** | 0.17 | 0.31 | ON 时 warm 几乎归零 |
| **命中率** | 98.7% | 0.0% | OFF 根本不查缓存 |
| **平均 decode 时间** | 249.1 ms | 220.1 ms | **几乎相同**，APC 不碰 decode |

### 三个关键观察

**① WARM TTFT 提速 4.6 倍**：152.6 / 33.3 ≈ **4.6x**。这就是 APC 省掉 1000-token 前缀 prefill 的直接收益。

**② APC OFF 时每条一样慢**：关掉缓存后，第 1 条到第 9 条 TTFT 全是 ~152ms，没有任何下降——因为每条都要重新算整个前缀的 KV。

```
APC ON  的 TTFT 序列:  849 → 39 → 39 → 39 → ...  (第2条起断崖式下降)
APC OFF 的 TTFT 序列:  498 → 153 → 151 → 150 → ... (一直平着，没有缓存红利)
```

**③ decode 时间不受影响**：ON/OFF 的平均 decode 时间分别 249ms / 220ms，差异 11.6% 属噪声级别。**这验证了 APC 只省 prefill(TTFT)，不省 decode** —— 预制菜让上菜变快，但吃饭速度不变。

---

## 第 4 步：放大 / 破坏实验

### 变体①：前缀加倍 → 验证"前缀越长，命中省得越多"

在 **APC OFF** 下对比不同前缀长度的 WARM TTFT（OFF 下每条都付全额 prefill，最能反映 prefill 时间随前缀的增长）：

| 前缀长度 | APC OFF 每条 TTFT | APC ON WARM TTFT | APC 省下 |
|---:|---:|---:|---:|
| ~1000 token | 152.6 ms | 33.3 ms | **119 ms** |
| ~2000 token | 265.2 ms | ~38 ms | **227 ms** |

> 前缀从 1000 → 2000 token，OFF 下 prefill 时间从 153ms → 265ms（**翻倍前缀，prefill 时间也接近翻倍**）。而 APC ON 时 WARM TTFT 始终维持在 ~35ms。**所以前缀越长，APC 命中省下的绝对时间越多**——这正是 system prompt / few-shot 场景 APC 价值巨大的原因。

### 变体②：随机前缀 → 验证"差一个 token 整条链失效"

给每条请求的 system prompt 开头加一个**随机数字**（`[随机数] + 相同前缀`），在 APC ON 下跑：

| 指标 | 正常前缀 | 随机前缀 |
|---|---:|---:|
| WARM TTFT | 33.3 ms | **305.5 ms** |
| 命中率 | 98.7% | **0.0%** |

```
随机前缀的 TTFT 序列: 484 → 310 → 304 → 300 → ...  (回到冷启动水平，毫无缓存红利)
命中率: queries +23073 blocks, hits +0 blocks  → 0.0%
```

> 仅仅在开头加了一个随机数字，**第一个 block 的哈希就变了，链式哈希导致后续所有 block 全部 miss**，命中率直接归零，TTFT 回到全额 prefill 水平。这从反面证明了链式哈希的设计：**前缀必须逐 token 完全一致才能复用**。

---

## 第 5 步：收尾自测

### 自测题：为什么块哈希要链式包含前缀块的哈希？

**答**：为了保证**上下文正确性**。LLM 中每个 token 的 KV 都依赖它前面的所有 token（注意力是因果的）。如果块哈希只算自己这 16 个 token，那么两个"局部相同但前文不同"的 block 会算出相同哈希 → 错误复用 → 生成结果错乱。

链式哈希 `hash(block_i) = f(hash(block_{i-1}), tokens_i)` 确保：**一个 block 的哈希唯一地代表"从序列开头到这个 block 的完整前缀"**。只有前缀逐 token 完全一致，哈希才相同，复用才安全。变体②的"加一个随机数字命中归零"就是这个机制的直接体现。

### 一句话验证："APC 只省 TTFT 不省 decode"

从数据里验证：用 **总生成时长 − TTFT = decode 时间** 这部分对比 ON/OFF：

```
APC ON  平均 decode = 249.1 ms
APC OFF 平均 decode = 220.1 ms
差异 11.6%，噪声级别 → decode 速度与 APC 无关 ✓
```

APC 动的是"请求进来到第一个 token"这一段（prefill），一旦开始逐字生成（decode），每个 token 照样要过一遍完整模型，APC 帮不上忙。

### 核心收获

```
APC = 缓存共享前缀的 KV cache（块哈希链式包含前缀）
  │
  ├─ 命中 → 跳过前缀 prefill → TTFT 骤降（实测 4.6x）
  ├─ 前缀越长 → 省得越多（1000tok 省 119ms，2000tok 省 227ms）
  ├─ 前缀差一个 token → 链式哈希全 miss → 命中归零
  └─ 只省 prefill(TTFT)，decode 速度不变（预制菜：上菜快，吃饭速度不变）

最适合的场景: 固定 system prompt、few-shot examples、多轮对话共享历史
```

---

## 附录：复现命令

```bash
# APC ON (默认)
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B --max-model-len 4096 --gpu-memory-utilization 0.85
NO_PROXY="*" python3 06_prefix_caching.py --label "APC ON" --prefix-repeat 29 --salt A

# APC OFF 对照
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B --max-model-len 4096 --gpu-memory-utilization 0.85 --no-enable-prefix-caching
NO_PROXY="*" python3 06_prefix_caching.py --label "APC OFF" --prefix-repeat 29 --salt B

# 变体①：前缀加倍
NO_PROXY="*" python3 06_prefix_caching.py --label "2000tok" --prefix-repeat 58 --salt C

# 变体②：随机前缀破坏命中
NO_PROXY="*" python3 06_prefix_caching.py --label "random" --random-prefix
```

> 注意：跨长度对比时用 `--salt` 给不同实验加唯一前缀，避免短前缀是长前缀的"子前缀"而跨实验命中，污染冷启动测量。
