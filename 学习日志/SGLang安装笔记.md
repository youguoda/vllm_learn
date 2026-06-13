---
title: SGLang 安装笔记（WSL2 + RTX 3060，与 vLLM 环境隔离）
date: 2026-06-13
tags:
  - R/技术框架/SGLang
  - R/技术框架/vLLM
  - output/active
related:
  - "[[SGLang论文-RadixAttention精读-06-13]]"
  - "[[vLLM-PagedAttention实测-06-13]]"
---
# SGLang 安装笔记

> 完成 04-30 计划。在已装 vLLM 的机器上**用独立 venv 隔离**安装 SGLang，跑通首次推理，并与 vLLM 并排对照启动日志（KV 池、RadixCache、启动耗时）。

---

## 0. 环境背景

| 项                    | 值                                                                |
| --------------------- | ----------------------------------------------------------------- |
| GPU                   | RTX 3060 12GB (WSL2)                                              |
| **vLLM 装在哪** | **基础环境** `/home/guoda/python`（非 venv），vLLM 0.22.1 |
| 基础环境 torch        | 2.11.0+cu130 (CUDA 13.0)                                          |
| 磁盘空闲              | 918 GB（充足，venv 会占 ~11G）                                    |

---

## 1. 为什么用独立 venv 隔离

计划要求"不要装进 vLLM 的环境"。原因：

- SGLang 和 vLLM 各自依赖一套 `torch / flashinfer / *-kernel`，**版本可能冲突**。同环境 `pip install` 会互相覆盖，可能把已经能跑的 vLLM 搞坏。
- 即使本机两者 torch 恰好都是 2.11.0+cu130，**flashinfer、sglang-kernel、flash-attn-4 等底层 wheel 仍会互相覆盖**，隔离最稳。

> **决策**：vLLM 留在基础环境 `/home/guoda/python`，SGLang 装进独立 venv `~/venv-sglang`。两边各跑各的，端口也不同（vLLM 8000 / SGLang 30000），可以并存（但单卡显存只够一个时分别启停）。

---

## 2. 完整安装命令

```bash
# 1) 建独立 venv（基于基础环境的 python3.12）
cd ~
python3 -m venv ~/venv-sglang
source ~/venv-sglang/bin/activate

# 2) 升级 pip 并安装 SGLang 全量
pip install --upgrade pip
pip install "sglang[all]"

# 3) 验证
python3 -c "import sglang; print('sglang', sglang.__version__)"      # 0.5.13
python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"  # 0.6.12
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # 2.11.0+cu130 True
```

### 安装结果

| 组件              | 版本                                   |
| ----------------- | -------------------------------------- |
| sglang            | **0.5.13**                       |
| flashinfer_python | 0.6.12（**自动装上，无需手动**） |
| flash-attn-4      | 4.0.0b16                               |
| torch             | 2.11.0+cu130                           |
| sglang-kernel     | 0.4.3                                  |
| venv 体积         | ~11 GB                                 |

> **好消息**：FlashInfer 被自动装上且能 `import`，**不需要**退回 triton backend（计划里预留的坑没踩到）。SGLang 默认 `attention-backend=flashinfer` 直接可用。

---

## 3. 启动并跑通

```bash
source ~/venv-sglang/bin/activate
NO_PROXY="*" HF_HUB_OFFLINE=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B \
  --context-length 4096 \
  --mem-fraction-static 0.8 \
  --port 30000
```

启动约 **77 秒** 后看到 `The server is fired up and ready to roll!`

### 验证 health + chat

```bash
NO_PROXY="*" curl --noproxy '*' -s -w " [HTTP %{http_code}]" http://127.0.0.1:30000/health
# → [HTTP 200]

NO_PROXY="*" curl --noproxy '*' -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B","messages":[{"role":"user","content":"用一句话解释KV cache"}],"max_tokens":64}'
```

返回正常中文回答（"KV cache是键值对缓存的缩写..."），服务跑通 ✓。

> SGLang 返回体比 vLLM 多了 `reasoning_content`、`matched_stop`、`metadata.weight_version` 等字段。

---

## 4. 踩坑记录

### 坑 1：pip 依赖解析慢（非报错，别误判卡死）

`pip install "sglang[all]"` 在解析依赖阶段会**长时间无输出**（venv 体积停在 13M 不动），容易误以为卡死。

- **判断方法**：`ps aux | grep "[p]ip install"` 看进程是否还活着（CPU 有占用就是在跑）。
- 实测从开始到装完约几分钟，全程正常，耐心等即可。

### 坑 2：端口与代理（沿用 vLLM 的经验）

- SGLang 默认端口 **30000**（vLLM 是 8000），不冲突。
- WSL2 代理拦 localhost → 所有 curl/python 加 `NO_PROXY="*" --noproxy '*'`。

### 坑 3：单卡显存只够一个服务

RTX 3060 12GB 同时只能跑一个（各占 ~11GB）。切换时务必先停干净：

```bash
# 停 SGLang
fuser -k 30000/tcp
ps aux | grep "[s]glang" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# 停 vLLM（注意残留 EngineCore 子进程）
fuser -k 8000/tcp
ps aux | grep EngineCore | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

### 坑 4（未踩到，但计划预留）：flashinfer 不匹配

如果 flashinfer 装不上或 `import` 报错，SGLang 可退回 triton：启动加 `--attention-backend triton`。本次 flashinfer 0.6.12 正常，无需此操作。

---

## 5. vLLM ↔ SGLang 参数对照表

| vLLM                         | SGLang                       | 含义           |
| ---------------------------- | ---------------------------- | -------------- |
| `--max-model-len`          | `--context-length`         | 最大上下文长度 |
| `--gpu-memory-utilization` | `--mem-fraction-static`    | 静态显存占比   |
| 默认端口 `8000`            | 默认端口 `30000`           | HTTP 服务端口  |
| `--enforce-eager`          | `--disable-cuda-graph`     | 关 CUDA graph  |
| `--tensor-parallel-size`   | `--tp-size`                | 张量并行       |
| block_size=16（固定块）      | `--page-size`（默认 1）    | KV 分页粒度    |
| APC（块哈希，默认开）        | RadixCache（前缀树，默认开） | 前缀复用机制   |

---

## 6. 启动日志对照（KV 池 / RadixCache / 耗时）

**公平对照**：两者都 Qwen2.5-1.5B、context=4096、显存比例 **0.80**、单卡。

### KV 池大小

| 指标                       |             SGLang (mem-frac 0.8) |                      vLLM (gpu-util 0.80) |
| -------------------------- | --------------------------------: | ----------------------------------------: |
| **KV 可用 token 数** |                 **212,918** |                         **205,808** |
| KV 显存                    |             5.68 GB (K/V 各 2.84) |                                   5.5 GiB |
| 分页粒度                   | page_size=**1**（token 级） | block_size=**16**（12863 块 × 16） |
| 模型权重                   |                           3.07 GB |                                  ~3.0 GiB |

> 同样 0.80 显存比例下，两者 KV token 容量很接近（**212,918 vs 205,808，差 ~3.5%**）。差异来源：
>
> - CUDA graph 预留不同（SGLang 抓 bs[1,2,4,8] + piecewise graph；vLLM 抓更多 size）
> - 激活内存/显存估算策略不同
> - SGLang page_size=1 无块对齐浪费；vLLM block_size=16 有少量块粒度余量

### RadixCache（关键确认）

SGLang 日志：

```
Tree cache initialized: source=default impl=RadixCache hybrid_swa=False ... hierarchical=False
radix_eviction_policy='lru'
page_size=1
```

- ✅ **RadixCache 默认启用**（`disable_radix_cache=False`）
- ✅ **page_size=1** → token 级前缀复用，对齐 [[SGLang论文-RadixAttention精读-06-13]] 里"任意长度前缀都能命中"的设计
- ✅ **LRU 淘汰**（`radix_eviction_policy='lru'`），与论文 §3 一致

对照 vLLM：用 APC（块哈希，block_size=16，整块对齐才命中）。**这正是前一天实验测到"B 共享截断前缀命中率只有 82.1%"的根源**——SGLang 的 page_size=1 radix 树没有这个块对齐限制。

### 启动耗时

| 阶段                   |                                 SGLang |            vLLM |
| ---------------------- | -------------------------------------: | --------------: |
| 权重加载               |                                 2.51 s |            ~3 s |
| CUDA graph capture     | 33.2 s (bs graph) + 13.3 s (piecewise) |    含在 init 内 |
| init engine 总         |                                     — |          8.76 s |
| **端到端 ready** |                        **~77 s** | **~35 s** |

> 本次 vLLM 启动更快（35s vs 77s），主因是 **vLLM 有 torch.compile 缓存命中**（之前多次启动已缓存），而 SGLang 首次启动要完整抓 CUDA graph（含 42 个 size 的 piecewise graph，耗时 13s）。这是**冷/热启动差异**，不是框架本质差距——SGLang 二次启动也会快很多。

---

## 今日产出

- [X] **SGLang 服务可用**：health 200 + chat completion 正常返回中文 ✓
- [X] **安装笔记**（环境隔离方案 + 完整命令 + 4 个坑 + 参数对照）
- [X] **vLLM/SGLang 参数对照表 + KV 容量对比**（212,918 vs 205,808 tokens @ 0.80）

## 快速复现卡片

```bash
# === SGLang (venv) ===
source ~/venv-sglang/bin/activate
NO_PROXY="*" HF_HUB_OFFLINE=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B --context-length 4096 \
  --mem-fraction-static 0.8 --port 30000

# === vLLM (基础环境) ===
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B \
  --max-model-len 4096 --gpu-memory-utilization 0.85

# 单卡只能跑一个，切换前先停干净（见踩坑 3）
```
