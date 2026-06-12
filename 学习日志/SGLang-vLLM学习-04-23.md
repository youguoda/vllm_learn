---
title: "SGLang vs vLLM 学习日志 - 04-23"
date: 2026-04-23
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/核心算法/LLM推理
  - R/AI工具/vLLM
status: in-progress
related:
  - "[[SGLang与vLLM推理框架对比分享]]"
---

# 📚 SGLang vs vLLM 学习日志 — 4月23日 周四

> **距技术分享还有 73 天** | **阶段：M1: 基础认知** | **Week 1/10 | 学习第 2 天**

---

## 📍 本周阶段

**Week 1：环境搭建 + vLLM 上手**
> 本周目标：vLLM 能跑起来，理解 PagedAttention 核心思想

---

## ✅ 今日任务

- [ ] 🔧 安装 vLLM（pip install vllm），用小模型（如 Qwen2.5-1.5B）跑通基础推理
- [ ] 预期产出：安装笔记

### 任务指引

今天的关键是**动手实践**，让 vLLM 在我的环境跑起来。建议关注：
1. **安装**：`pip install vllm`，注意 CUDA 版本兼容性
2. **基础推理**：用 Qwen2.5-1.5B 或类似小模型快速验证
3. **OpenAI 兼容 API**：`vllm serve` 启动服务，用 curl/Python 调用
4. **关键参数**：`--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`

### 快速验证命令
```bash
pip install vllm
python -c "from vllm import LLM; print('vLLM installed OK')"
vllm serve Qwen/Qwen2.5-1.5B-Instruct --trust-remote-code
```

---

## 📝 学习记录

### 核心概念理解

*(在这里记录你的理解)*

### 实践操作记录

*(记录安装步骤、命令、参数等)*

### 遇到的问题

*(记录学习过程中的疑问)*

### 今日心得

*(一句话总结今天的收获)*

---

## 📅 本周进度一览

| 日期 | 任务 | 状态 |
|------|------|------|
| 4/22 周三 | 读 vLLM 原始论文（PagedAttention），记核心笔记 | ☐ |
| 4/23 周四 | **🔧 安装 vLLM（pip install vllm），用小模型（如 Qwen2.5-1.5B）跑通基础推理** | **⬅️ 当天** |
| 4/24 周五 | 体验 Continuous Batching，调参数 | ☐ |
| 4/25 周六 | 读 vLLM 架构文档，画 PagedAttention 内存管理流程图 | ☐ |
| 4/26 周日 | 体验 Prefix Caching（APC），对比性能 | ☐ |
| 4/28 周一 | 整理本周笔记，写"vLLM 初体验" | ☐ |

---

## 🔗 相关笔记

- [[SGLang与vLLM推理框架对比分享]] — 主项目计划
- [[SGLang-vs-vLLM-研究笔记]] — 研究资料汇总
- [[论文笔记-PagedAttention-vLLM]] — PagedAttention 论文笔记

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### 最短跑通路径（NVIDIA 卡）
```bash
# 1. 独立环境（强烈建议，vLLM 对 torch 版本敏感）
uv venv vllm-env && source vllm-env/bin/activate
uv pip install vllm   # 自带匹配的 torch

# 2. 国内下载模型加速
export HF_ENDPOINT=https://hf-mirror.com

# 3. 启动 OpenAI 兼容服务
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85

# 4. 验证
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"你好"}]}'
```

### 离线批量推理（不起服务）
```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", max_model_len=4096)
out = llm.generate(["介绍一下PagedAttention"], SamplingParams(max_tokens=128))
print(out[0].outputs[0].text)
```

### 高频踩坑
- **OOM 启动失败**：先降 `--max-model-len`（默认取模型上限如 32K，KV 预算大），再降 `--gpu-memory-utilization`
- **torch 版本冲突**：不要在已装 torch 的旧环境里 pip install vllm，让它自己解析依赖
- **`--trust-remote-code`**：Qwen 等模型需要时加上
- 显存小的卡跑不动就 `--dtype half` + 1.5B 以下模型

### 自测题
1. `gpu_memory_utilization=0.9` 的含义？（vLLM 预占 90% 显存：权重 + 激活 + 剩余全部划给 KV Cache 块池）
2. 服务起来后去 `http://localhost:8000/metrics` 看哪些指标？（running/waiting 请求数、cache 使用率、TTFT 直方图）
