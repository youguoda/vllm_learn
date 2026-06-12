---
title: SGLang与vLLM学习日志 - 5月20日
date: 2026-05-20
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

# 📖 SGLang与vLLM学习日志 - 5月20日（周二）

> 距分享还有 **46** 天 | [[SGLang与vLLM推理框架对比分享]]

## 📅 今日任务

- [ ] 研究 vLLM 中投机解码的实现方式（代码层面）
- [ ] 研究 SGLang 中投机解码的实现方式（代码层面）
- [ ] 对比 EAGLE/Medusa 在两个框架中的集成差异
- [ ] 分析投机解码对延迟和吞吐量的实际影响

**预期产出：** 投机解码框架对比笔记

## 🎯 本周阶段

**Week 5：架构深入 — 进阶特性**

> 覆盖投机解码、结构化输出、量化等进阶特性
>
> 里程碑：M2: 架构深入

## 📝 学习记录

### vLLM 投机解码实现

（待填写）

### SGLang 投机解码实现

（待填写）

### EAGLE vs Medusa 框架集成对比

（待填写）

### 性能影响分析

（待填写）

---

**关联：** [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### 两框架的投机解码入口（代码层面看这里）
```bash
# vLLM：speculative_config 统一入口，支持多种 draft 方式
vllm serve $MODEL --speculative-config '{
  "method": "eagle", "model": "yuhuili/EAGLE-...", 
  "num_speculative_tokens": 5}'
# 备选 method: "ngram"(免训练,拿prompt里的n-gram当草稿) / draft 小模型

# SGLang：EAGLE 集成最深，参数直给
python -m sglang.launch_server --model-path $MODEL \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path $EAGLE_HEAD \
  --speculative-num-steps 5 --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 8
```

### 源码地标
- vLLM: `vllm/spec_decode/`（v0）/ `vllm/v1/spec_decode/`：`propose → score → accept` 三段式
- SGLang: `python/sglang/srt/speculative/`：`eagle_worker.py`、draft 树构建与 tree attention 验证

### 集成差异（对比要点）
| | vLLM | SGLang |
|--|------|--------|
| 思路 | 通用抽象，多后端可插 | 深度绑定 EAGLE，调度器协同 |
| ngram 免训练模式 | ✅（RAG/摘要白拿加速） | 有限 |
| 与 radix cache 协同 | — | 草稿/验证均吃前缀缓存 |

### 看数据时盯三个指标
1. **接受长度均值**（日志里有）：<2 基本白干，3-4 是好状态
2. ITL 降幅 vs 并发数曲线：并发↑收益↓（验证算力红利消失）
3. 显存增量：draft 头/小模型 + 树的额外 KV

### 自测题
1. ngram 投机为什么在 RAG 场景特别有效？（答案常逐字出现在检索文档里，n-gram 命中率极高）
