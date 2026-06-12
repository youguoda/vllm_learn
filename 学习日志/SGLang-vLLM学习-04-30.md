---
title: "SGLang vs vLLM 学习日志 - 04-30"
date: 2026-04-30
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

# 📚 SGLang vs vLLM 学习日志 — 4月30日 周四

> **距技术分享还有 66 天** | **阶段：M1 基础认知** | **Week 2/10**

---

## 📍 本周阶段

**Week 2：SGLang 上手 + 初步对比**
> 本周目标：SGLang 跑通，体验 RadixAttention 效果

---

## ✅ 今日任务

- [ ] 🔧 安装 SGLang + FlashInfer（注意版本兼容性）
- [ ] 🚀 用同样的模型跑推理（对比 vLLM）
- [ ] 📝 记录安装过程和踩坑点
- [ ] 预期产出：安装笔记

### 任务指引

今天是 **SGLang 环境搭建日**，核心任务是把 SGLang + FlashInfer 安装好。建议：

1. **版本兼容性**：SGLang 和 FlashInfer 对 CUDA 版本有要求，先确认当前环境 CUDA 版本，再选择对应的安装命令
2. **FlashInfer 安装**：建议使用预编译 wheel 安装，避免从源码编译耗时过长
3. **用同样的模型**：延续 vLLM 使用的 Qwen2.5-1.5B（或之前跑过的模型），方便后续直接对比
4. **记录踩坑点**：安装过程中的版本冲突、依赖问题等详细记录，后续分享时是很有价值的实战内容

---

## 📝 学习记录

### 安装过程

*(记录 SGLang + FlashInfer 安装步骤和遇到的问题)*

### 首次推理体验

*(记录用 SGLang 跑推理的首次体验和初步感受)*

### 与 vLLM 的初步对比

*(记录与 vLLM 在安装、使用体验上的初步差异)*

### 今日心得

*(一句话总结今天的收获)*

---

## 📅 本周进度一览

| 日期 | 任务 | 状态 |
|------|------|------|
| 4/29 周三 | 🔧 安装 SGLang + FlashInfer（注意版本兼容性） | ☐ |
| **4/30 周四** | **🔧 安装 SGLang + FlashInfer，用同样模型跑推理** | **⬅️ 今天** |
| 5/1 周五 | 📖 读 SGLang 论文，理解 RadixAttention 原理 | ☐ |
| 5/2 周六 | 🚀 体验 RadixAttention（prefix reuse），对比无缓存性能 | ☐ |
| 5/3 周日 | 📖 对比 SGLang vs vLLM 的架构设计差异 | ☐ |
| 5/4 周一 | 💡 自由安排 / 补进度日 | ☐ |
| 5/5 周二 | 整理本周笔记，写"SGLang 初体验 + 与 vLLM 初步对比" | ☐ |

---

## 🔗 相关笔记

- [[SGLang与vLLM推理框架对比分享]] — 主项目计划
- [[SGLang-vs-vLLM-研究笔记]] — 研究资料汇总
- [[论文笔记-PagedAttention-vLLM]] — PagedAttention 论文笔记

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### 最短安装路径
```bash
uv venv sgl-env && source sgl-env/bin/activate
uv pip install "sglang[all]"
# FlashInfer 装不上时用官方 wheel 源（按 CUDA/torch 版本选）：
# pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.4/

# 启动（OpenAI 兼容，注意默认端口 30000）
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --port 30000 --mem-fraction-static 0.85
```
```bash
curl http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"你好"}]}'
```

### vLLM ↔ SGLang 参数对照（务必记住）

| 概念 | vLLM | SGLang |
|------|------|--------|
| 显存占比 | `--gpu-memory-utilization` | `--mem-fraction-static` |
| 张量并行 | `--tensor-parallel-size` | `--tp-size` |
| 上下文上限 | `--max-model-len` | `--context-length` |
| 默认端口 | 8000 | 30000 |
| 关前缀缓存 | `--no-enable-prefix-caching` | `--disable-radix-cache` |

### 踩坑提醒
- FlashInfer 与 CUDA/torch 版本强绑定，**先 `nvcc --version` 再选 wheel**，源码编译要很久
- vLLM 和 SGLang 各自的 torch 依赖可能冲突 → **务必分开虚拟环境**
- 服务日志里直接打印 cache hit rate，是明天实验的观察口

### 自测题
1. 用同一张卡先后起两个框架对比时，最容易忘的公平性设置是什么？（显存占比、max len、dtype 对齐，且各自重启清缓存）
