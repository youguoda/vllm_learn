---
title: "SGLang-vLLM 学习日志 05-11"
date: 2026-05-11
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📌 今日任务

- [ ] 整理本周 KV Cache 相关笔记
- [ ] 撰写"KV Cache 管理对比"专题（PagedAttention vs RadixAttention）
- [ ] 产出：专题笔记

## 📍 本周阶段

> **Week 3：架构深入 — KV Cache 管理**
> 阶段：M2: 架构深入
> 本周目标：彻底搞懂 PagedAttention vs RadixAttention 的设计哲学差异

---

## ✍️ 学习记录

### 专题：KV Cache 管理对比

#### PagedAttention（vLLM）

> 

#### RadixAttention（SGLang）

> 

#### 核心差异对比

| 维度 | PagedAttention | RadixAttention |
|------|---------------|----------------|
| 设计哲学 |  |  |
| 内存管理 |  |  |
| 前缀复用 |  |  |
| 适用场景 |  |  |

#### 关键收获

-

---

## 🔗 关联

- [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### "KV Cache 管理对比"专题笔记骨架（直接套用）

```markdown
# KV Cache 管理：PagedAttention vs RadixAttention

## 1. 为什么 KV Cache 是推理系统的核心矛盾
显存公式：每token KV = 2×layers×kv_heads×head_dim×字节数
→ 7B 模型 1万token ≈ 0.5GB → 并发受限于 KV 而非算力

## 2. vLLM：内存管理视角（块/页表/CoW/抢占）
## 3. SGLang：缓存系统视角（radix树/LRU/cache-aware调度/HiCache）
## 4. 对比表（粒度/判定/调度/淘汰/扩展）→ 引用 5/7、5/9 的表
## 5. 实测证据 → 引用 5/10 的复用率曲线
## 6. 选型结论（场景化，不站队）
```

### 第 6 节的场景化结论（先写假设）
| 场景 | 推荐 | 依据 |
|------|------|------|
| 异构请求、高吞吐批处理 | vLLM | 块管理碎片低、生态成熟 |
| 多轮对话/Agent/RAG 固定模板 | SGLang | 前缀自动复用，TTFT 优势随复用率放大 |
| 超多会话间歇活跃 | SGLang+HiCache | 分层缓存免重算 |
| 两者差距在缩小 | — | vLLM APC 持续改进，结论要写"截至本测试版本" |

### 写作纪律（防止专题变摘抄）
每个机制配一个**自己实验的数字或一张自己画的图**；写不出数字的小节降级为"待验证"。
