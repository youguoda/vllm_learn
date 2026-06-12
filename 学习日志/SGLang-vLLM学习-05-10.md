---
title: SGLang-vLLM学习-05-10
date: 2026-05-10
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📅 今日任务

> 🔧 动手实验：不同前缀复用率下的性能差异

- [ ] 设计不同前缀复用率场景（0%、25%、50%、75%、100%）
- [ ] 在 vLLM（PagedAttention）下运行各场景，记录 TTFT 和 throughput
- [ ] 在 SGLang（RadixAttention）下运行相同场景，对比数据
- [ ] 绘制性能对比图表，分析复用率对两种架构的影响差异

**预期产出：** 实验数据

## 📍 本周阶段

| 项目 | 内容 |
|------|------|
| **阶段** | M2: 架构深入 |
| **周次** | Week 3：架构深入 — KV Cache 管理 |
| **本周目标** | 搞懂 PagedAttention vs RadixAttention 设计哲学 |
| **距分享剩余** | 56 天 |

## 📝 学习记录

%% 在下方记录今天的学习内容 %%

### 核心笔记

-

### 疑问与待查

-

### 心得体会

-

## 🔗 关联

- [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### 实验设计：复用率是唯一自变量
构造 5 组负载，总 token 长度一致（如 2048 前缀 + 256 问题）：

| 复用率 | 构造方法 |
|--------|---------|
| 0% | 每条请求前缀完全随机 |
| 25% | 前 512 token 固定，其余随机 |
| 50% | 前 1024 固定 |
| 75% | 前 1536 固定 |
| 100% | 前缀完全相同，只换最后的问题 |

```python
import random, string
def make_prompt(shared_len, total_len=2048, shared="质检背景资料..."*200):
    fixed = shared[:shared_len]
    rand  = "".join(random.choices("的一是在了有和人这中大为", k=total_len-shared_len))
    return fixed + rand + " 请回答：编号" + str(random.random())
```

### 流程纪律
每个复用率档位：重启服务 → warmup 10 条 → 正式 100 条（rate 固定如 4 req/s）→ 记录 TTFT P50/P99、吞吐、命中率 → 下一档。**vLLM 与 SGLang 各跑一遍同样负载。**

### 预期曲线与解读
- SGLang：TTFT 随复用率近似线性下降（命中即免算）
- vLLM(APC)：阶梯式下降（块对齐才命中），低复用率档位收益弱于 SGLang
- 0% 档位：两者都应回到基线——**若 0% 时开缓存反而变慢，那就是缓存管理开销，单独记下来，这是分享里的好细节**

### 图表建议
X=复用率，Y=TTFT P50，两条线（vLLM/SGLang）+ 阴影标 P99——一张图讲完本周故事。
