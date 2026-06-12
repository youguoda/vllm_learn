---
title: SGLang与vLLM学习日志 - 5月30日
date: 2026-05-30
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

# 📖 SGLang与vLLM学习日志 - 5月30日（周五）

> 距分享还有 **36** 天 | [[SGLang与vLLM推理框架对比分享]]

## 📅 今日任务

- [ ] 执行 TTFT（首Token时间）对比实验：不同输入长度下的表现
- [ ] 执行 End-to-end Latency 对比实验：不同输出长度下的表现
- [ ] 测试流式输出（Streaming）场景下的性能差异
- [ ] 整理延迟维度实验数据

**预期产出：** 延迟对比实验数据

## 🎯 本周阶段

**Week 6：Benchmark 设计与实验执行**

> 完成核心对比实验场景
>
> 里程碑：M3: 对比实验

## 📝 学习记录

### TTFT 实验

（待填写）

### End-to-end Latency 实验

（待填写）

### Streaming 性能对比

（待填写）

### 数据汇总

（待填写）

---

**关联：** [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### 延迟实验设计（固定低负载，rate 取拐点的 1/4）
吞吐实验问"系统能扛多少"，今天问"用户等多久"——**必须在非饱和区测**，否则测的是排队不是引擎。

**实验1 TTFT vs 输入长度**：输入 128/512/2K/8K（输出固定64）
预期：TTFT 随输入近似线性（prefill 算力瓶颈）；超长输入看 chunked prefill 行为差异。

**实验2 E2E vs 输出长度**：输出 64/256/1K（输入固定512）
预期：E2E ≈ TTFT + 输出×ITL，斜率就是 ITL——验证两框架 decode 速度差异。

**实验3 流式体验**：逐 chunk 记录到达时间戳
```python
stamps = []
async for line in resp.content:
    if b"content" in line: stamps.append(time.perf_counter())
itl = np.diff(stamps)   # 看 P50/P99/最大间隔（卡顿）
```
ITL 直方图比均值有信息量得多：**双峰/长尾 = 调度抖动**（chunked prefill 插队、抢占、缓存淘汰），两框架的尾部形态对比是高级素材。

### 容易翻车的点
- stream 模式 server 可能多 token 合一个 chunk 发 → ITL 假性变好，确认 chunk 粒度
- 输入长度用 tokenizer 实际数，别按字符估
