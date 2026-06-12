---
title: SGLang-vLLM学习-05-05
date: 2026-05-05
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📅 今日任务

> ✍️ 整理 Week 1-2 笔记，写"SGLang vs vLLM 初体验"总结 — M1 交付

- [ ] 回顾 Week 1（vLLM 环境搭建、PagedAttention、Prefix Caching）
- [ ] 回顾 Week 2（SGLang 安装、RadixAttention、Structured Output、对比跑分）
- [ ] 整理关键发现和初步结论
- [ ] 写一份 M1 阶段总结文档

**预期产出：** 阶段 M1 交付笔记

## 📍 本周阶段

|| 项目 | 内容 ||
||------|------||
|| **阶段** | M1 交付日 ||
|| **周次** | Week 2：SGLang 上手 + 初步对比 ||
|| **本周目标** | SGLang 跑通，体验 RadixAttention 效果 ||
|| **距分享剩余** | 61 天 ||

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

### M1 总结写作模板（30 分钟填空式完成）

```markdown
# SGLang vs vLLM 初体验（M1 总结）

## 1. 我装了什么
环境 / 版本 / 模型 / 两个框架各自的安装坑（各一句话）

## 2. 两个核心创新的一句话理解
- vLLM = PagedAttention（显存分页）+ Continuous Batching
- SGLang = RadixAttention（前缀树复用）+ Compressed FSM

## 3. 我亲手验证过的三个事实
- 事实1：APC/Radix 开关对 TTFT 的影响 = ___（贴数据）
- 事实2：并发 1→32 时吞吐变化 = ___
- 事实3：JSON schema 约束输出 100% 可解析，速度 ___

## 4. 我现在的初步判断（M2 去验证）
- 通用高吞吐服务 → 倾向 ___，因为 ___
- 多轮/Agent/共享前缀 → 倾向 ___，因为 ___

## 5. 遗留疑问清单（带进源码周）
```

### 写总结的纪律
- 只写**自己跑出来的数字**，社区数据放参考链接
- 每个结论必须挂一个证据；没证据的写进"疑问清单"而不是结论
- 写完贴进 [[SGLang与vLLM推理框架对比分享]] 的素材区
