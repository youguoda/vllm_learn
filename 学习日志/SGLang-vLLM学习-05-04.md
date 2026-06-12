---
title: SGLang-vLLM学习-05-04
date: 2026-05-04
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📅 今日任务

> 🔧 同一模型、同一硬件，vLLM vs SGLang 简单对比跑分

- [ ] 使用相同模型（如 Qwen2.5-1.5B）分别在 vLLM 和 SGLang 上跑推理
- [ ] 测试不同并发数下的 throughput 和 latency
- [ ] 记录 TTFT（首 token 延迟）和 ITL（token 间延迟）数据
- [ ] 绘制简单对比图表，初步感受两者差异

**预期产出：** 初步对比数据

## 📍 本周阶段

|| 项目 | 内容 ||
||------|------||
|| **阶段** | M1: 基础认知 ||
|| **周次** | Week 2：SGLang 上手 + 初步对比 ||
|| **本周目标** | SGLang 跑通，体验 RadixAttention 效果 ||
|| **距分享剩余** | 62 天 ||

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

### 公平对比的 6 条军规（违反任何一条数据作废）
1. 同模型、同卡、同 dtype、同 `max-model-len`
2. 显存占比对齐（0.85 vs 0.85）
3. **前缀缓存要么都开、要么都关**（否则比的是缓存不是引擎）
4. 每轮测试前重启服务（清缓存、清碎片）
5. 先 warmup 几十条再计数（首批含编译/加载开销）
6. 同一负载数据集、同一随机种子

### 现成压测命令
```bash
# vLLM 侧
vllm bench serve --model $M --num-prompts 300 --request-rate 8

# SGLang 侧（同一脚本可测任意 OpenAI 兼容端点）
python -m sglang.bench_serving --backend sglang \
  --num-prompts 300 --request-rate 8 --port 30000
```

### 指标读法
| 指标 | 含义 | 谁在乎 |
|------|------|--------|
| Throughput (tok/s) | 系统总产能 | 算成本的人 |
| TTFT P50/P99 | 首字等待 | 交互体验 |
| TPOT/ITL | 打字速度 | 流式体验 |
| 注意 | P99 比均值诚实得多，尾延迟才暴露调度差异 | |

### 今天的预期形态
小模型+低并发：两者接近（瓶颈在 kernel，都用 FlashAttention/FlashInfer 一类）；**差异要到高并发、共享前缀、结构化输出场景才拉开**——这正是后面几周实验矩阵的设计逻辑。

### 数据表模板
| 并发 | 框架 | Throughput | TTFT P50 | TTFT P99 | ITL P50 |
|------|------|-----------|----------|----------|---------|
