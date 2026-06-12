---
title: SGLang-vLLM学习-05-02
date: 2026-05-02
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📅 今日任务

> 🔧 体验 SGLang 的 RadixAttention：构造多轮对话场景，观察 KV Cache 命中率

- [ ] 构造多轮对话场景（相同前缀 + 不同追问）
- [ ] 观察 SGLang 的 KV Cache 命中率指标
- [ ] 对比不同共享前缀长度下的性能提升
- [ ] 记录 RadixAttention 自动前缀复用的效果数据

**预期产出：** RadixAttention 效果数据

## 📍 本周阶段

|| 项目 | 内容 ||
||------|------||
|| **阶段** | M1: 基础认知 ||
|| **周次** | Week 2：SGLang 上手 + 初步对比 ||
|| **本周目标** | SGLang 跑通，体验 RadixAttention 效果 ||
|| **距分享剩余** | 64 天 ||

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

### 实验设计：让 RadixAttention 现出原形
变量只留一个：**共享前缀长度**。

```python
import time, openai
client = openai.OpenAI(base_url="http://localhost:30000/v1", api_key="x")
PREFIX = "（一段约2000 token的固定背景资料……）"
QUESTIONS = ["问题A","问题B","问题C", ...]  # 20个不同追问

def ttft(messages):
    t0 = time.time()
    s = client.chat.completions.create(model="default", messages=messages,
                                       stream=True, max_tokens=32)
    next(iter(s)); return time.time() - t0

# 场景1 冷启动：第一次发 → 全量 prefill
# 场景2 热命中：换问题再发 → 前缀全命中，TTFT 应大降
for q in QUESTIONS:
    print(ttft([{"role":"user","content":PREFIX + q}]))
```

### 观察哪里
1. 服务端日志每个 batch 打印 **cache hit rate**（token 级命中率）
2. 对比 `--disable-radix-cache` 重启后的同负载 TTFT → 这就是"开关前后"硬数据
3. 多轮对话场景：把上一轮回答拼进 messages 再问 → 命中率应随轮数走高

### 记录模板
| 场景 | 前缀token | 命中率 | TTFT(冷) | TTFT(热) | 提升 |
|------|----------|--------|----------|----------|------|

### 预期结论（先写假设再验证）
前缀越长、追问越多 → 命中率越高 → TTFT 降幅越大；输出长度不变时 ITL 几乎不变（复用只省 prefill）。
