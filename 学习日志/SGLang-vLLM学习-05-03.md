---
title: SGLang-vLLM学习-05-03
date: 2026-05-03
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📅 今日任务

> 🔧 体验 SGLang 的 Structured Output（JSON schema 约束生成）

- [ ] 使用 SGLang 的 JSON schema 约束生成功能
- [ ] 对比有/无 schema 约束的生成速度和质量
- [ ] 测试复杂 schema 的可靠性（嵌套字段、数组等）
- [ ] 记录结构化输出的使用方法和注意事项

**预期产出：** 结构化输出笔记

## 📍 本周阶段

|| 项目 | 内容 ||
||------|------||
|| **阶段** | M1: 基础认知 ||
|| **周次** | Week 2：SGLang 上手 + 初步对比 ||
|| **本周目标** | SGLang 跑通，体验 RadixAttention 效果 ||
|| **距分享剩余** | 63 天 ||

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

### SGLang 结构化输出三种姿势
```python
# ① JSON Schema（最常用）
import json, openai
client = openai.OpenAI(base_url="http://localhost:30000/v1", api_key="x")
schema = {"type":"object","properties":{
    "name":{"type":"string"},
    "defects":{"type":"array","items":{"type":"string"}},
    "pass":{"type":"boolean"}},
  "required":["name","defects","pass"]}
r = client.chat.completions.create(model="default",
    messages=[{"role":"user","content":"输出质检结果JSON"}],
    response_format={"type":"json_schema",
        "json_schema":{"name":"qc","schema":schema}})
print(json.loads(r.choices[0].message.content))  # 必定合法
```
```python
# ② 正则约束：extra_body={"regex": r"(优|良|差)"}
# ③ EBNF 文法：extra_body={"ebnf": "root ::= ..."}（复杂DSL场景）
```

### 原理一句话
把 schema 编译成**有限状态机**，每步解码时只允许"合法的下一个 token"（给非法 token 的 logits 置 -inf）——所以输出**保证**可解析，不是"大概率"。

### 今天实验要点
1. 对比有/无 schema 的**输出速度**：确定性强的段（键名、引号、括号）SGLang 会 jump-forward 直接跳过 → 复杂 schema 反而可能**更快**
2. 压力测试嵌套 schema（数组里套对象）看可靠性
3. 记录：约束生成对质量的副作用（过严的 schema 会把模型"挤"出自然表达）

### 工程意义（写进分享）
质检/Agent 场景下游是程序不是人——**100% 可解析**比"提示词里求模型输出 JSON"可靠一个数量级。
