---
title: SGLang与vLLM学习日志 - 5月23日
date: 2026-05-23
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

# 📖 SGLang与vLLM学习日志 - 5月23日（周五）

> 距分享还有 **43** 天 | [[SGLang与vLLM推理框架对比分享]]

## 📅 今日任务

- [ ] 深入 vLLM constrained decoding 的 FSM 构建（正则→NFA→DFA→mask）
- [ ] 深入 SGLang Compressed FSM 的压缩算法原理
- [ ] 对比两种结构化输出方案的 token-level 约束效率
- [ ] 整理结构化输出完整对比笔记

**预期产出：** 结构化输出深度对比笔记

## 🎯 本周阶段

**Week 5：架构深入 — 进阶特性**

> 覆盖投机解码、结构化输出、量化等进阶特性
>
> 里程碑：M2: 架构深入

## 📝 学习记录

### vLLM constrained decoding FSM 构建

（待填写）

### SGLang Compressed FSM 压缩算法

（待填写）

### Token 级约束效率对比

（待填写）

### 关键发现与待深入

（待填写）

---

**关联：** [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### FSM 构建全链路（深入版，画一遍）
```
正则 r'\{"name": "[a-z]+"\}'
  → NFA（Thompson 构造，含 ε 边）
  → DFA（子集构造，状态确定化）
  → token-level 转移表：state × vocab → next_state / 非法
     （把"字符级 DFA"提升到"token 级"：对每个 token 的
      字符序列在 DFA 上走一遍，能走通才合法）
  → 运行时：当前 state 查表得合法 token 位图 → logits mask
```
开销集中在**词表×状态的预计算**（10万 token × 上千状态），所以：编译结果按 schema 缓存、首请求慢后续快——压测时务必排除首次编译。

### Compressed FSM 的压缩算法
1. 在 token 级 DFA 上找**单出边链**（每个状态只有一个合法 token）
2. 链上节点合并成一条复合边，边上挂整段 token 序列
3. 解码命中复合边 → 直接 append 整段（jump-forward），跳过 N 次前向

注意一个细节：jump-forward 段落的 KV 仍需补算一次 prefill（内容进上下文），但这是并行 prefill 而非串行 decode，**仍然省大头**。

### token-level 约束效率对比实验（今天产出）
用同一个复杂 schema（嵌套对象+数组+枚举）测：
| 方案 | 吞吐 | 相对无约束 | 备注 |
|------|------|-----------|------|
| 无约束（提示词求JSON） | 基线 | 100% | 可解析率 <100%！ |
| vLLM+xgrammar | | | |
| SGLang compressed FSM | | | 确定段占比越高越占优 |

### 给分享的金句
"约束解码把'求模型输出 JSON'变成'模型只能输出 JSON'——SGLang 还顺手把确定的部分跳过不算了。"
