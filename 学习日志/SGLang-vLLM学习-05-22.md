---
title: SGLang与vLLM学习日志 - 5月22日
date: 2026-05-22
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

# 📖 SGLang与vLLM学习日志 - 5月22日

> 距分享还有 **44** 天 | [[SGLang与vLLM推理框架对比分享]]

## 📅 今日任务

- [ ] 研读 vLLM constrained decoding 实现原理
- [ ] 研读 SGLang Compressed FSM 实现原理
- [ ] 对比两种结构化输出方案的性能与设计取舍
- [ ] 整理笔记，产出"结构化输出笔记"

## 🎯 本周阶段

**Week 5：架构深入 — 进阶特性**

> 覆盖投机解码、结构化输出、量化等进阶特性
>
> 里程碑：M2 - 架构深入

## 📝 学习记录

### 结构化输出：vLLM constrained decoding vs SGLang Compressed FSM

#### vLLM Constrained Decoding

（待填写）

#### SGLang Compressed FSM

（待填写）

#### 对比分析

（待填写）

#### 关键收获

（待填写）

#### 疑问与待深入

（待填写）

---

**关联：** [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### 约束解码的统一原理（两家共用的底座）
```
JSON Schema / 正则 / EBNF
   ↓ 编译
有限状态机 FSM（状态=已生成内容的语法位置）
   ↓ 每步解码
当前状态 → 合法 token 集合 → 非法 token logits = -inf → 采样
   ↓
生成的每个 token 都语法合法 ⇒ 输出 100% 可解析
```
难点不在"屏蔽"，在**预计算**：词表 10万+，每个 FSM 状态都要算一遍"哪些 token 合法"（token 是多字符的，跨语法边界）——这一步的工程质量决定性能。

### vLLM 的路线：可插拔后端
`--guided-decoding-backend`：outlines / **xgrammar** / guidance。Outlines 思路 = 正则→DFA + 预计算 state→token mask 表（构建慢、运行 O(1)）；xgrammar 用上下文无关文法+缓存优化，现为推荐默认。

### SGLang 的路线：Compressed FSM（论文贡献）
观察：JSON 里大量 token 是**确定的**（键名、引号、冒号——状态只有一条出边）。
做法：把 FSM 中的单出边链**压缩成一条边**，解码到这里直接**jump-forward**：整段确定文本一次性追加，不必逐 token 过模型 → 约束越强反而越快。SGLang 同样支持 xgrammar 后端。

### 设计取舍对比
| | vLLM | SGLang |
|--|------|--------|
| 架构 | 通用后端接口 | FSM 压缩深度集成 |
| 确定段处理 | 照常逐 token（后端逐步改进） | jump-forward 跳过 |
| 首次编译开销 | schema 首次使用需编译（可缓存） | 同 |

### 自测题
1. 为什么 token 级 mask 不能简单按"字符合法"算？（一个 token 可能跨越语法边界，如 `",` ）
2. jump-forward 为什么不损失质量？（跳过的内容本来就只有唯一合法选择）
