---
title: SGLang-vLLM学习-05-06
date: 2026-05-06
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

## 📅 今日任务

> 📖 深入 vLLM PagedAttention 源码（vllm/attention/），理解 block table 机制

- [ ] 阅读 vLLM attention 层源码，梳理 PagedAttention 类结构
- [ ] 理解 block table 的分配、映射和释放流程
- [ ] 分析 KV Cache block 的物理内存管理策略
- [ ] 记录关键数据结构（BlockSpaceManager、AttentionMetadata）

**预期产出：** 源码阅读笔记

## 📍 本周阶段

|| 项目 | 内容 ||
||------|------||
|| **阶段** | M2: 架构深入 ||
|| **周次** | Week 3：架构深入 — KV Cache 管理 ||
|| **本周目标** | 搞懂 PagedAttention vs RadixAttention 设计哲学 ||
|| **距分享剩余** | 60 天 ||

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

### 源码阅读路线（v0.x 经典路径 → v1 新引擎）

```
入口: LLMEngine.step()
  └─ Scheduler.schedule()            # vllm/core/scheduler.py
       └─ BlockSpaceManager          # vllm/core/block_manager.py ★今天主角
            ├─ can_allocate()        # 新请求能进吗（空闲块够不够）
            ├─ allocate()            # prefill 分配块，建 Block Table
            ├─ append_slots()        # decode 每步追加 slot/块
            ├─ free()                # 序列结束归还块
            └─ swap_in/swap_out()    # 抢占时 GPU↔CPU 搬运
  └─ ModelRunner.execute_model()
       └─ Attention forward → PagedAttention kernel
            (输入: query + kv_cache + block_tables + seq_lens)
```

### 三个关键数据结构
| 结构 | 角色 |
|------|------|
| `PhysicalTokenBlock` | 物理块：设备、块号、引用计数（CoW 靠它） |
| `BlockTable`（List[block]） | 每序列一张，prefill 建立、decode 增长 |
| `AttentionMetadata` | 把 block_tables/seq_lens 等打包传给 kernel |

注意：新版 vLLM（v1 引擎）把这套重构进 `vllm/v1/core/kv_cache_manager.py`，思想不变（块池+引用计数+按需分配），读旧版结构更清晰。

### 高效读法
- 用 IDE 全局搜 `block_table`，沿调用链上下追，比从 main 开始快得多
- 给 `allocate()/free()` 打断点跑一条请求，**比读十遍代码有用**
- 产出物：把上面那张调用链图补全成你自己的版本

### 自测题
1. 引用计数什么时候 >1？（beam search / parallel sampling / prefix caching 共享块）
2. `can_allocate` 返回 False 会发生什么？（请求留在 waiting 队列，或触发抢占——明天调度器篇）
