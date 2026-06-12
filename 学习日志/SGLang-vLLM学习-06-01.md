---
title: SGLang与vLLM学习日志 - 6月1日
date: 2026-06-01
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

# 📖 SGLang与vLLM学习日志 - 6月1日（周日）

> 距分享还有 **34** 天 | [[SGLang与vLLM推理框架对比分享]]

## 📅 今日任务

- [ ] 整理所有原始实验数据，统一格式导入分析工具
- [ ] 绘制吞吐量对比图（并发数 vs Throughput，分 SGLang/vLLM）
- [ ] 绘制延迟对比图（并发数 vs TTFT/TPOT/Latency）
- [ ] 标注关键差异点和异常数据

**预期产出：** 吞吐量与延迟对比图表

## 🎯 本周阶段

**Week 7：实验数据分析 + 可视化**

> 把实验数据变成图表和结论
>
> 里程碑：M3: 对比实验

## 📝 学习记录

### 数据清洗与整理

（待填写）

### 吞吐量可视化

（待填写）

### 延迟可视化

（待填写）

### 关键差异标注

（待填写）

---

**关联：** [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### 数据汇总管线（pandas 一把梭）
```python
import pandas as pd, json, glob

rows = []
for f in glob.glob("results/*.jsonl"):
    framework, scene, rate = f.split("/")[-1].replace(".jsonl","").split("_")
    for line in open(f):
        d = json.loads(line); d.update(framework=framework, scene=scene, rate=int(rate))
        rows.append(d)
df = pd.DataFrame(rows)
agg = df.groupby(["scene","framework","rate"]).agg(
    ttft_p50=("ttft","median"),
    ttft_p99=("ttft", lambda x: x.quantile(.99)),
    tput=("tokens","sum"))   # 除以测试时长得 tok/s
agg.to_csv("summary.csv")
```

### 画图规范（图是给 PPT 用的，按演示标准画）
```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 中文不豆腐块
for fw, g in agg.reset_index().query("scene=='multiturn'").groupby("framework"):
    plt.plot(g["rate"], g["ttft_p99"], marker="o", label=fw)
plt.xlabel("请求速率 (req/s)"); plt.ylabel("TTFT P99 (s)"); plt.legend()
```
规范：每图一个结论、标题写人话（"多轮场景 SGLang TTFT 稳定"而不是"图3"）、两框架颜色全文统一（vLLM 一色、SGLang 一色）、关键差异点上加箭头标注。

### 异常数据处理原则
离群点不删除——标注。先查原因（超时重试？首次编译？后台进程？），能解释的写脚注，不能解释的标记"未解释异常"。**删数据是 benchmark 造假的开始。**
