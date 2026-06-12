---
title: SGLang与vLLM学习日志 - 5月28日
date: 2026-05-28
tags:
  - R/AI工具
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/AI工具/vLLM
---

# 📖 SGLang与vLLM学习日志 - 5月28日（周三）

> 距分享还有 **38** 天 | [[SGLang与vLLM推理框架对比分享]]

## 📅 今日任务

- [ ] 搭建 vLLM 实验环境：安装、模型加载、服务启动
- [ ] 搭建 SGLang 实验环境：安装、模型加载、服务启动
- [ ] 编写自动化测试脚本（Python + asyncio / Locust）
- [ ] 跑通 baseline 测试：单请求延迟基准

**预期产出：** 实验环境就绪 + baseline 数据

## 🎯 本周阶段

**Week 6：Benchmark 设计与实验执行**

> 搭建环境并运行第一批对比实验
>
> 里程碑：M3: 对比实验

## 📝 学习记录

### vLLM 环境搭建

（待填写）

### SGLang 环境搭建

（待填写）

### 测试脚本开发

（待填写）

### Baseline 测试结果

（待填写）

---

**关联：** [[SGLang与vLLM推理框架对比分享]]

---

## 📚 知识详解（快速上手 · 2026-06-12 补充）

### 环境搭建 SOP（一次配好，三周不动）
```bash
# 两个独立 venv + 版本锁定记录
uv venv env-vllm && source env-vllm/bin/activate && uv pip install vllm
python -c "import vllm; print(vllm.__version__)" >> versions.txt
uv venv env-sgl && source env-sgl/bin/activate && uv pip install "sglang[all]"
python -c "import sglang; print(sglang.__version__)" >> versions.txt
nvidia-smi --query-gpu=name,driver_version --format=csv >> versions.txt
```

### 自动化脚本骨架（asyncio 开环压测）
```python
import asyncio, time, aiohttp, json, numpy as np

async def one_request(session, prompt, results):
    t0 = time.perf_counter(); ttft = None; n = 0
    async with session.post(URL, json={
        "model":"default","stream":True,"max_tokens":256,
        "messages":[{"role":"user","content":prompt}]}) as resp:
        async for line in resp.content:
            if line.startswith(b"data:") and b"content" in line:
                n += 1
                if ttft is None: ttft = time.perf_counter() - t0
    results.append({"ttft":ttft,"total":time.perf_counter()-t0,"tokens":n})

async def main(rate, prompts):
    results = []
    async with aiohttp.ClientSession() as s:
        tasks = []
        for p in prompts:                      # 泊松到达
            tasks.append(asyncio.create_task(one_request(s,p,results)))
            await asyncio.sleep(np.random.exponential(1/rate))
        await asyncio.gather(*tasks)
    return results   # 存 jsonl，分析周用
```

### Baseline 测试（今天必须拿到的数）
并发=1、固定 20 条 prompt → 两框架的单请求 TTFT/总时长基准。**baseline 差距若 >10%，先排查配置对齐问题再继续**（常见元凶：dtype、max-model-len、缓存开关不一致）。

### 数据管理纪律
每组实验存独立 jsonl：`{框架}_{场景}_{rate}.jsonl` + 一行 metadata（版本、参数、时间）——分析周的痛苦程度由今天的命名规范决定。
