# 05-02 详细计划：RadixAttention 多轮对话实验

## 步骤 1：主实验脚本（60 min）

新建 `06_radix_attention.py`：

```python
"""
RadixAttention 多轮对话实验
场景：同一 system prompt，多轮追问（每轮携带全部历史）→ 前缀逐轮变长
观察：每轮 TTFT + SGLang 日志/指标里的 cache hit rate
启动: NO_PROXY="*" python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B \
      --context-length 4096 --mem-fraction-static 0.8 --port 30000
"""
import time
import openai

client = openai.OpenAI(base_url="http://localhost:30000/v1", api_key="x")
MODEL = "Qwen/Qwen2.5-1.5B"

def make_sys(n_tokens_approx):
    base = "你是一位资深工业质检专家，精通缺陷检测、统计过程控制与六西格玛方法论。"
    return base * max(1, n_tokens_approx // 30)

QUESTIONS = ["轴承故障有哪些类型？", "如何用振动信号检测？", "采样频率怎么选？",
             "频谱分析关注哪些特征？", "如何区分内圈和外圈故障？"]

def run_dialog(sys_prompt, tag):
    msgs = [{"role": "system", "content": sys_prompt}]
    print(f"\n=== {tag} ===")
    for i, q in enumerate(QUESTIONS):
        msgs.append({"role": "user", "content": q})
        t0 = time.time()
        r = client.chat.completions.create(model=MODEL, messages=msgs,
                                           max_tokens=100, stream=True)
        first, chunks = None, []
        for c in r:
            if first is None:
                first = time.time() - t0
            d = c.choices[0].delta.content
            if d: chunks.append(d)
        print(f"轮{i+1}  TTFT={first*1000:.0f}ms  历史长度≈{sum(len(m['content']) for m in msgs)}字")
        msgs.append({"role": "assistant", "content": "".join(chunks)})

if __name__ == "__main__":
    for n in (500, 1000, 2000):
        run_dialog(make_sys(n), f"system prompt≈{n} token")
```

跑的同时盯 server 终端：SGLang 每个 batch 打印 `cached token` / `token usage`，记下每轮命中数。

**预期**：轮 1 冷（要算整个 system prompt）；轮 2 起 TTFT 大降且随轮数缓慢增加（只需 prefill 新增的上一轮回答+新问题，历史全在树上）。

## 步骤 2：前缀长度变体（30 min）

上面脚本已含 500/1000/2000 三档。填表：

| 前缀≈token | 轮1 TTFT | 轮2-5 平均 TTFT | 降幅% |
|---|---|---|---|
| 500 | | | |
| 1000 | | | |
| 2000 | | | |

预期规律：前缀越长，轮 1 越慢、降幅越大——命中省的就是前缀的 prefill。

## 步骤 3：关键对照——任意长度前缀命中（30 min）

复用 04-29 的 exp_partial_prefix.py，把 base_url 改成 30000 端口在 SGLang 上跑。对比第 3 条（半块对齐前缀）：

| 框架 | 共享截断前缀 TTFT | 结论 |
|---|---|---|
| vLLM（04-29 数据） | | 只命中整块部分 |
| SGLang | | 命中全部共享前缀 |

这张小表是两框架缓存机制差异最直接的实验证据，PPT 必用。

## 步骤 4：记录（20 min）

数据 + 一句话结论写进日志："RadixAttention 的本质优势不是更快的 kernel，而是更细的缓存粒度。"

## 今日产出
- [x] 07_radix_attention.py（06 已被 vLLM prefix caching 占用，改用 07）+ 三档前缀数据表
- [x] 半块前缀对照表（SGLang 94.9% vs vLLM 82.1%，多命中 15 token）

> 完成于 2026-06-13。完整笔记：[[RadixAttention多轮对话实验-06-13]]
> 核心数据：多轮对话轮2起每轮只 prefill 16-19 新 token；半块对照 SGLang 多救回 15 token（一个 block_size 的损失）。
