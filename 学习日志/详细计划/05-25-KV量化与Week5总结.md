# 05-25 详细计划：KV Cache 量化实验 + Week 5 总结

## 步骤 1：KV Cache FP8 量化实验（50 min）

vLLM 支持 `--kv-cache-dtype fp8_e5m2`（Ampere 可用，只影响 KV 存储格式，计算仍用 fp16）：

```bash
# 实验1：默认 fp16 KV cache
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --port 8000 2>&1 | grep "GPU blocks"
# 记录：# GPU blocks: XXXX

# 实验2：fp8_e5m2 KV cache
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8_e5m2 \
  --port 8000 2>&1 | grep "GPU blocks"
# 记录：# GPU blocks: YYYY（应约为 XXXX × 2）
```

**为什么 blocks 约翻倍**：KV cache 每个 token 的存储从 2 bytes（fp16）降到 1 byte（fp8），同样显存能放约 2 倍的 KV blocks，理论最大并发翻倍。

用核账公式验证（04-25 学的）：

```python
# verify_kv_quant.py
if __name__ == "__main__":
    # Qwen2.5-1.5B 配置（从 config.json 读取真实值）
    import json, os
    config_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/*/config.json")
    import glob
    paths = glob.glob(config_path)
    if paths:
        with open(paths[0]) as f: cfg = json.load(f)
        layers   = cfg.get("num_hidden_layers", 28)
        kv_heads = cfg.get("num_key_value_heads", 2)
        head_dim = cfg.get("hidden_size", 1536) // cfg.get("num_attention_heads", 12)
    else:
        layers, kv_heads, head_dim = 28, 2, 128  # Qwen2.5-1.5B 估计值

    block_size = 16  # vLLM 默认

    for dtype, bytes_per_elem in [("fp16", 2), ("fp8", 1)]:
        bytes_per_token = 2 * layers * kv_heads * head_dim * bytes_per_elem
        bytes_per_block = bytes_per_token * block_size
        print(f"{dtype}: {bytes_per_token/1024:.1f} KB/token, "
              f"{bytes_per_block/1024:.1f} KB/block")

    print("\n说明: fp8 每 token 减半 → 同显存 blocks 数约翻倍 → 最大并发约翻倍")
    print(f"实测 blocks: fp16={input('fp16 blocks=')} fp8={input('fp8 blocks=')}")
```

## 步骤 2：KV 量化的精度影响（30 min）

FP8 存储会引入量化误差，测一下是否影响输出质量：

```python
# kv_quant_quality.py
import openai, json

# 需要两个 server 分别跑（fp16 和 fp8_e5m2）
# 或者改用 offline LLM 方式串行比较
from vllm import LLM, SamplingParams

PROMPTS = [
    "1+1等于几？请只回答数字。",
    "计算：sin(30°)的值是多少？保留两位小数。",
    "以下哪个是质数：21, 23, 25, 27？",
    "描述巴黎铁塔的建造年份和高度。",
]

if __name__ == "__main__":
    params = SamplingParams(temperature=0.0, max_tokens=50)
    for dtype in ("auto", "fp8_e5m2"):
        print(f"\n=== kv_cache_dtype={dtype} ===")
        llm = LLM("Qwen/Qwen2.5-1.5B", max_model_len=4096,
                  gpu_memory_utilization=0.85, kv_cache_dtype=dtype)
        for p in PROMPTS:
            out = llm.generate([p], params)[0].outputs[0].text.strip()
            print(f"  Q: {p[:30]}... A: {out[:40]}")
        del llm
        import torch; torch.cuda.empty_cache()
```

预期：简单问题（数字计算、事实）两者答案相同；长推理链可能有微小差异，但对 1.5B 模型不明显。

## 步骤 3：换算成"并发提升"（20 min）

用 fp8 blocks 数算出理论最大并发变化，写进笔记：

```
fp16 blocks = XXXX，可缓存 XXXX×16 token
  → max_model_len=4096 时，理论并发 = XXXX×16/4096 = ?? 请求

fp8 blocks ≈ XXXX×2，可缓存更多 token
  → 理论并发 ≈ ?? × 2 请求

对于长上下文场景（max_model_len=8192）：
  fp16 下 GPU blocks 严重不足 → 大量排队
  fp8 下 blocks 翻倍 → 勉强能跑 8K 上下文
  → KV 量化对长上下文是"质变"而不是"量变"
```

## 步骤 4：Week 5 总结表（30 min）

新建 `学习日志/进阶特性总结.md`：

```markdown
# Week 5：三大进阶特性总结

## 1. 投机解码

| 方案 | 原理 | 适用场景 | 我的实测 | 官方数据 |
|---|---|---|---|---|
| ngram | prompt 内重复片段 | 代码/模板生成 | __x 加速 | - |
| EAGLE | 特征级 draft head | 通用（需训练head）| 无法实测 | ~2-3x |
| Medusa | 多解码头+树验证 | 通用（需训练heads）| 无法实测 | ~2x |

关键结论：加速比 = f(接受率)，接受率高才有效；ngram 在重复内容外基本无用。

## 2. 结构化输出

| 框架 | 后端 | 首次耗时 | 热身后损耗 | 合法率 |
|---|---|---|---|---|
| vLLM | xgrammar/outlines | ___ms | ~0% | 100% |
| SGLang | Compressed FSM | ___ms | ~0% | 100% |

关键结论：热身后速度损耗可忽略；Compressed FSM 在确定性字符多时略优；
生产用途固定 schema 则编译开销只发生一次。

## 3. 量化

| 方案 | 显存节省 | 速度变化 | 精度影响 | 硬件要求 |
|---|---|---|---|---|
| AWQ INT4 权重 | ~50% | ___x | 微小 | 任意 GPU |
| FP8 KV cache | KV 减半→并发翻倍 | 无明显变化 | 几乎无 | Ampere+ |
| FP8 W8A8 计算 | 权重减半 | ~1.3-2x | 微小 | Hopper(H100) |

关键结论：3060 Ampere 可用 AWQ INT4 + FP8 KV cache；FP8 计算需 H100。
KV 量化对长上下文是质变（并发翻倍），权重量化对大模型速度提升更明显。

## 三个特性的组合

理论上可以同时开：AWQ INT4 权重 + FP8 KV cache + 投机解码（ngram）
但注意：三者叠加时调试复杂，生产环境建议逐一验证后再组合。
```

## 今日产出
- [ ] KV 量化 blocks 数对比（fp16 vs fp8_e5m2）
- [ ] 质量对比结果（kv_quant_quality.py）
- [ ] 并发提升换算（带实测数字）
- [ ] 进阶特性总结.md（三大特性汇总表）
