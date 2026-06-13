# 05-09 详细计划：HiCache + KV Cache 设计哲学对比

## 步骤 1：制造"需要 HiCache"的现场（30 min）

先亲手感受 radix 树被迫淘汰是什么样子。把 `--mem-fraction-static` 调低，同时跑大量不同前缀的请求，触发 evict：

```python
# exp_evict.py — 观察 radix 树淘汰
import time, openai, random, string
client = openai.OpenAI(base_url="http://localhost:30000/v1", api_key="x")
MODEL = "Qwen/Qwen2.5-1.5B"

def rand_prefix(n_char):
    return "".join(random.choices(string.ascii_lowercase + "你我他的", k=n_char))

if __name__ == "__main__":
    # 第一批：20 个不同的长前缀（填满缓存）
    print("=== 第一批：填满缓存 ===")
    prefixes = [rand_prefix(500) for _ in range(20)]
    for i, p in enumerate(prefixes):
        t0 = time.time()
        r = client.chat.completions.create(model=MODEL,
            messages=[{"role":"system","content":p},
                      {"role":"user","content":"用一句话总结。"}],
            max_tokens=32)
        print(f"请求{i:02d} TTFT≈{(time.time()-t0)*1000:.0f}ms  (都是冷，无命中)")

    # 第二批：重放第一批前缀（此时部分已被淘汰）
    print("\n=== 第二批：重放前缀，观察命中率下降 ===")
    for i, p in enumerate(prefixes[:10]):
        t0 = time.time()
        r = client.chat.completions.create(model=MODEL,
            messages=[{"role":"system","content":p},
                      {"role":"user","content":"用一句话总结。"}],
            max_tokens=32)
        print(f"重放{i:02d} TTFT≈{(time.time()-t0)*1000:.0f}ms")
```

盯 SGLang server 日志，找 `evict` 字样和 cache utilization 下降。问自己：被淘汰的 KV 去哪了？（**直接丢弃**，下次还要重算）——这就是 HiCache 要解决的问题。

## 步骤 2：读 HiCache 博客（40 min）

SGLang 官方博客：https://lmsys.org/blog/2025-01-17-sglang-v0-4/#hierachical-kv-cache-hicache

重点摘录四个问题的答案：

1. **三级结构是什么？** GPU 显存 → CPU 内存 → 磁盘（NVMe），每级容量约大 10-100 倍
2. **搬移时机？** radix 树 evict 时，不直接丢弃，而是异步搬到 CPU；CPU 满了搬磁盘
3. **命中时怎么用？** CPU 命中 → 异步搬回 GPU（有延迟，但比重算省）；磁盘命中 → 更慢但仍优于重算
4. **适合什么负载？** 长前缀 + 高并发 + GPU 显存不够放所有缓存——agent 任务、RAG、多轮长对话

## 步骤 3：动手感受"CPU 缓存救援"（30 min）

HiCache 在当前 SGLang 版本里用 `--enable-hierarchical-cache` 开启（确认版本支持，不支持则只记原理）：

```bash
# 如果版本支持：
NO_PROXY="*" python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B \
  --context-length 4096 \
  --mem-fraction-static 0.4 \      # 故意调小，让 GPU 缓存很快满
  --enable-hierarchical-cache \
  --port 30000
```

再跑 exp_evict.py 的第二批，对比开关 HiCache 的第二批 TTFT——如果有命中从 CPU 搬回，TTFT 应该介于"冷"和"全命中"之间。

## 步骤 4：写"设计哲学对比"（30 min）

`学习日志/KVCache设计哲学.md`，核心是两个类比：

**vLLM = OS 虚拟内存哲学**
- 目标：管好有限显存，让尽可能多的请求能跑
- 手段：分页（PagedAttention）、swap（换出到 CPU/重算）
- 关注点：显存利用率、吞吐量
- 类比：OS 不关心你在看什么文件，只管把内存管好

**SGLang = 缓存系统哲学**
- 目标：最大化 KV 复用，减少重复计算
- 手段：radix 树精确索引、HiCache 多级扩展
- 关注点：缓存命中率、TTFT（对延迟敏感的场景）
- 类比：数据库查询缓存，不关心内存分配，只管把热数据留住

**两者不是对立的**：SGLang 底层也要分页管理物理内存，只是在上面加了一层更智能的索引。

## 步骤 5：自测（10 min）

回答：什么负载下 HiCache 收益最大？写出三个条件：
1. 前缀长（被淘汰的 KV 量大，值得搬移）
2. 高并发（GPU 显存不够放所有活跃前缀）
3. 前缀复用率高（搬回来的 KV 还有人用，不是白搬）

## 今日产出
- [ ] exp_evict.py 运行结果（观察到淘汰现象）
- [ ] HiCache 博客四问四答笔记
- [ ] KVCache设计哲学.md（两类比 + 两者关系）
