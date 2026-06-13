# 04-29 详细计划：SGLang 论文 / RadixAttention

## 步骤 1：先制造"vLLM 命中不了"的现场（20 min）

目的：亲眼看到块哈希的局限，再去读 radix 树就有了"为什么"。

启动 vLLM（APC 默认开）后跑：

```python
# exp_partial_prefix.py — 构造"半块对齐"前缀
# 原理：vLLM 块大小16 token，APC 只能命中【整块】。
# 两个请求共享前 200 字符但不在块边界对齐时，尾部半块无法复用。
import time, openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
MODEL = "Qwen/Qwen2.5-1.5B"

BASE = "你是一位资深工业质检专家，精通缺陷检测、统计过程控制、六西格玛方法论。" * 10

def ttft(prompt):
    t0 = time.time()
    r = client.completions.create(model=MODEL, prompt=prompt, max_tokens=16, stream=True)
    next(iter(r))
    return time.time() - t0

if __name__ == "__main__":
    print("A 冷启动:", ttft(BASE + "请分析轴承故障。"))
    print("A 重复(应命中):", ttft(BASE + "请分析轴承故障。"))
    # 前缀截断到非块边界位置再接不同内容 → 共享部分的最后一个不满块无法命中
    print("B 共享截断前缀:", ttft(BASE[: len(BASE)//2 + 3] + "请分析齿轮磨损。"))
```

记录三个 TTFT。预期：第 2 条远小于第 1 条；第 3 条只省了整块对齐的部分。然后抓指标：

```bash
NO_PROXY="*" curl -s http://localhost:8000/metrics | grep -i prefix
```

## 步骤 2：带问题读论文（60 min）

论文：*SGLang: Efficient Execution of Structured Language Model Programs*（arXiv:2312.07104），只精读 §3 RadixAttention。带着四个问题，读到答案就摘到笔记：

1. radix 树的节点存什么？（token 序列片段 + 指向 KV 池的引用）
2. 为什么用 radix 树而不是普通 trie？（边可以存"一串"token，节点数少）
3. 命中任意长度前缀怎么做到？（匹配到一半可以 split 节点）
4. 淘汰策略？（LRU，叶子优先，引用计数>0 的不淘汰）

## 步骤 3：手画 radix 树演化（30 min）

纸上推演，三个请求依次到达：

```
R1: [系统提示S] + "分析轴承故障"
R2: [系统提示S] + "分析齿轮磨损"
R3: [系统提示S] + "分析轴承寿命"
```

画出每步后的树：
- R1 后：root → (S+"分析轴承故障") 一条边
- R2 到达：公共前缀 S+"分析" → 原边 **split**，root → (S+"分析") → 分叉("轴承故障" / "齿轮磨损")
- R3 到达：在"轴承"处再 split 一次
- 然后假设内存不足，标出 LRU 会先淘汰哪个叶子（最久未访问且引用计数为 0 的）

把这张图拍照/重画存入笔记——05-07 读源码、05-08 画架构图都要用它。

## 步骤 4：自测（10 min）

一句话写出：vLLM 哈希按**16-token 块**为粒度、必须整块对齐才命中；SGLang radix 树按**token 前缀**为粒度、任意长度都能命中，代价是树的维护和锁开销。再对照步骤 1 第 3 条请求的数据验证。

## 今日产出
- [ ] exp_partial_prefix.py 三条 TTFT 数据
- [ ] 论文笔记（四问四答）
- [ ] 手画 radix 树演化图
