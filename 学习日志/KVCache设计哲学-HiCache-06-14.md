---
title: KV Cache 设计哲学 —— 从 HiCache 看 vLLM 与 SGLang 的世界观
date: 2026-06-14
tags:
  - R/技术框架/vLLM
  - R/技术框架/SGLang
  - R/核心算法/LLM推理
  - R/核心算法/推理优化
  - output/active
related:
  - "[[SGLang源码精读-RadixAttention-06-14]]"
  - "[[vLLM源码精读-PagedAttention内存管理-06-14]]"
  - "[[KVCache架构对比图-06-14]]"
---

# KV Cache 设计哲学：从 HiCache 看两个框架的世界观

> M3 起步。先亲手制造 radix 树被迫淘汰的现场（"被踢掉的 KV 直接丢了，重放要重算"），再读懂 HiCache 怎么用"CPU 当二级缓存"救援，最后升华到两个框架的**设计哲学差异**。
>
> 本次实测还踩到一个真实的环境坑（HiCache 的 kernel IO backend 在 WSL2+RTX3060 崩溃），解法也记录在内。

---

## 步骤 1：亲手制造"淘汰=丢弃=重算"的现场

脚本 [exp_evict.py](../exp_evict.py)：先用大量不同的长前缀填满 GPU 的 radix 树缓存，再重放早期前缀，看是否还命中。

### 关键对照：缓存够 vs 缓存满

| 场景 | 第一批(全冷) TTFT | 第二批(重放) TTFT | 结论 |
|---|---:|---:|---|
| **轻压力**(20×500字, 总量<<池) | 79.9ms | **26.9ms** | 全命中(缓存够，没淘汰) |
| **重压力**(80×3000字, 总量>>池) | 266.6ms | **271.2ms** | 几乎没命中(早期前缀被淘汰) |

> KV 池 = 89764 token（mem-fraction=0.5）。轻压力下 20 个短前缀塞得下，重放全命中（27ms）；重压力下 80 个长前缀总量远超池容量，**早期前缀被 LRU 淘汰**，重放等于冷启动（271ms，比命中慢 10 倍）。

### 核心问题：被淘汰的 KV 去哪了？

读源码（[[SGLang源码精读-RadixAttention-06-14]]）知道：普通 RadixCache 的 `evict()` 调 `_evict_regular()` → `_delete_leaf()` → **`mem_pool.free(node.value)`，KV 直接丢弃**。下次还要从头 prefill 重算。

> **这就是 HiCache 要解决的问题**：GPU 显存有限，热前缀被迫淘汰后直接丢弃太浪费——能不能先搬到容量更大的 CPU 内存"暂存"，下次命中再搬回来？

---

## 步骤 2：HiCache 原理（四问四答 + 源码伏笔）

HiCache = **Hierarchical KV Cache**，分层 KV 缓存。源码在 `sglang/srt/mem_cache/hiradix_cache.py`（`HiRadixCache` 是 `RadixCache` 的子类）。

### 昨天源码里埋的伏笔

读 RadixAttention 时，`TreeNode` 有几个当时没用上的字段——**正是为 HiCache 准备的**：

```python
class TreeNode:
    self.host_value = None        # KV 在 CPU host 内存的副本
    self.host_ref_counter = 0     # CPU 副本的引用计数
    @property
    def backuped(self):           # 是否已备份到 CPU
        return self.host_value is not None
    def protect_host(self): ...   # 保护 CPU 副本不被淘汰
```

### 四问四答

**Q1：三级结构是什么？**
> GPU 显存（最快最小）→ CPU 内存（host pool，大 1.5-2 倍）→ 磁盘/NVMe（可选，更大）。实测启动日志：`Allocating 2.98 GB host memory for hierarchical KV cache`（hicache-ratio=1.5，即 host 池 = GPU 池的 1.5 倍）。

**Q2：搬移时机？**
> radix 树 evict 时**不直接丢弃，而是搬到 CPU**。源码 `hiradix_cache.py:evict()` 的关键分支：
> ```python
> if not x.backuped:                      # 还没备份
>     if write_policy == "write_back":
>         self.write_backup(x)            # ★搬到 CPU (而非丢弃)
>     else:
>         self._evict_regular(x)          # 普通: 直接丢
> else:                                   # 已备份
>     self._evict_backuped(x)             # GPU→CPU 降级: node.value=None, host_value 保留
> ```
> `_evict_backuped` 注释原文："**GPU → CPU demotion: block moves from device to host**"。节点**留在树上**（保留拓扑），只是 GPU 索引清空，CPU 副本还在。

**Q3：命中时怎么用？**
> CPU 命中 → `load_back()` 异步搬回 GPU（有延迟，但比重算省）。实测：重放第一条 **TTFT=45.6ms**，介于冷启动(280ms)和全 GPU 命中(27ms)之间——这正是"从 CPU 搬回有成本，但远优于重算"的铁证。

**Q4：适合什么负载？**
> 长前缀 + 高并发 + GPU 显存不够放所有缓存——agent 任务、RAG、多轮长对话。**关键是前缀复用率要高**，否则搬到 CPU 的也是一次性的，白搬。

### 普通 RadixCache vs HiRadixCache 的淘汰本质区别

```
普通 RadixCache evict:
  叶子 → _delete_leaf() → 从树上删除 + free KV    [KV 没了, 要重算]

HiRadixCache evict:
  叶子 → _evict_backuped() → node.value=None       [节点留树上]
                            host_value 保留          [KV 在 CPU, 可救回]
  命中 → load_back() → 异步搬回 GPU                 [省掉重算]
```

---

## 步骤 3：动手实测 HiCache 救援（含踩坑）

### 踩坑：HiCache 的 kernel IO backend 在 WSL2+RTX3060 崩溃

第一次开 HiCache 直接 `CUDA error: an illegal memory access`，反复崩溃。逐步排查：

| 尝试 | 配置 | 结果 |
|---|---|---|
| 1 | `--hicache-write-policy write_back` | CUDA illegal access 崩溃 |
| 2 | 默认 write_through | host memory 不足报错(ratio=2.0要7.5G) |
| 3 | mem-fraction 0.6 | 仍 host memory 不足 |
| 4 | ratio 1.5(host池2.98G) | warmup 阶段 CUDA illegal access 崩溃 |
| 5 | **`--hicache-io-backend direct`** | **✓ 跑通!** |

> **根因**：HiCache 默认 `io-backend=kernel`（自定义 CUDA kernel 做 GPU↔CPU 拷贝），这个 kernel 在 WSL2 + RTX 3060 + CUDA 13 环境不兼容。换成 **`direct`（普通 cudaMemcpy）** 就稳定了。
>
> **教训**：消费级卡/WSL 上跑 HiCache，加 `--hicache-io-backend direct`。这是官方文档没强调、但边缘环境必踩的坑。

### 救援实测

HiCache(direct) 跑通后，重放第一条 **TTFT=45.6ms**（vs 普通版重放 268ms）——**HiCache 从 CPU 救回了这条前缀的 KV**，省掉了重算。

> 后续几条仍是 250-280ms 未命中，因为本实验是 80 个**完全随机零复用**前缀，搬到 CPU 的总量也超过 host 池——这反而印证了 Q4：**HiCache 要前缀复用率高才有收益，纯随机负载多一层 CPU 也救不回**。

---

## 步骤 4：设计哲学对比 —— 两种世界观

读完两个框架的内存管理源码，我越来越觉得：**vLLM 和 SGLang 在 KV Cache 上的差异，本质是两种不同的"世界观"。**

### vLLM = 操作系统的"虚拟内存"哲学

```
目标：管好有限显存，让尽可能多的请求能跑
手段：分页(PagedAttention) + 引用计数 + LRU + 抢占时换出/重算
关注：显存利用率、吞吐量
```

> **类比：操作系统不关心你在看什么文件，只管把物理内存这块"地"管好。** 谁来都一视同仁地分页、回收。PagedAttention 就是把 OS 的页表、free list、COW 原样搬来——它的世界里，KV 是"需要被高效分配的内存"，至于这块 KV 是谁的前缀、热不热，它不太关心（APC 是后来加的）。

### SGLang = 数据库的"缓存系统"哲学

```
目标：最大化 KV 复用，减少重复计算
手段：radix 树精确索引 + HiCache 多级扩展(GPU→CPU→磁盘)
关注：缓存命中率、TTFT(对延迟敏感的场景)
```

> **类比：数据库的查询缓存，不太关心内存怎么分配，只管把"热数据"留住、把"前缀关系"索引好。** RadixAttention 的世界里，KV 是"值得复用的计算结果"——它用前缀树精确记录谁和谁共享、用 lock_ref 保护热前缀、用 HiCache 把放不下的热数据降级到 CPU 而不是丢弃。它**心疼每一份算过的 KV**。

### 一张表看清两种世界观

| 维度 | vLLM（虚拟内存观） | SGLang（缓存系统观） |
|---|---|---|
| KV 是什么 | 要高效分配的**内存** | 值得复用的**计算结果** |
| 核心数据结构 | 哈希表 + free list | radix 树 |
| 满了怎么办 | 抢占：换出 CPU 或重算 | HiCache：降级到 CPU 二级缓存 |
| 优化重心 | 显存利用率、吞吐 | 命中率、TTFT |
| 思想来源 | 操作系统内存管理 | 数据库/编译器缓存 |
| 类比 | "管好这块地" | "留住热数据" |

### 但两者不是对立的

这是最重要的一点：**SGLang 底层也要分页管理物理内存**（token_to_kv_pool 也是按 page 管理的），它只是**在 PagedAttention 那一层之上，又加了一层更智能的前缀索引**。

```
        SGLang
   ┌─────────────────┐
   │ RadixCache      │  ← 前缀树索引(缓存系统观)
   │   + HiCache     │
   ├─────────────────┤
   │ token_to_kv_pool│  ← 分页物理内存管理(虚拟内存观, 和 vLLM 同源)
   └─────────────────┘
```

> 所以不是"谁取代谁"，而是**SGLang = vLLM 的分页底座 + 一层缓存大脑**。vLLM 后来也加了 APC（块哈希前缀缓存），相当于在虚拟内存观上补了一点缓存观。两者在互相靠近。

---

## 步骤 5：自测 —— HiCache 收益最大的三个条件

HiCache 不是银弹（多一层 CPU 搬移有成本）。收益最大需**同时满足**三个条件：

1. **前缀长**：被淘汰的 KV 量大，搬一次 CPU 省下的重算多，值得搬。短前缀重算也就几毫秒，搬移开销可能更大。
2. **高并发**：GPU 显存不够放下所有活跃前缀，必然发生淘汰——这才给了 HiCache "接住被淘汰 KV"的用武之地。
3. **前缀复用率高**：搬到 CPU 的 KV 后面还有人用（同样的 system prompt / few-shot / 对话历史会再次到来），搬回来不白搬。

> 我的实验恰好是**反例**：80 个随机零复用前缀 → 条件 3 不满足 → HiCache 只在第一条偶然命中（45ms），其余照样未命中。这从反面证明了"复用率高"是 HiCache 的命门。

### 一句话

> **vLLM 把 KV 当内存管（够用就行），SGLang 把 KV 当宝贝护（能省则省，放不下就降级到 CPU 也不丢）。** HiCache 是"缓存系统世界观"的极致体现——连被淘汰的 KV 都舍不得扔，要给它在 CPU 上留个位置。这种"心疼计算结果"的执念，正是 SGLang 在长前缀/多轮/RAG 场景碾压的根源。

---

## 今日产出

- [x] **exp_evict.py 运行结果**：轻压力重放命中27ms，重压力重放271ms(淘汰=重算)
- [x] **HiCache 四问四答**（+ 源码 host_value/_evict_backuped/load_back 印证）
- [x] **HiCache 实测救援**：重放首条 45ms(CPU 搬回), 介于冷启动与全命中之间
- [x] **环境坑记录**：kernel IO backend 在 WSL2 崩溃 → 用 `--hicache-io-backend direct`
- [x] **KVCache 设计哲学**：虚拟内存观 vs 缓存系统观 + 两者不对立(SGLang=分页底座+缓存大脑)

## 复现命令

```bash
# 普通 RadixCache (观察淘汰=丢弃)
NO_PROXY="*" HF_HUB_OFFLINE=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B --context-length 4096 \
  --mem-fraction-static 0.5 --port 30000
NO_PROXY="*" python3 exp_evict.py 80 3000

# HiCache (WSL/消费级卡务必加 --hicache-io-backend direct!)
NO_PROXY="*" HF_HUB_OFFLINE=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B --context-length 4096 \
  --mem-fraction-static 0.45 \
  --enable-hierarchical-cache --hicache-ratio 1.5 \
  --hicache-io-backend direct --port 30000
```
