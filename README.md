# vLLM 学习项目

LLM 推理框架（vLLM + SGLang）的学习与实验仓库，目标是为 2026 年 7 月的技术分享积累深入的技术理解。包含动手实验脚本、论文笔记和每日学习日志。

## 环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 3060 12GB（WSL2） |
| CUDA | Driver 13.1 / Toolkit 12.4 |
| Python | 3.12 |
| vLLM | 0.22.1 |
| 模型 | Qwen/Qwen2.5-1.5B（约 3 GiB 显存） |

## 目录结构

```
├── 01_basic_inference.py            # 离线批量推理（vllm.LLM），验证安装
├── 02_api_server.py                 # OpenAI 兼容 API 服务器测试客户端
├── 03_params.py                     # 关键参数探索基准测试
├── 04_continuous_batching.py        # 连续批处理基准（并发请求，测 RPS/TTFT/延迟）
├── 论文笔记-PagedAttention-vLLM.md   # PagedAttention 论文笔记（vLLM 核心创新）
├── SGLang与vLLM推理框架对比分享.md    # 技术分享大纲（2026.07）
└── 学习日志/                         # 每日学习日志（04-22 起）
```

## 快速开始

```bash
# 1. 离线推理
python3 01_basic_inference.py

# 2. API 服务器（先启动 server，再跑客户端）
NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B --max-model-len 4096 --gpu-memory-utilization 0.85
NO_PROXY="*" python3 02_api_server.py

# 3. 参数探索
python3 03_params.py

# 4. 连续批处理基准（需先启动 server）
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B --max-model-len 4096 --gpu-memory-utilization 0.85
NO_PROXY="*" python3 04_continuous_batching.py --concurrency 1 5 10 20 --num-requests 40
```

## 关键参数（RTX 3060 12GB）

| 参数 | 作用 | 推荐值 |
|---|---|---|
| `--max-model-len` | 最大上下文长度 → KV cache 大小 | `4096`（默认 131072 会 OOM） |
| `--gpu-memory-utilization` | vLLM 占用显存比例 | `0.85`（桌面约占 2.7GB） |
| `--enforce-eager` | 跳过 CUDA Graphs + torch.compile | 启动更快，推理慢约 20% |
| `--tensor-parallel-size` | 多卡张量并行 | `1`（单卡） |

显存分布（max_model_len=4096）：模型权重 ~3 GiB，KV cache ~6 GiB，CUDA graphs ~0.5 GiB。

## WSL2 注意事项

- vLLM 强制使用 `spawn` 多进程，所有脚本必须用 `if __name__ == "__main__"` 保护
- HTTP 代理会拦截 localhost，curl / python 调用需加 `NO_PROXY="*"`
- `vllm serve` 启动约需 60–90 秒（模型加载 + CUDA graph 捕获）
- 模型已缓存时用 `HF_HUB_OFFLINE=1` 跳过 HuggingFace 网络请求，避免代理超时
- 杀掉 server 后检查残留进程：
  ```bash
  ps aux | grep EngineCore | grep -v grep | awk '{print $2}' | xargs kill -9
  ```

## 参考资料

- vLLM 源码：https://github.com/vllm-project/vllm
- vLLM 文档：https://docs.vllm.ai
- PagedAttention 论文：https://arxiv.org/abs/2309.06180
