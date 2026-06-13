# 04-30 详细计划：安装 SGLang + 首次推理

## 步骤 1：独立环境安装（60 min）

不要装进 vLLM 的环境——两者依赖的 torch/flashinfer 版本可能冲突。

```bash
cd ~
python3 -m venv ~/venv-sglang
source ~/venv-sglang/bin/activate

pip install --upgrade pip
pip install "sglang[all]"
# FlashInfer 若未被自动装上，按官方说明装对应 torch/cu 版本的 wheel：
# pip install flashinfer-python -i https://flashinfer.ai/whl/<cuXXX>/<torchX.X>/
python3 -c "import sglang; print(sglang.__version__)"
python3 -c "import flashinfer" 2>&1   # 报错就记录，SGLang 可退回 triton backend
```

WSL2 预期坑（遇到就记进安装笔记，这些本身就是分享素材）：
- 代理拦 localhost → 所有 curl/python 加 `NO_PROXY="*"`
- flashinfer wheel 与 torch 版本不匹配 → 用 `--attention-backend triton` 绕过
- 显存被 vLLM 残留进程占着 → `ps aux | grep -E "EngineCore|sglang" | grep -v grep`

## 步骤 2：启动并跑通（30 min）

```bash
NO_PROXY="*" HF_HUB_OFFLINE=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B \
  --context-length 4096 \
  --mem-fraction-static 0.8 \
  --port 30000
```

参数对照（写进笔记，05-04 跑分要对齐用）：

| vLLM | SGLang | 含义 |
|---|---|---|
| --max-model-len | --context-length | 最大上下文 |
| --gpu-memory-utilization | --mem-fraction-static | 显存占比 |
| 默认 8000 | 默认 30000 | 端口 |

验证：

```bash
NO_PROXY="*" curl -s http://localhost:30000/health
NO_PROXY="*" curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B","messages":[{"role":"user","content":"用一句话解释KV cache"}],"max_tokens":64}'
```

## 步骤 3：对照启动日志（30 min）

把 SGLang 启动日志和 vLLM 的并排看，找到并记录：

1. **KV 池大小**：SGLang 打印 `KV Cache is allocated. #tokens: XXXX`，vLLM 打印 `# GPU blocks: XXXX`（×16 换算成 token 数）——同一显存比例下两个数接近吗？差在哪（CUDA graph 预留、激活内存估算不同）
2. **radix cache 字样**：确认 `RadixCache` 默认启用
3. 启动耗时对比（都含 CUDA graph capture）

## 步骤 4：安装笔记（30 min）

`学习日志/SGLang安装笔记.md`：环境隔离方式、完整命令、每个报错+解法、参数对照表、KV 池对比数字。

## 今日产出
- [ ] SGLang 服务可用（health + chat 都通）
- [ ] 安装笔记（含踩坑）
- [ ] vLLM/SGLang 参数对照表 + KV 容量对比
