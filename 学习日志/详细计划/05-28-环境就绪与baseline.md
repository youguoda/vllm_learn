# 05-28 详细计划：环境就绪 + Baseline 数据

## 步骤 1：两框架启动验证（40 min）

每次实验前的标准检查流程，写成脚本：

```bash
# healthcheck.sh — 每次启动 server 后执行
#!/bin/bash
FRAMEWORK=${1:-vllm}
PORT=${2:-8000}
echo "=== $FRAMEWORK healthcheck ==="

# 等待健康
for i in $(seq 1 30); do
    STATUS=$(NO_PROXY="*" curl -s -o /dev/null -w "%{http_code}" \
             http://localhost:$PORT/health 2>/dev/null)
    if [ "$STATUS" = "200" ]; then
        echo "✓ 健康检查通过 (${i}s)"
        break
    fi
    echo "等待... ($i/30)"
    sleep 3
done

# 单请求测试
echo "=== 单请求测试 ==="
NO_PROXY="*" curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B",
       "messages":[{"role":"user","content":"1+1=?"}],
       "max_tokens":16}' | python3 -c "import sys,json; d=json.load(sys.stdin); \
       print('输出:', d['choices'][0]['message']['content'])"

# 显存状态
echo "=== 显存状态 ==="
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu \
  --format=csv,noheader,nounits
```

## 步骤 2：Baseline 完整测试（50 min）

今天目标：每个框架 × 场景 A + 场景 B 各跑一次（各 1 次，不是 3 次重复，确认数据合理即可）：

```bash
# ==== vLLM ====
HF_HUB_OFFLINE=1 NO_PROXY="*" vllm serve Qwen/Qwen2.5-1.5B \
  --max-model-len 4096 --gpu-memory-utilization 0.85
bash healthcheck.sh vllm 8000

NO_PROXY="*" python3 12_bench_runner.py --framework vllm --scenario fixed_output
NO_PROXY="*" python3 12_bench_runner.py --framework vllm --scenario latency

# 清残留（重要！）
ps aux | grep EngineCore | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
sleep 5

# ==== SGLang ====
source ~/venv-sglang/bin/activate
NO_PROXY="*" HF_HUB_OFFLINE=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B --context-length 4096 \
  --mem-fraction-static 0.85 --port 30000
bash healthcheck.sh sglang 30000

NO_PROXY="*" python3 12_bench_runner.py --framework sglang --scenario fixed_output
NO_PROXY="*" python3 12_bench_runner.py --framework sglang --scenario latency
```

## 步骤 3：Baseline 快速核查（20 min）

跑完立刻检查数据是否合理，写一个快速核查脚本：

```python
# check_baseline.py
import csv
from pathlib import Path

def load(path):
    with open(path) as f: return list(csv.DictReader(f))

if __name__ == "__main__":
    print("=== Fixed Output 核查 ===")
    for fw in ("vllm", "sglang"):
        p = Path(f"bench_results/{fw}_fixed_output.csv")
        if not p.exists(): print(f"  {fw}: 文件不存在！"); continue
        rows = load(p)
        print(f"\n  {fw}:")
        for r in rows:
            rps = float(r["rps"])
            ttft = float(r["ttft_median"])
            # 合理范围检查
            ok_rps  = 0.1 < rps < 100
            ok_ttft = 10  < ttft < 5000
            status = "✓" if ok_rps and ok_ttft else "⚠ 异常"
            print(f"    并发={r['concurrency']}: RPS={rps:.2f}, TTFT={ttft:.0f}ms  {status}")

    print("\n=== Latency 核查 ===")
    for fw in ("vllm", "sglang"):
        p = Path(f"bench_results/{fw}_latency.csv")
        if not p.exists(): print(f"  {fw}: 文件不存在！"); continue
        rows = load(p)
        print(f"\n  {fw}: TTFT 应随输入长度单调递增")
        prev = 0
        for r in rows:
            ttft = float(r["ttft_median"])
            mono = "✓" if ttft > prev else "⚠ 未单调"
            print(f"    input≈{r['input_len_approx']} token: TTFT={ttft:.0f}ms  {mono}")
            prev = ttft
```

预期结果范围（不在范围内先查 server 日志）：
- 单并发 RPS：0.5-5（1.5B 小模型，每请求 1-2s）
- 单并发 TTFT：50-300ms（输入 256 token）
- 输入 2048 TTFT：应比 256 大 5-8 倍（prefill 线性）

## 步骤 4：记录入 experiment_log.md（20 min）

填写今天的环境快照和执行记录。

## 今日产出
- [ ] healthcheck.sh（可复用）
- [ ] 四个 baseline CSV（vllm/sglang × fixed_output/latency）
- [ ] check_baseline.py 输出结果（数据合理性确认）
- [ ] experiment_log.md 首条记录填写
