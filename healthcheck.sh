#!/bin/bash
FRAMEWORK=${1:-vllm}; PORT=${2:-8000}
echo "=== $FRAMEWORK healthcheck (port $PORT) ==="
for i in $(seq 1 60); do
    STATUS=$(NO_PROXY="*" curl --noproxy '*' -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:$PORT/health 2>/dev/null)
    if [ "$STATUS" = "200" ]; then echo "OK 健康检查通过 (~$((i*5))s)"; break; fi
    sleep 5
done
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits
