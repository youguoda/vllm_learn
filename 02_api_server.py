"""
vLLM OpenAI 兼容 API 服务器测试

启动服务:
    vllm serve Qwen/Qwen2.5-1.5B --max-model-len 4096 --gpu-memory-utilization 0.85

然后用 curl 或 Python 客户端调用。
"""

import json
import subprocess
import sys
import time

import httpx

BASE_URL = "http://localhost:8000"


def wait_for_server(timeout=120):
    """等服务就绪"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                print("Server is ready!")
                return True
        except httpx.ConnectError:
            pass
        time.sleep(2)
    return False


def test_models():
    """列出可用模型"""
    r = httpx.get(f"{BASE_URL}/v1/models")
    data = r.json()
    print("=== Models ===")
    for m in data["data"]:
        print(f"  {m['id']}")
    print()
    return data["data"][0]["id"]


def test_completion(model_id):
    """测试 /v1/completions"""
    print("=== Completion ===")
    r = httpx.post(
        f"{BASE_URL}/v1/completions",
        json={
            "model": model_id,
            "prompt": "San Francisco is a",
            "max_tokens": 32,
            "temperature": 0.7,
        },
        timeout=30,
    )
    data = r.json()
    print(f"  Prompt:    'San Francisco is a'")
    print(f"  Generated: {data['choices'][0]['text']!r}")
    print(f"  Tokens:    {data['usage']}")
    print()


def test_chat_completion(model_id):
    """测试 /v1/chat/completions"""
    print("=== Chat Completion ===")
    r = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of Japan?"},
            ],
            "max_tokens": 64,
            "temperature": 0.7,
        },
        timeout=30,
    )
    data = r.json()
    msg = data["choices"][0]["message"]["content"]
    print(f"  User: What is the capital of Japan?")
    print(f"  Bot:  {msg}")
    print(f"  Tokens: {data['usage']}")
    print()


def main():
    if not wait_for_server():
        print("Server not ready, exiting")
        sys.exit(1)

    model_id = test_models()
    test_completion(model_id)
    test_chat_completion(model_id)


if __name__ == "__main__":
    main()
