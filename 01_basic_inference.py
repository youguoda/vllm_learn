"""
vLLM 基础离线推理 — 验证安装正确性
模型: Qwen/Qwen2.5-1.5B (约 3GB, RTX 3060 12GB 轻松运行)

注意: WSL2 下 vLLM 使用 spawn 多进程, 必须用 if __name__ == '__main__' 保护
"""

from vllm import LLM, SamplingParams


def main():
    # 1. 加载模型 — vLLM 会自动从 HuggingFace 下载
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"Loading model: {model_name} ...")
    llm = LLM(
        model=model_name,
        max_model_len=4096,          # 默认 131072 太大, 12GB 卡放不下 KV cache
        gpu_memory_utilization=0.85,  # 留一些给桌面和系统
    )

    # 2. 准备 prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    # 3. 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
    )

    # 4. 批量推理
    print("\n--- Generating ---\n")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print()


if __name__ == "__main__":
    main()
