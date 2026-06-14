import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("assets", exist_ok=True)

gpus = ["RTX3060级\n¥1.5/h", "RTX4090级\n¥3.5/h", "A10\n¥5/h", "A100\n¥18/h"]
vllm = [0.289, 0.674, 0.962, 3.465]
sglang = [0.310, 0.724, 1.035, 3.725]
sglang_hi = [c / 2.1 for c in sglang]  # 高复用 2.1x → 成本减半

import numpy as np
x = np.arange(len(gpus))
w = 0.27
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w, vllm, w, label="vLLM(无复用)", color="#4C72B0")
ax.bar(x, sglang, w, label="SGLang(无复用)", color="#55A868")
ax.bar(x + w, sglang_hi, w, label="SGLang(高复用2.1x)", color="#2d6a3e")
ax.set_xticks(x); ax.set_xticklabels(gpus)
ax.set_ylabel("每百万 token 成本 (¥)")
ax.set_title("每百万 token 成本对比 (无复用持平, 高复用 SGLang 减半)")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig("assets/p6_cost_comparison.png", dpi=150, bbox_inches="tight")
print("saved assets/p6_cost_comparison.png")
