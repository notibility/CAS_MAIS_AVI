import os
import numpy as np
from sentence_transformers import SentenceTransformer

# 配置路径
TEXT_DIR = '/root/autodl-tmp/yanglongjiang/project/AVI/merged_txts_r1'   # 文本目录
OUTPUT_DIR = 'features_merged_dp_r1_qwen3'    # 输出特征目录

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={"device_map": "auto"},
    tokenizer_kwargs={"padding_side": "left"},
)

for filename in os.listdir(TEXT_DIR):
    if not filename.endswith('.txt'):
        continue

    txt_path = os.path.join(TEXT_DIR, filename)
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    if not text:
        print(f"file empty：{filename}")
        continue

    # 使用 prompt_name="query" 进行编码
    embedding = model.encode(text, prompt_name="query")  # shape: (hidden_dim,)

    # 保存为 .npy 文件
    out_path = os.path.join(OUTPUT_DIR, filename.replace('.txt', '.npy'))
    np.save(out_path, embedding)
    print(f"save to: {out_path}")