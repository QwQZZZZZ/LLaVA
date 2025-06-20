import os
import ast
import matplotlib.pyplot as plt

# 日志路径
log_path = "/lab/zhangjg_lab/30028000/llava/llava.31302.out"
# 图片保存路径
save_dir = "/lab/zhangjg_lab/30028000/llava/loss_images"
save_path = os.path.join(save_dir, "Resnet_loss_curve.png")

# 确保保存路径存在
os.makedirs(save_dir, exist_ok=True)

# 记录loss值
loss_values = []

# 读取日志文件
with open(log_path, "r") as f:
    lines = f.readlines()

# 提取目标行范围内的loss值
for i in range(72, 6545):  # 注意：第167行为索引166
    line = lines[i].strip()
    try:
        if line.startswith("{") and "loss" in line:
            log_dict = ast.literal_eval(line)
            if 'loss' in log_dict:
                loss_values.append(log_dict['loss'])
    except Exception as e:
        print(f"Warning: Line {i+1} parse error: {e}")

import numpy as np

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
# 画图并保存
smoothed_loss = moving_average(loss_values, window_size=10)

# 画图
plt.figure(figsize=(10, 5))
plt.plot(range(len(loss_values)), loss_values, alpha=0.3, label='Raw Loss')  # 原始曲线（浅色）
plt.plot(range(len(smoothed_loss)), smoothed_loss, color='red', label='Smoothed Loss')  # 平滑曲线
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Resnet50 encoder Training Loss Curve (Smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
