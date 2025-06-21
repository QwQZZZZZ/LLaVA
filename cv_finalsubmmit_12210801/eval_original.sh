nvidia-smi
# ==== 调试信息 ====
# echo "Python 路径: $(which python)"

CUDA_VISIBLE_DEVICES=6 python /home/moting/llava_project/eval_metric.py \
    --model-path /home/moting/llava_project/models/llava-v1.5-7b \



echo "评估结束"