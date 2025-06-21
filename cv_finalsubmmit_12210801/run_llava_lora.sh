#!/bin/bash

# ==== 设置环境变量 ====
# 1. 明确指定使用 cuda:5 这张GPU。
# export CUDA_VISIBLE_DEVICES=5
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
# 2. 确保下面的路径指向你的conda环境和llava项目根目录
# （保持你之前的绝对路径设置，我这里只是示例）
# export PYTHONPATH="/home/moting/llava_project/LLaVA:$PYTHONPATH"
# export PATH="/home/moting/anaconda3/envs/llava/bin:$PATH"

# ==== 调试信息 ====
echo "======================================="
echo "Starting LLaVA Fine-tuning directly on server GPU(s)..."
echo "Targeting CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "Current directory: $(pwd)"
echo "Hostnames: $(hostname)"
echo "Date: $(date)"
echo "======================================="

echo "Python 路径: $(which python)"
echo "accelerate 路径: $(which accelerate)"
nvidia-smi # 显示GPU状态，此时应该只显示或高亮 cuda:5

# --- LLaVA 训练参数配置 ---
# ... (其他参数保持不变) ...
MODEL_BASE="/home/moting/llava_project/models/llava-v1.5-7b"
VISION_TOWER="/home/moting/llava_project/models/clip-vit-large-patch14-336"
DATA_PATH="/home/moting/llava_project/coco2014/annotations/annotations/llava_coco_captions_train.json"
IMAGE_FOLDER="/home/moting/llava_project/coco2014"
OUTPUT_DIR="./checkpoints/llava-v1.5-7b-lora-coco-caption_$(date +%Y%m%d_%H%M%S)"
DEEPSPEED_CONFIG="/home/moting/llava_project/LLaVA/scripts/zero3.json"
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4
LORA_R=128
LORA_ALPHA=256
MM_PROJECTOR_LR="2e-5"

echo "======================================="
echo "LLM Base Model: $MODEL_BASE"
echo "Vision Tower: $VISION_TOWER"
echo "Data Path: $DATA_PATH"
echo "Image Folder: $IMAGE_FOLDER"
echo "Output Directory: $OUTPUT_DIR"
echo "Deepspeed Config: $DEEPSPEED_CONFIG"
echo "Per Device Train Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "======================================="

# 创建输出目录，如果不存在的话
mkdir -p "$OUTPUT_DIR"

# 核心训练命令
# **重新添加 --num_gpus 1**。
# 这样 DeepSpeed 知道要用多少个本地可见的 GPU（这里是 1 个），
# 并且会从 CUDA_VISIBLE_DEVICES=5 暴露出来的唯一一个 GPU (内部视为 0 号卡) 上运行。
deepspeed --include localhost:2,5,6,7 LLaVA/llava/train/train_mem.py \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$MODEL_BASE" \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

echo "LLaVA LoRA 微调训练结束。"
echo "最终模型检查点保存在: $OUTPUT_DIR"
echo "训练日志会直接输出到终端。"
echo "======================================="