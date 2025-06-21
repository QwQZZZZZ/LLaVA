#!/bin/bash
#SBATCH -o output/train_with_qformer.%j.out          # 脚本执行的输出将被保存在当cuda_status.%j.out文件下，%j表示作业号;
#SBATCH --partition=a100      # 作业提交的指定分区队列为titan
#SBATCH --qos=a100            # 指定作业的QOS
#SBATCH -J cuda_status       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=6    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:1           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 

# #通过.bashrc激活conda
# source ~/.bashrc
# conda activate llava

# export PYTHONPATH="/home/songx_lab/cse12211120/.conda/envs/llava:$PYTHONPATH"
# export PATH="/home/songx_lab/cse12211120/.conda/envs/llava/bin:$PATH"

# 验证环境
# echo "Python路径: $(which python)"
# echo "CUDA版本: $(nvcc --version)"

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./models/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower ./models/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
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
    --report_to wandb