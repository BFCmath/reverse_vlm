#!/bin/bash
# Single GPU training script for LLaVA-v1.5-7B
# Adjusted settings for training on 1 GPU instead of 8

LLM_PATH="./vicuna1.5_7b_with_new_tokens"   # path to the LLM checkpoint
MODE="llava_v15"                          # mode to run the script
RUN_NAME="reverse_v15_7b_single_gpu"  # name of the run
DATA_PATH="./final_dataset_train.json"  # path to the training data
EVAL_DATA_PATH="./final_dataset_eval.json"  # path to the eval data
PROJECTOR_PATH="checkpoints/llava_v15_pretraining/mm_projector.bin"  # path to the projector
export TOKENIZER_PATH=$LLM_PATH            # path to the tokenizer


if [ "$MODE" = "llava_v15" ]; then
    LLM_SETTING="--llm_backbone vicuna1.5_7b --version v1"
elif [ "$MODE" = "llama31" ]; then
    LLM_SETTING="--llm_backbone llama_3_1 --llm_pad_token pad --version llama_3_1"
else
    echo "Invalid mode: $MODE"
    exit
fi

# Local setting - SINGLE GPU
GPU_SETTINGS="localhost:0"
MASTER_PORT="19487"

# Optional flags - disabled for single GPU
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Calculate effective batch size
# Original: 8 GPUs × 8 batch × 4 grad_accum = 256 global batch size
# Single GPU: 1 GPU × 1 batch × 32 grad_accum = 32 global batch size (smaller due to memory)
# You can increase grad_accum if you want larger effective batch

deepspeed --include $GPU_SETTINGS --master_port=$MASTER_PORT llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/train/zero2.json \
    --model_name_or_path $LLM_PATH \
    $LLM_SETTING \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $PROJECTOR_PATH \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/$RUN_NAME \
    --num_train_epochs 0.02 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --do_dehallucination_training True
