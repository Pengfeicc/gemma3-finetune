#!/bin/bash

MODEL_NAME="google/gemma-3-4b-it"

# It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `flash_attention_2`
# Cause the GPU limition i used a single A4000 GPU for finetuning, so this script is modified by using A4000 & cpu offload method
export PYTHONPATH=src:$PYTHONPATH

deepspeed src/train/train_sft.py \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_rank 32 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --num_lora_modules -1 \
    --use_liger True \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/training/data.json \
    --image_folder /path/to/your/image/folder \
    --disable_flash_attn2 True \
    --freeze_projector False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --output_dir output/test_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --logging_dir output/logs \
    --lazy_preprocess True \
    --dataloader_num_workers 0 \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 2 \
