# 微调的数据脚本
## 1.单卡RTX 3090 24GB，纯对话文本，基于finetune_lora.sh上更改，lora微调

    --lora_rank 64 \
    --lora_alpha 128 \
    --deepspeed scripts/zero3_offload.json \
    --disable_flash_attn2 True \
    --freeze_projector False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 3 \

    显卡的利用率：11684MiB /  24576MiB
    训练结果：

