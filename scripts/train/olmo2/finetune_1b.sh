#!/bin/bash
set -a && source .env && set +a

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 2 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name olmo2_1b_sft \
    --model_name_or_path allenai/OLMo-2-0425-1B \
    --model_revision main \
    --tokenizer_name allenai/OLMo-2-1124-7B \
    --tokenizer_revision main \
    --use_slow_tokenizer False \
    --add_bos \
    --chat_template_name tulu \
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
    --use_flash_attn \
    --gradient_checkpointing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --report_to wandb \
    --wandb_project_name finetune-1b-sft \
    --with_tracking \
    --logging_steps 1 \
    --checkpointing_steps 2000 \
    --seed 1
