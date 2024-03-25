#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"
export OMP_NUM_THREADS=8
RUN_NAME="bgem3_unified_finetune"

torchrun --nproc_per_node 4 \
    -m BGE_M3.src.main \
    --output_dir saved_models/$RUN_NAME \
    --model_name_or_path BAAI/bge-m3 \
    --train_data data/final/train/splitted_data \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed BGE_M3/ds_config.json\
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last False \
    --normalized True \
    --temperature 0.05 \
    --query_max_len 128 \
    --passage_max_len 8192 \
    --train_group_size 2 \
    --negatives_cross_device \
    --logging_steps 5 \
    --save_strategy epoch \
    --same_task_within_batch True \
    --unified_finetuning True \
    --use_self_distill True \
    --report_to tensorboard \
    --logging_dir "saved_models/$RUN_NAME/log_tensorboard" \
    "$@"