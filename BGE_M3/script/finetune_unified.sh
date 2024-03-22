#!/bin/bash

export CUDA_VISIBLE_DEVICES="6,7"

torchrun --nproc_per_node 2 \
    -m BGE_M3.src.main \
    --output_dir saved_models/unified_finetune_bgem3 \
    --model_name_or_path BAAI/bge-m3 \
    --train_data data/final/train/splitted_data \
    --learning_rate 2e-5 \
    --fp16 \
    --gradient_checkpointing \
    --deepspeed BGE_M3/ds_config.json\
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last False \
    --normlized True \
    --temperature 0.05 \
    --query_max_len 128 \
    --passage_max_len 2048 \
    --train_group_size 2 \
    --negatives_cross_device \
    --logging_steps 10 \
    --same_task_within_batch True \
    --unified_finetuning True \
    --use_self_distill True