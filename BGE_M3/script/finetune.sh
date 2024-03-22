#!/bin/bash

# export CUDA_VISIBLE_DEVICES="6,7"
export WANDB_PROJECT="bge_m3-law"
RUN_NAME="bgem3_plen2048_bs8_lr2e5_ga8"

torchrun --nproc_per_node 1 \
    -m FlagEmbedding.baai_general_embedding.finetune.run \
    --output_dir saved_models/$RUN_NAME \
    --model_name_or_path BAAI/bge-m3 \
    --corpus_file ./data/final/corpus/merged_dedup_corpus_indexed.json \
    --train_data ./data/final/train/merged_data_train_minedHN.jsonl \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --fp16 \
    --gradient_checkpointing True \
    --deepspeed ./ds_config.json \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --dataloader_drop_last False \
    --dataloader_num_workers 2 \
    --dataloader_prefetch_factor 2 \
    --normlized True \
    --temperature 0.05 \
    --query_max_len 128 \
    --passage_max_len 2048 \
    --train_group_size 2 \
    --negatives_cross_device \
    --use_inbatch_neg True \
    --logging_steps 5 \
    --save_strategy epoch \
    --query_instruction_for_retrieval "" \
    --report_to tensorboard \
    --logging_dir "saved_models/$RUN_NAME/log_tensorboard" \
    --run_name $RUN_NAME \
    "$@"