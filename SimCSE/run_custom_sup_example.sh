#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=2

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)
SEED=$(expr $RANDOM)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train_custom.py \
    --seed $SEED \
    --model_name_or_path vinai/phobert-base-v2 \
    --zalo_train_file generated_data/zalo_legal/zalo_legal_hardneg.json \
    --zalo_collection_file generated_data/zalo_legal/zalo_legal_collections.json \
    --msmarco_train_file generated_data/msmarco/msmarco_hardneg.json\
    --msmarco_collection_file generated_data/msmarco/msmarco_collections.json \
    --squadv2_train_file generated_data/squadv2/squadv2_hardneg.json\
    --squadv2_collection_file generated_data/squadv2/squadv2_collections.json \
    --output_dir result/simcse_zalo_msmarco_squadv2_avg_0.7 \
    --do_mlm False \
    --hard_negative_weight 0.7 \
    --num_train_epoch 10 \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_seq_length 256 \
    --evaluation_strategy steps \
    --metric_for_best_model acc_top_1 \
    --eval_steps 4400 \
    --save_steps 4400 \
    --logging_steps 50 \
    --pooler_type avg \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train True \
    --do_eval \
    --fp16 \
    "$@"
