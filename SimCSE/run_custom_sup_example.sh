#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=2

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train_custom.py \
    --seed 2023 \
    --model_name_or_path vinai/phobert-base-v2 \
    --dataset_type msmarco \
    --train_file generated_data/zalo_msmarco.csv \
    --collection_file data/msmarco_vietnamese/collection_translated_tokenized.tsv \
    --output_dir result/supervise-simcse-phobert-base-v2 \
    --do_mlm False \
    --hard_negative_weight 0.4 \
    --num_train_epoch 3 \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_seq_length 256 \
    --evaluation_strategy steps \
    --metric_for_best_model acc_top_1 \
    --eval_steps 3000 \
    --save_steps 3000 \
    --logging_steps 200 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
