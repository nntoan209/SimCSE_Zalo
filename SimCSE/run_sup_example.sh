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
# python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
python train.py \
    --model_name_or_path vinai/phobert-base-v2 \
    --train_file generated_data/train_data.csv \
    --output_dir result/supervise-simcse-phobert-base-v2 \
    --do_mlm True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --max_seq_length 256 \
    --evaluation_strategy steps \
    --metric_for_best_model acc_top_10 \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
