#!/bin/bash

python3 split_data.py \
    --input_path data/final/train/merged_data_train_minedHN_fulltext.jsonl \
    --output_dir data/final/train/splitted_data \
    --cache_dir data/cache \
    --log_name .split_log \
    --length_list 0 512 1024 2048 4096 \
    --model_name_or_path BAAI/bge-m3 \
    --num_proc 16 \
    "$@"