#!/bin/bash

python split_data_by_length.py \
    --input_path train_data \
    --output_dir train_data_split \
    --cache_dir .cache \
    --log_name .split_log \
    --length_list 0 500 1000 2000 3000 4000 5000 6000 7000 \
    --model_name_or_path BAAI/bge-m3 \
    --num_proc 16 \
    --overwrite False \
    "$@"