#!/bin/bash

export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export CUDA_VISIBLE_DEVICES="0,1"

python3 -m BGE_M3.src.hn_mine \
    --model_name_or_path BAAI/bge-m3 \
    --input_file data/final/merged_data_train.jsonl \
    --corpus_file data/final/merged_dedup_corpus_indexed.json \
    --output_file data/final/merged_data_train_minedHN.jsonl \
    --range_for_sampling 5-40 \
    --negative_number 20 \
    --use_gpu_for_searching \
    "$@"