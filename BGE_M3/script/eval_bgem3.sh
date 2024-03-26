#!/bin/bash

export CUDA_VISIBLE_DEVICES="6,7"
export OMP_NUM_THREADS=8

python3 BGE_M3/eval/eval_bgem3.py \
    --model_savedir saved_models/bgem3_unified_finetune \
    --query_max_length 128 \
    --query_batch_size 256 \
    --passage_max_length 8192 \
    --passage_batch_size 4 \
    --corpus_file data/final/corpus/merged_dedup_corpus_indexed.json \
    --dev_queries_file data/final/test/merged_dev_queries.json \
    --dev_rel_docs_file data/final/test/merged_dev_rel_docs.json \
    --save_dir BGE_M3/results.txt \
    --colbert_rerank \
    "$@"
