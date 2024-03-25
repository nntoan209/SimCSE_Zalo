#!/bin/bash

# export CUDA_VISIBLE_DEVICES="6,7"

python3 BGE_M3/eval/eval_bgem3_dense_rerank_colbert.py \
    --model_savedir BAAI/bge-m3 \
    --query_max_length 128 \
    --query_batch_size 512 \
    --passage_max_length 8192 \
    --passage_batch_size 8 \
    --corpus_file data/final/corpus/merged_dedup_corpus_indexed.json \
    --dev_queries_file data/final/test/merged_dev_queries.json \
    --dev_rel_docs_file data/final/test/merged_dev_rel_docs.json \
    --save_dir results.txt \