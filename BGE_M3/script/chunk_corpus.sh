#!/bin/bash

export JAVA_HOME='/usr/lib/jvm/jre-11-openjdk'
export JVM_PATH='/usr/lib/jvm/jre-11-openjdk/lib/server/libjvm.so'

python FlagEmbedding/eval/chunk_corpus.py \
    --corpus_path FlagEmbedding/data/final/corpus/merged_dedup_corpus_indexed.json \
    --chunk_size 256 \
    --chunk_overlap 10 \
    --save_path FlagEmbedding/data/final/corpus/merged_dedup_chunked_corpus_indexed.json \
