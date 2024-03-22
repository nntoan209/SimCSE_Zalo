#!/bin/bash

export JAVA_HOME='/usr/lib/jvm/jre-11-openjdk'
export JVM_PATH='/usr/lib/jvm/jre-11-openjdk/lib/server/libjvm.so'

python FlagEmbedding/eval/segment_query.py \
    --dev_queries_path FlagEmbedding/data/final/test/merged_dev_queries.json \
    --save_path FlagEmbedding/data/final/test/merged_segmented_dev_queries.json \
