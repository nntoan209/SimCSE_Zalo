#!/bin/bash

python3 -m text_dedup.minhash \
  --path "data/merged" \
  --split "train" \
  --cache_dir "./cache" \
  --output "data/merged_dedup" \
  --output_cluster "data/merged_dedup/cluster" \
  --column "text" \
  --batch_size 10000