import json
import os
import pandas as pd
from tqdm import tqdm
import argparse
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./SimCSE/data/msmarco_vietnamese", type=str, help="path to training data")
    parser.add_argument("--save_dir", default="./SimCSE/generated_data", type=str, help="path to training data")
    parser.add_argument("--hard_neg_model", default='bm25', type=str, help="method for mining hard negatives",
                        choices=['bm25', 'msmarco-distilbert-base-tas-b', 'msmarco-distilbert-base-v3',
                                 'msmarco-MiniLM-L-6-v3', 'distilbert-margin_mse-cls-dot-v2', 'distilbert-margin_mse-cls-dot-v1',
                                 'distilbert-margin_mse-mean-dot-v1', 'mpnet-margin_mse-mean-v1', 'co-condenser-margin_mse-cls-v1',
                                 'distilbert-margin_mse-mnrl-mean-v1', 'distilbert-margin_mse-sym_mnrl-mean-v1',
                                 'distilbert-margin_mse-sym_mnrl-mean-v2', 'co-condenser-margin_mse-sym_mnrl-mean-v1'])
    args = parser.parse_args()
    
    os.makedirs(args.save_dir,exist_ok=True)
    zalo_data_path = os.path.join(args.save_dir, "zalo_3_hardneg.csv")
    queries_path = os.path.join(args.data_dir, "queries_translated_tokenized.train.tsv")
    collection_path = os.path.join(args.data_dir, "collection_translated_tokenized.tsv")
    id_triples_path = os.path.join(args.data_dir, "msmarco_hard_negatives.jsonl")
    output_path = os.path.join(args.save_dir, f"zalo_msmarco_3_hardneg.csv")

    zalo_data = pd.read_csv(zalo_data_path, encoding='utf-8')
    
    print("Reading queries data...")
    queries_df = pd.read_csv(queries_path,
                            sep='\t', header=None, names=['id', 'text'], encoding='utf-8', index_col='id')
    print("Reading collections data...")
    collection_df = pd.read_csv(collection_path,
                            sep='\t', header=None, names=['id', 'text'], encoding='utf-8', index_col='id')        
                    
    print("Creating MSMARCO data...")
    sent0s = []
    sent1s = []
    hard_neg_0s = []
    hard_neg_1s = []
    hard_neg_2s = []

    with open(id_triples_path) as file:
        for line in tqdm(file):
            data = json.loads(line)
            
            sent0_id, sent1_ids = data['qid'], data['pos'] 
            if not sent1_ids:
                continue                    
            hard_neg_ids = data['neg'].get(args.hard_neg_model, None)
            if hard_neg_ids is None:
                hard_neg_ids = data['neg'].get('msmarco-distilbert-base-v3')
                
            sent1_id = random.choice(sent1_ids)
            hard_neg_0_id, hard_neg_1_id, hard_neg_2_id = random.sample(hard_neg_ids, 3)
            
            sent0s.append(queries_df['text'][sent0_id])
            sent1s.append(collection_df['text'][sent1_id])
            hard_neg_0s.append(collection_df['text'][hard_neg_0_id])
            hard_neg_1s.append(collection_df['text'][hard_neg_1_id])
            hard_neg_2s.append(collection_df['text'][hard_neg_2_id])
            
            
        new_msmarco_data = pd.DataFrame({
            "sent0": sent0s,
            "sent1": sent1s,
            "hard_neg_0": hard_neg_0s,
            "hard_neg_1": hard_neg_1s,
            "hard_neg_2": hard_neg_2s,
        })
        
        zalo_data = pd.concat((zalo_data, new_msmarco_data), ignore_index=True)
    
    print("Saving data...")   
    # zalo_data = zalo_data.sample(frac=1)
    zalo_data.to_csv(output_path, index=False, encoding='utf-8')
        