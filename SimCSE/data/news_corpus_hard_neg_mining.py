import os
import json
import argparse
import numpy as np
import py_vncorenlp
import pandas as pd
from tqdm import tqdm
from rank_bm25 import *
from utils import bm25_tokenizer
from sentence_transformers import SentenceTransformer

def cosine_similarity(query_embedding, document_embeddings):
    # Calculate the dot product between the query and all documents
    dot_products = np.dot(document_embeddings, query_embedding)
    
    # Calculate the L2 (Euclidean) norm of the query and all documents
    query_norm = np.linalg.norm(query_embedding)
    document_norms = np.linalg.norm(document_embeddings, axis=1)
    
    # Calculate the cosine similarity by dividing the dot products by (query_norm * document_norms)
    similarities = dot_products / (query_norm * document_norms)
    
    return similarities

def get_rank_id(scores):
    sorted_indices = np.argsort(-scores)
    ranked_indices = np.argsort(sorted_indices) + 1
    return ranked_indices
    
def final_chunk_scores(bm25_scores, sbert_scores, alpha: float = 0.8):
    """
    bm25_scores: numpy array of shape (n, ) where n is the number of documents
    sbert_scores: numpy array of shape (n, ) where n is the number of documents
    alpha: hyperparameter for combining the score of 2 models
    """    
    
    assert (bm25_scores is not None) or (sbert_scores is not None)
    
    sbert_rank_id = 1
    bm25_rank_id = 1
    
    if sbert_scores is not None:
        sbert_rank_id = get_rank_id(sbert_scores)
        
    if bm25_scores is not None:
        bm25_rank_id = get_rank_id(bm25_scores)
        
    final_chunk_scores = alpha * (1 / bm25_rank_id) + (1 - alpha) * (1 / sbert_rank_id)
    
    return final_chunk_scores

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./SimCSE/data/news_corpus", type=str, help="path to training data")
    parser.add_argument("--save_dir", default="./SimCSE/generated_data/news_corpus", type=str, help="path to save data")
    parser.add_argument('--num_hard_neg', type=int, default=50, help="Number of hard negatives for each sample")
    parser.add_argument('--continue_mining', action="store_true", help="Continue mining hard negatives")

    args = parser.parse_args()
    
    os.makedirs(args.save_dir,exist_ok=True)
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
    # with open("SimCSE/generated_data/hf_token.key") as f:
    #     hf_key = f.read()
    # sbert_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', use_auth_token=hf_key)
    
    # Create a dictionary to store unique passages with IDs
    if args.continue_mining:
        print("Continue mining")
        with open(os.path.join(args.save_dir, "news_corpus_collections.json"), "r", encoding="utf-8") as f:
            passage_dict = json.load(f)
            passage_dict = {v: k for k, v in passage_dict.items()}
            
            passage_id = int(len(passage_dict))

        with open(os.path.join(args.save_dir, "news_corpus_hardneg.json"), "r", encoding="utf-8") as f:
            new_hardneg_dict = json.load(f)
            
            hardneg_dict = {}
            for idx, item in tqdm(new_hardneg_dict.items(), desc=f"Reformatting"):
                new_key = item['query']
                new_value = {"pos": item["pos"],
                            "hard_neg": item["hard_neg"]}
                hardneg_dict[new_key] = new_value
       
    else:
        passage_dict = {}
        passage_id = 0
        
        hardneg_dict = {}
    
    for file in sorted(os.listdir(args.data_dir))[9:11]:
        file_path = os.path.join(args.data_dir, file)
        df = pd.read_parquet(file_path, engine='auto')
        df = df[['sen_1', 'sen_2']]
        df.drop_duplicates(keep='first', inplace=True)
        df.reset_index(inplace=True, drop=True)
        
        df['sen_1'] = df['sen_1'].apply(lambda x: " ".join(rdrsegmenter.word_segment(x)))
        df['sen_2'] = df['sen_2'].apply(lambda x: " ".join(rdrsegmenter.word_segment(x)))
        
        for passage in tqdm(df['sen_2'].unique(), desc=f"{file} Creating collections"):
            if passage not in passage_dict:
                passage_dict[passage] = str(passage_id)
                passage_id += 1

        for index, row in tqdm(df.iterrows(), desc=f"{file} Creating positive pairs"):
            
            query = row['sen_1']
            passage = row['sen_2']
            
            passage_id_hardneg = passage_dict[passage]
            
            if query not in hardneg_dict:
                hardneg_dict[query] = {'pos': [], 'hard_neg': []}
                
            if passage_id_hardneg not in hardneg_dict[query]['pos']:
                hardneg_dict[query]['pos'].append(passage_id_hardneg)
                
        # print("Calculating embeddings for all passages")
        all_passages = df['sen_2'].values
        # all_passages_embeddings = sbert_model.encode(list(all_passages), batch_size=1024, show_progress_bar=True, device='cuda')
        
        documents = [bm25_tokenizer(passage) for passage in all_passages]
        bm25 = BM25Plus(documents, k1=0.4, b=0.6)
        
        for index, row in tqdm(df.iterrows(), desc=f"{file} Creating hard negative paris"):
            
            query = row['sen_1']

            # filtered_df = df[df['sen_1'] != query]
            # filtered_passage_ids = filtered_df.index.values

            # Calculate the BM25 scores only for the chunks outside of the relevant articles
            tokenized_query = bm25_tokenizer(query)
            chunk_bm25_scores = bm25.get_scores(tokenized_query)#[filtered_passage_ids] 
            
            # # Calculate the SBERT scores only for the chunks outside of the relevant articles
            # question_embedding = sbert_model.encode(query, device="cuda")
            # chunk_embeddings = all_passages_embeddings#[filtered_passage_ids]
            # chunk_sbert_scores = cosine_similarity(query_embedding=question_embedding, document_embeddings=chunk_embeddings)

            # Combine these 2 scores to get the final chunks score
            chunk_scores = final_chunk_scores(bm25_scores=chunk_bm25_scores, sbert_scores=None)
            
            top_k_indices = np.argsort(chunk_scores)[::-1][:args.num_hard_neg]
            # top_k_passage_ids = filtered_passage_ids[top_k_indices]
            top_k_passages = all_passages[top_k_indices]
            
            hard_neg_ids = [passage_dict[p] for p in top_k_passages if passage_dict[p] not in hardneg_dict[query]['pos']]

            hardneg_dict[query]['hard_neg'].extend(hard_neg_ids)

        # save checkpoint
        passage_dict_checkpoint = {v: k for k, v in passage_dict.items()}
        with open(os.path.join(args.save_dir, "news_corpus_collections_checkpoint.json"), "w", encoding="utf-8") as f:
            json.dump(passage_dict_checkpoint, f)

        hardneg_dict_checkpoint = {}
        idx_checkpoint = 0
        
        for query, pos_hardneg in hardneg_dict.items():
            new_key = str(idx_checkpoint)
            new_value = {"query": query,
                        "pos": pos_hardneg["pos"],
                        "hard_neg": pos_hardneg["hard_neg"]}
            hardneg_dict_checkpoint[new_key] = new_value
            
            idx_checkpoint += 1
            
        with open(os.path.join(args.save_dir, "news_corpus_hardneg_checkpoint.json"), "w", encoding="utf-8") as f:
            json.dump(hardneg_dict_checkpoint, f)
        
        print("\n\n")
        
    passage_dict = {v: k for k, v in passage_dict.items()}
    with open(os.path.join(args.save_dir, "news_corpus_collections.json"), "w", encoding="utf-8") as f:
        json.dump(passage_dict, f)

    new_hardneg_dict = {}
    idx = 0
    
    for query, pos_hardneg in tqdm(hardneg_dict.items(), desc=f"{file} Reformatting"):
        new_key = str(idx)
        new_value = {"query": query,
                    "pos": pos_hardneg["pos"],
                    "hard_neg": pos_hardneg["hard_neg"]}
        new_hardneg_dict[new_key] = new_value
        
        idx += 1
        
    with open(os.path.join(args.save_dir, "news_corpus_hardneg.json"), "w", encoding="utf-8") as f:
        json.dump(new_hardneg_dict, f)
