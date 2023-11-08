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
    
def final_chunk_scores(bm25_scores, sbert_scores, alpha: float = 0.4):
    """
    bm25_scores: numpy array of shape (n, ) where n is the number of documents
    sbert_scores: numpy array of shape (n, ) where n is the number of documents
    alpha: hyperparameter for combining the score of 2 models
    """
    bm25_rank_id = get_rank_id(bm25_scores)
    sbert_rank_id = get_rank_id(sbert_scores)
    
    final_chunk_scores = alpha * (1 / bm25_rank_id) + (1 - alpha) * (1 / sbert_rank_id)
    
    return final_chunk_scores

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./SimCSE/data/zalo_legal_2021", type=str, help="path to training data")
    parser.add_argument("--save_dir", default="./SimCSE/generated_data", type=str, help="path to training data")
    parser.add_argument('-l','--num_hard_neg', type=int, nargs='+', required=True, help="Number of hard negatives for each sample")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir,exist_ok=True)
    train_qa_pairs_path = os.path.join(args.data_dir, "original/train_question_answer.json")
    train_split_path = os.path.join(args.data_dir, "processed/queries_train.json")
    corpus_256_path = os.path.join(args.save_dir, "corpus_256.csv")
    
    train_qa_pairs = json.load(open(train_qa_pairs_path, encoding='utf-8'))['items']
    train_split = json.load(open(train_split_path, encoding='utf-8'))
    corpus_256 = pd.read_csv(corpus_256_path, encoding='utf-8')
    
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
    with open("SimCSE/generated_data/hf_token.key") as f:
        hf_key = f.read()
    sbert_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', use_auth_token=hf_key)
    
    sent0s = []
    sent1s = []
    
    print("Calculating embeddings for all chunks")
    all_chunks = corpus_256['text'].values
    all_chunk_embeddings = sbert_model.encode(list(all_chunks), batch_size=256, show_progress_bar=True)
    
    print("Creating positive pairs")
    i = 0
    for question_id, question in tqdm(train_split.items()):
        question = " ".join(rdrsegmenter.word_segment(question))
        annotation = train_qa_pairs[i]
        assert question_id == annotation['question_id']
        
        # For each relevant article, choose 1 chunk to be the positive sample
        for article in annotation['relevant_articles']:
            chunks_idx = corpus_256[(corpus_256["law_id"]==article["law_id"]) & (corpus_256["article_id"]==int(article["article_id"]))].index.values
            chunks = all_chunks[chunks_idx]
            # If the article only has 1 chunk, it is the positive sample
            if len(chunks) == 1:
                positive_sample = chunks[0]
            
            # Else, the chunks must be ranked
            else:
                # Calculate bm25 scores only for the chunks in the article
                documents = [bm25_tokenizer(chunk) for chunk in chunks]
                bm25 = BM25Plus(corpus=documents, k1=0.4, b=0.6)
                
                tokenized_question = bm25_tokenizer(question)
                chunk_bm25_scores = bm25.get_scores(tokenized_question)
                
                # Calculate the SBERT scores only for the chunks in the article
                question_embedding = sbert_model.encode(question)
                chunk_embeddings = all_chunk_embeddings[chunks_idx]
                chunk_sbert_scores = cosine_similarity(query_embedding=question_embedding, document_embeddings=chunk_embeddings)
                
                # Combine these 2 scores to get the final chunks score
                chunk_scores = final_chunk_scores(bm25_scores=chunk_bm25_scores, sbert_scores=chunk_sbert_scores)
                
                # For positive sample, we get the chunk with the highest score
                max_idx = np.argmax(chunk_scores)
                positive_sample = all_chunks[chunks_idx[max_idx]]
                
            sent0s.append(question)
            sent1s.append(positive_sample)
        i += 1
    
    for n in args.num_hard_neg:
        output_path = os.path.join(args.save_dir, f"train_data_{n}_hardneg.csv")
        hard_negs = {f"hard_neg_{i}": [] for i in range(n)}
        
        print(f"Mining hard negative sample for n = {n}")
        idx = 0
        idx_in_df = 0
        
        documents = [bm25_tokenizer(chunk) for chunk in all_chunks]
        bm25 = BM25Plus(documents, k1=0.4, b=0.6)
        
        for question_id, question in tqdm(train_split.items()):
            question = " ".join(rdrsegmenter.word_segment(question))
            annotation = train_qa_pairs[idx]
            assert question_id == annotation['question_id']
            
            # For each question, find the chunk outside of the relevant articles with high similarity to be the negative sample
            # The number of negative samples will be equal to the number of relevant articles * args.num_hard_neg
            num_relevant_articles = len(annotation['relevant_articles'])
            num_hard_negatives = num_relevant_articles * n
            
            filtered_list = []
            for article in annotation['relevant_articles']:
                filtered_list.append((article['law_id'], article['article_id']))
                
                if article['law_id'] == "15/2019/tt-byt" and article['article_id'] == "26":
                    filtered_list.append(("11/2016/tt-byt", "17"))
                elif article['law_id'] == "05/2021/nđ-cp" and article['article_id'] == "23":
                    filtered_list.append(("56/2019/nđ-cp", "14"))   
                elif article['law_id'] == "115/2020/nđ-cp" and article['article_id'] == "6":
                    filtered_list.append(("138/2020/nđ-cp", "5"))    
                elif article['law_id'] == "28/2019/nđ-cp" and article['article_id'] == "12":
                    filtered_list.append(("220/2013/nđ-cp", "12"))    
                elif article['law_id'] == "05/2021/nđ-cp" and article['article_id'] == "23":
                    filtered_list.append(("56/2019/nđ-cp", "14"))    
                    
            exclusion_conditions = [(corpus_256['law_id'] == law_id) & (corpus_256['article_id'] == int(article_id))\
                                    for law_id, article_id in filtered_list]
            combined_condition = ~np.any(exclusion_conditions, axis=0)
            chunks_idx = corpus_256[combined_condition].index.values
            chunks = all_chunks[chunks_idx]
            
            # Calculate the BM25 scores only for the chunks outside of the relevant articles
            tokenized_question = bm25_tokenizer(question)
            chunk_bm25_scores = bm25.get_scores(tokenized_question)[chunks_idx]
            
            # Calculate the SBERT scores only for the chunks outside of the relevant articles
            question_embedding = sbert_model.encode(question)
            chunk_embeddings = all_chunk_embeddings[chunks_idx]
            chunk_sbert_scores = cosine_similarity(query_embedding=question_embedding, document_embeddings=chunk_embeddings)
            
            # Combine these 2 scores to get the final chunks score
            chunk_scores = final_chunk_scores(bm25_scores=chunk_bm25_scores, sbert_scores=chunk_sbert_scores)
            
            # For hard negative samples, we choose the top k = num_hard_negatives chunks with the highest scores
            top_k_indices = np.argsort(chunk_scores)[::-1][:num_hard_negatives]
            for i in range(num_relevant_articles):
                for j in range(n):
                    hard_negs[f'hard_neg_{j}'].append(all_chunks[chunks_idx[top_k_indices[i + j * num_relevant_articles]]])
                idx_in_df += 1
                    
            idx += 1
        
        assert (len(sent0s) == len(sent1s)) and (len(sent0s) == len(hard_negs['hard_neg_0']))
        dataframe = {
            "sent0": sent0s,
            "sent1": sent1s,
        }
        dataframe.update(hard_negs)
        
        train_data_df = pd.DataFrame(dataframe)
        train_data_df = train_data_df.sample(frac=1)
        train_data_df.to_csv(output_path, index=False)
