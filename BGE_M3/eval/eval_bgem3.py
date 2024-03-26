from BGE_M3.src.utils import BGEM3FlagModel
import numpy as np
import os
import json
from typing import List
import torch
from tqdm import tqdm
from sentence_transformers import util
from argparse import ArgumentParser

parser = ArgumentParser(description='Eval law using BGE-M3')
parser.add_argument("--model_savedir", type=str, required=True, help='model(s) save dir')
parser.add_argument("--query_max_length", type=int, default=128, help='query max length')
parser.add_argument("--query_batch_size", type=int, default=128, help='query batch size')
parser.add_argument("--passage_max_length", type=int, default=8192, help='passage max length')
parser.add_argument("--passage_batch_size", type=int, default=2, help='passage batch size')
parser.add_argument("--corpus_file", type=str, help="path to the corpus file")
parser.add_argument("--dev_queries_file", type=str, help="path to the dev queries file")
parser.add_argument("--dev_rel_docs_file", type=str, help="path to the dev relevant documents file")
parser.add_argument("--sparse_hybrid", action='store_true', help="use sparse mode for hybrid retrieval")
parser.add_argument("--sparse_weight", type=float, default=0.1, help="sparse weight for hybrid retrieval")
parser.add_argument("--colbert_rerank", action='store_true', help="rerank with colbert")
parser.add_argument("--save_dir", type=str)

args = parser.parse_args()

if __name__ == "__main__":

    corpus = json.load(open(args.corpus_file, encoding='utf-8'))
    dev_queries = json.load(open(args.dev_queries_file, encoding='utf-8'))
    dev_rel_docs = json.load(open(args.dev_rel_docs_file, encoding='utf-8'))

    def convert_score_to_rank(scores: List):
        sorted_indices = np.argsort(-scores)
        ranked_indices = np.argsort(sorted_indices) + 1
        return ranked_indices
    
    def rrf_score(dense_ranks, sparse_ranks, sparse_weight, k):
        return 1 / (dense_ranks + k) \
             + 1 / (sparse_ranks + k) * sparse_weight

    def calculate_metrics(only_pidqids_results, dev_rel_docs, dev_queries,  mrr_at_k, accuracy_at_k, precision_recall_at_k, map_at_k):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in accuracy_at_k}
        precisions_at_k = {k: [] for k in precision_recall_at_k}
        recall_at_k = {k: [] for k in precision_recall_at_k}
        MRR = {k: 0 for k in mrr_at_k}
        AveP_at_k = {k: [] for k in map_at_k}

        for pid in only_pidqids_results.keys(): #qid : [pid1, pid2, pid3, ...pid 4]
                                                #label: qid: [positive pid1, pid2, ..]
            top_hits = only_pidqids_results[pid]
            query_relevant_docs = dev_rel_docs[pid]  # only one right now

            # Accuracy @k
            for k_val in accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in precision_recall_at_k:
                num_correct = 0
                for hit in list(dict.fromkeys(top_hits[0:k_val])):
                    if hit in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in mrr_at_k:
                for rank, hit in enumerate(list(dict.fromkeys(top_hits[0:k_val]))):
                    if hit in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # MAP@k
            for k_val in map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(list(dict.fromkeys(top_hits[0:k_val]))):
                    if hit in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)


        ## Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(dev_queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in MRR:
            MRR[k] /= len(dev_queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])


        ## For logging
        scores = {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'mrr@k': MRR, 'map@k': AveP_at_k}
        
        final_result = {}
        for k in scores['accuracy@k']:
            final_result[f'eval_acc_top_{k}'] = scores['accuracy@k'][k]*100
        for k in scores['precision@k']:
            final_result[f'eval_precision_top_{k}'] = scores['precision@k'][k]*100
        for k in scores['recall@k']:
            final_result[f'eval_recall_top_{k}'] = scores['recall@k'][k]*100
        for k in scores['mrr@k']:
            final_result[f'eval_mrr_top_{k}'] = scores['mrr@k'][k]
        for k in scores['map@k']:
            final_result[f'eval_map_top_{k}'] = scores['map@k'][k]
            
        return final_result

    if args.model_savedir != "BAAI/bge-m3":
        assert os.path.isdir(args.model_savedir)
    if args.model_savedir == "BAAI/bge-m3" \
    or args.model_savedir.split("/")[-1].startswith("checkpoint"): # the original model or a specific checkpoint
        models_list = [args.model_savedir]
    else: # list of checkpoints
        models_list = [os.path.join(args.model_savedir, path) for path in os.listdir(args.model_savedir) if path.startswith("checkpoint")]
    for model_path in models_list:
        model = BGEM3FlagModel(model_path,  
                            pooling_method='cls',
                            use_fp16=True, # Setting use_fp16 to True speeds up computation with a slight performance degradation
                            device=None) # Use device=None to use the default device (cuda if available, else cpu)
        
        print(f"Eval {model_path}")

        print("Documents Embedding ...")
        sentences_embedding = model.encode(sentences=list(corpus.values()),
                                        batch_size=args.passage_batch_size,
                                        max_length=args.passage_max_length,
                                        return_sparse=args.sparse_hybrid,
                                        return_colbert_vecs=args.colbert_rerank)

        print("Dev Queries Embedding ...")
        queries_dev_embedding = model.encode(sentences=list(dev_queries.values()),
                                            batch_size=args.query_batch_size,
                                            max_length=args.query_max_length,
                                            return_sparse=args.sparse_hybrid,
                                            return_colbert_vecs=args.colbert_rerank)

        only_pidqids_results = {}
        qids = list(dev_queries.keys())
        pids = list(corpus.keys())

        if not args.sparse_hybrid: # only dense retrieval
            print("Dense Search ...")
            sentences_embedding_dense = torch.from_numpy(sentences_embedding['dense_vecs']).to(torch.float32)
            queries_dev_embedding_dense = torch.from_numpy(queries_dev_embedding['dense_vecs']).to(torch.float32)

            results_dense_search = util.semantic_search(queries_dev_embedding_dense, sentences_embedding_dense, top_k=100) #chunk

            converted_results = {}
            for idx, result in enumerate(results_dense_search):
                    for answer in result:
                        answer['corpus_id'] = pids[answer['corpus_id']]
                    converted_results[qids[idx]] = result

            ### Get passage from chunk results 
            only_pidqids_results = {}
            for qid, result in converted_results.items():
                only_pidqids_results[qid] = [answer['corpus_id'].strip() for answer in result]

        else: # hybrid retrieval with dense and sparse
            print("Hybrid Search ...")

            # compute dense ranking
            sentences_embedding_dense = sentences_embedding['dense_vecs'].astype(np.float32)
            queries_dev_embedding_dense = queries_dev_embedding['dense_vecs'].astype(np.float32)

            dense_scores = model.dense_score(q_reps=queries_dev_embedding_dense, p_reps=sentences_embedding_dense)

            dense_ranks = np.apply_along_axis(convert_score_to_rank, 1, dense_scores)

            # compute sparse ranking
            sentences_embedding_sparse = sentences_embedding['lexical_weights'] # list of lexical weights
            queries_dev_embedding_sparse = queries_dev_embedding['lexical_weights'] # list of lexical weights

            sparse_scores = np.zeros((len(queries_dev_embedding_sparse), len(sentences_embedding_sparse)))

            for i in range(len(queries_dev_embedding_sparse)):
                for j in range(len(sentences_embedding_sparse)):
                    sparse_scores[i][j] = model.lexical_matching_score(queries_dev_embedding_sparse[i], sentences_embedding_sparse[j])

            sparse_ranks = np.apply_along_axis(convert_score_to_rank, 1, sparse_scores)

            # compute rrf score
            rrf_scores = rrf_score(dense_ranks, sparse_ranks, args.sparse_weight, k=10)

            for idx, qid in enumerate(qids):
                rrf_scores_for_qid = rrf_scores[idx]
                top_k = np.argsort(-rrf_scores_for_qid)[:100]
                only_pidqids_results[qid] = [pids[i] for i in top_k]

        ### Compute metrics  parameters
        mrr_at_k = [5, 10, 100]
        accuracy_at_k = [1, 5, 10, 100]
        precision_recall_at_k = [1, 5, 10, 100]
        map_at_k = [5, 10, 100]

        metrics = calculate_metrics(only_pidqids_results=only_pidqids_results,
                                    dev_rel_docs=dev_rel_docs,
                                    dev_queries=dev_queries,
                                    mrr_at_k=mrr_at_k,
                                    accuracy_at_k=accuracy_at_k,
                                    precision_recall_at_k=precision_recall_at_k,
                                    map_at_k=map_at_k) 

        with open(args.save_dir, "a") as f:
            if args.colbert_rerank:
                f.write(f"{model_path} Colbert rerank \n\tBefore rerank:\n")
            else:
                f.write(f"{model_path}\n")
            for key, value in metrics.items():
                indent = '\t' * (1 + args.colbert_rerank)
                f.write(f"{indent}{key}: {value}\n")

        # rerank with Colbert
        if args.colbert_rerank:

            sentences_embedding_colbert = sentences_embedding['colbert_vecs'] # list of colbert vecs
            queries_dev_embedding_colbert = queries_dev_embedding['colbert_vecs'] # list of colbert vecs

            rerank_only_pidqids_results = {}
            colbert_embedded_corpus = {k: v for k,v in zip(pids, sentences_embedding_colbert)}
            colbert_embedded_queries = {k: v for k,v in zip(qids, queries_dev_embedding_colbert)}

            for qid in tqdm(only_pidqids_results.keys(), desc="Rerank with Colbert",
                            disable=len(only_pidqids_results.keys()) < 5):
                
                colbert_scores_for_qid = [model.colbert_score(q_reps=colbert_embedded_queries[qid], p_reps=colbert_embedded_corpus[pid]) \
                                          for pid in only_pidqids_results[qid]]
                
                pids_with_colbert_scores_for_qid = list(zip(only_pidqids_results[qid], colbert_scores_for_qid))

                rerank_pids_with_colbert_scores_for_qid = sorted(pids_with_colbert_scores_for_qid, key=lambda x: x[1], reverse=True)
                
                rerank_only_pidqids_results[qid] = [answer[0] for answer in rerank_pids_with_colbert_scores_for_qid]

            ### Compute rerank metrics  parameters
            mrr_at_k = [5, 10, 100]
            accuracy_at_k = [1, 5, 10, 100]
            precision_recall_at_k = [1, 5, 10, 100]
            map_at_k = [5, 10, 100]

            rerank_metrics = calculate_metrics(only_pidqids_results=rerank_only_pidqids_results,
                                        dev_rel_docs=dev_rel_docs,
                                        dev_queries=dev_queries,
                                        mrr_at_k=mrr_at_k,
                                        accuracy_at_k=accuracy_at_k,
                                        precision_recall_at_k=precision_recall_at_k,
                                        map_at_k=map_at_k) 

            with open(args.save_dir, "a") as f:
                f.write(f"\tAfter rerank:\n")
                for key, value in rerank_metrics.items():
                    f.write(f"\t\t{key}: {value}\n")
    with open(args.save_dir, "a") as f:
        f.write(f"---------------------------------------------------------------------------------------------\n\n")