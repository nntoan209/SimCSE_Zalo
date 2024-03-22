from FlagEmbedding import BGEM3FlagModel
import numpy as np
import os
import json
import torch
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
parser.add_argument("--save_dir", type=str)

args = parser.parse_args()

if __name__ == "__main__":

    corpus = json.load(open(args.corpus_file, encoding='utf-8'))
    dev_queries = json.load(open(args.dev_queries_file, encoding='utf-8'))
    dev_rel_docs = json.load(open(args.dev_rel_docs_file, encoding='utf-8'))

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
                            use_fp16=True,
                            device='cuda') # Setting use_fp16 to True speeds up computation with a slight performance degradation
        
        print(f"Eval {model_path}")

        print("Documents Embedding ...")
        sentences_embedding = model.encode(sentences=list(corpus.values()),
                                        batch_size=args.passage_batch_size,
                                        max_length=args.passage_max_length)['dense_vecs']
        sentences_embedding = torch.from_numpy(sentences_embedding).to(torch.float32)

        print("Dev Queries Embedding ...")
        queries_dev_embedding = model.encode(sentences=list(dev_queries.values()),
                                            batch_size=args.query_batch_size,
                                            max_length=args.query_max_length)['dense_vecs']
        queries_dev_embedding = torch.from_numpy(queries_dev_embedding).to(torch.float32)

        print("Semantic Search ...")
        results_semantic_search = util.semantic_search(queries_dev_embedding, sentences_embedding, top_k=100) #chunk

        ### Convert results
        qids = list(dev_queries.keys())
        pids = list(corpus.keys())

        converted_results = {}
        for idx, result in enumerate(results_semantic_search):
                for answer in result:
                    answer['corpus_id'] = pids[answer['corpus_id']]
                converted_results[qids[idx]] = result

        ### Get passage from chunk results 
        only_pidqids_results = {}
        for qid, result in converted_results.items():
            only_pidqids_results[qid] = [" ".join(answer['corpus_id'].split()).strip() for answer in result]

        ### Compute metrics  parameters
        mrr_at_k = [5, 10, 100]
        accuracy_at_k = [1, 10, 100]
        precision_recall_at_k = [1, 10, 100]
        map_at_k = [5, 10, 100]

        metrics = calculate_metrics(only_pidqids_results=only_pidqids_results,
                                    dev_rel_docs=dev_rel_docs,
                                    dev_queries=dev_queries,
                                    mrr_at_k=mrr_at_k,
                                    accuracy_at_k=accuracy_at_k,
                                    precision_recall_at_k=precision_recall_at_k,
                                    map_at_k=map_at_k) 

        with open(args.save_dir, "a") as f:
            f.write(f"{model_path} {args.passage_max_length}\n")
            for key, value in metrics.items():
                f.write(f"\t{key}: {value}\n")
