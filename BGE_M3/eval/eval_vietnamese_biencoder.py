import numpy as np
import json
import torch
from sentence_transformers import util
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

parser = ArgumentParser(description='Eval law using Vietnamese Bi-encoder')
parser.add_argument("--query_batch_size", type=int, default=128, help='query batch size')
parser.add_argument("--passage_batch_size", type=int, default=2, help='passage batch size')
parser.add_argument("--corpus_file", type=str, help="path to the corpus file")
parser.add_argument("--dev_queries_file", type=str, help="path to the dev queries file")
parser.add_argument("--dev_rel_docs_file", type=str, help="path to the dev relevant documents file")
parser.add_argument("--save_dir", type=str, help="path to save results")

args = parser.parse_args()

if __name__ == "__main__":

    corpus = json.load(open(args.corpus_file, encoding='utf-8'))
    dev_queries = json.load(open(args.dev_queries_file, encoding='utf-8'))
    dev_rel_docs = json.load(open(args.dev_rel_docs_file, encoding='utf-8'))

    model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder', use_safetensors=True)
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_texts(texts, batch_size):
        encoded_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]

            encoded_input = tokenizer(batch_texts, max_length=256, padding=True, truncation=True, return_tensors='pt').to("cuda")

            with torch.no_grad():
                model_output = model(**encoded_input)
        
            pooled_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            encoded_embeddings.append(pooled_embeddings)

        encoded_embeddings = torch.cat(encoded_embeddings, dim=0)
        return encoded_embeddings

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
            recall_at_k[k] = np.mean(recall_at_k[k] * k)

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

    print("Documents Embedding ...")
    sentences_embedding = encode_texts(texts=list(corpus.values()),
                                    batch_size=args.passage_batch_size)

    print("Dev Queries Embedding ...")
    queries_dev_embedding = encode_texts(texts=list(dev_queries.values()),
                                        batch_size=args.query_batch_size)

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
        only_pidqids_results[qid] = [" ".join(answer['corpus_id'].split()[:-1]).strip() for answer in result]

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
        f.write(f"\n\nVietnamese bi encoder\n")
        for key, value in metrics.items():
            f.write(f"\t{key}: {value}\n")
        f.write("\n\n")
