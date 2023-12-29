from torch.utils.data import Dataset
import random
import pandas as pd
import json
from typing import *

class SimCSEData():
    def __init__(self, train_triples_file: str, collection_file: str):
        self.train_triples_file = train_triples_file
        self.collection_file = collection_file
    
    def get_train_triples(self):
        raise NotImplementedError()
    
    def get_corpus(self):
        raise NotImplementedError()

    
class ZaloData(SimCSEData):
    def __init__(self, train_triples_file: str, collection_file: str):
        super().__init__(train_triples_file, collection_file)
        self.name = "zalo"
        with open(self.train_triples_file) as f:
            queries = json.load(f)
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = json.load(open(self.collection_file, encoding='utf-8'))
        
        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['hard_neg'] = list(self.queries[qid]['hard_neg'])
            random.shuffle(self.queries[qid]['hard_neg'])
    
    def get_train_triples(self):
        return self.queries
    
    def get_corpus(self):
        return self.corpus 

    
class MSMARCOData(SimCSEData):
    def __init__(self, train_triples_file: str, collection_file: str):
        super().__init__(train_triples_file, collection_file)
        self.name = "msmarco"
        with open(self.train_triples_file) as f:
            queries = json.load(f)
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = json.load(open(self.collection_file, encoding='utf-8'))
        
        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['hard_neg'] = list(self.queries[qid]['hard_neg'])
            random.shuffle(self.queries[qid]['hard_neg'])
    
    def get_train_triples(self):
        return self.queries
    
    def get_corpus(self):
        return self.corpus 

    
class SquadV2Data(SimCSEData):
    def __init__(self, train_triples_file: str, collection_file: str):
        super().__init__(train_triples_file, collection_file)
        self.name = "squadv2"
        with open(self.train_triples_file) as f:
            queries = json.load(f)
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = json.load(open(self.collection_file, encoding='utf-8'))
        
        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['hard_neg'] = list(self.queries[qid]['hard_neg'])
            random.shuffle(self.queries[qid]['hard_neg'])
    
    def get_train_triples(self):
        return self.queries
    
    def get_corpus(self):
        return self.corpus 
    

class SimCSEDataset(Dataset):
    def __init__(self, tokenizer,
                 data_list: List[SimCSEData]):
        super(SimCSEDataset, self).__init__()
        self.tokenizer = tokenizer
        
        data_names_list = [data.name for data in data_list]
        corpus_list = [data.get_corpus() for data in data_list]
        train_queries_list = [data.get_train_triples() for data in data_list]
        
        corpus = {}     #pid -> corpus
        train_queries = {} # same format all
        corpus_id_map = {} #(name_dataset, old_pid) -> new_pid

        current_pid = 0
        current_qid = 0

        for dataset_name, corpus_data in zip(data_names_list, corpus_list):
            for pid, para in corpus_data.items():
                corpus_id_map[(dataset_name, pid)] = current_pid
                corpus[current_pid] = para
                current_pid += 1

        for dataset_name, train_query in zip(data_names_list, train_queries_list):
            for data in train_query.values():
                data["qid"] = current_qid
                data["pos"] = [corpus_id_map[(dataset_name, pid)] for pid in list(data["pos"])]
                data["hard_neg"] = [corpus_id_map[(dataset_name, pid)] for pid in list(data["hard_neg"])]
                train_queries[current_qid] = data
                current_qid += 1
        
        self.queries = train_queries
        self.queries_ids = list(train_queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['hard_neg'] = list(self.queries[qid]['hard_neg'])
            random.shuffle(self.queries[qid]['hard_neg'])
        
    def __getitem__(self, index):
        query = self.queries[self.queries_ids[index]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['hard_neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['hard_neg'].append(neg_id)
        
        if query_text is None:
            query_text = " "
        if pos_text is None:
            pos_text = " "
        if neg_text is None:
            neg_text = " "
        
        sentences = [query_text, pos_text, neg_text]
        
        sent_features = self.tokenizer(
            sentences,
            max_length=256,
            truncation=True,
            padding=False,
        )
                
        return sent_features 
    
    def __len__(self):
        return len(self.queries)
