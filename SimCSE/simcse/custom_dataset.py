from torch.utils.data import Dataset
import random
import pandas as pd

class ZaloMSMarcoDataset(Dataset):
    def __init__(self, train_file: str, collection_file: str, tokenizer):
        super(ZaloMSMarcoDataset, self).__init__()
        self.tokenizer = tokenizer
        self.train_df = pd.read_csv(train_file, encoding='utf-8')
        self.collection_df = pd.read_csv(collection_file, sep='\t', header=None, names=['id', 'text'], encoding='utf-8', index_col='id')
        
    def __getitem__(self, index):
        if index <= 2644:
            sent0, sent1, hard_neg = self.train_df.iloc[index]
        else:
            sent0, sent1_ids, hard_neg_ids = self.train_df.iloc[index]
            sent1_id = random.choice(eval(sent1_ids))
            sent1 = self.collection_df['text'][sent1_id]
            hard_neg_id = random.choice(eval(hard_neg_ids))
            hard_neg = self.collection_df['text'][hard_neg_id]
        
        if sent0 is None:
            sent0 = " "
        if sent1 is None:
            sent1 = " "
        if hard_neg is None:
            hard_neg = " "
        
        sentences = [sent0, sent1, hard_neg]

        sent_features = self.tokenizer(
            sentences,
            max_length=256,
            truncation=True,
            padding=False,
        )
                
        return sent_features
    
    def __len__(self):
        return len(self.train_df)
    
class ZaloNewsDataset(Dataset):
    def __init__(self, train_file: str, tokenizer):
        super(ZaloNewsDataset, self).__init__()
        self.tokenizer = tokenizer
        self.train_df = pd.read_csv(train_file, encoding='utf-8')
        
    def __getitem__(self, index):
        sent0, sent1, hard_neg = self.train_df.iloc[index]
        
        if sent0 is None:
            sent0 = " "
        if sent1 is None:
            sent1 = " "
        if hard_neg is None:
            hard_neg = " "
        
        sentences = [sent0, sent1, hard_neg]

        sent_features = self.tokenizer(
            sentences,
            max_length=256,
            truncation=True,
            padding=False,
        )
        sent_features['has_hard_negative'] = 1 if index <= 2644 else 0
                
        return sent_features
    
    def __len__(self):
        return len(self.train_df)

def get_dataset(type, tokenizer, train_file: str, collection_file: str = None):
    if type == "msmarco":
        return ZaloMSMarcoDataset(train_file=train_file,
                                collection_file=collection_file,
                                tokenizer=tokenizer)
    elif type == "news":
        return ZaloNewsDataset(train_file=train_file,
                               tokenizer=tokenizer)
    else:
        raise ValueError
