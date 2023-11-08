import json
import os
import pandas as pd
from tqdm import tqdm
import py_vncorenlp
import argparse

def break_into_chunks(input_list, max_chunk_size=256):
    result = []
    current_chunk = ""

    for string in input_list:
        tokens = string.split()
        if len(tokens) < max_chunk_size:
            if len(current_chunk.split()) + len(tokens) <= max_chunk_size:
                current_chunk = " ".join([current_chunk, string]).strip()
            else:
                result.append(current_chunk)
                current_chunk = string
        else:
            if current_chunk:
                result.append(current_chunk)
                current_chunk = ""
            for i in range(0, len(tokens), max_chunk_size):
                if i + max_chunk_size <= len(tokens):
                    result.append(" ".join(tokens[i:i+max_chunk_size]))
                else:
                    current_chunk = " ".join(tokens[i:])

    if current_chunk:
        result.append(current_chunk)

    return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./SimCSE/data/zalo_legal_2021/original", type=str, help="path to training data")
    parser.add_argument("--save_dir", default="./SimCSE/generated_data", type=str, help="path to training data")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir,exist_ok=True)
    corpus_path = os.path.join(args.data_dir, "legal_corpus.json")
    output_path = os.path.join(args.save_dir, "corpus_256.csv")

    data = json.load(open(corpus_path, encoding='utf-8'))
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
    
    law_ids = []
    article_ids = []
    contents = []

    for law_article in tqdm(data):
        law_id = law_article["law_id"]
        law_articles = law_article["articles"]
        
        # Replace duplicating law_id
        if law_id.endswith("nd-cp"):
            law_id = law_id.replace("nd-cp", "nđ-cp")
        if law_id.endswith("nđ-"):
            law_id = law_id.replace("nđ-", "nđ-cp")
        if law_id.endswith("nð-cp"):
            law_id = law_id.replace("nð-cp", "nđ-cp")
        if law_id == "09/2014/ttlt-btp-tandtc-vksndtc":
            law_id = "09/2014/ttlt-btp-tandtc-vksndtc-btc"
        
        for sub_article in law_articles:
            article_id = sub_article["article_id"]
            article_title = sub_article["title"]
            article_text = sub_article["text"]
            article_text = article_text.replace(u'\xa0', "")
            segment_result = rdrsegmenter.word_segment(article_text)
            chunks = break_into_chunks(segment_result, 256)
            
            for chunk in chunks:
                law_ids.append(law_id)
                article_ids.append(article_id)
                contents.append(chunk)
            
    corpus_df = pd.DataFrame({"law_id": law_ids,
                               "article_id": article_ids,
                               "text": contents})
    
    corpus_df.drop_duplicates(keep='first', inplace=True)
    corpus_df.reset_index(drop=True, inplace=True)
    
    corpus_df.to_csv(output_path, index=False, encoding='utf-8')
                    