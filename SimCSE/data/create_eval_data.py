import json
import os
import pandas as pd
from tqdm import tqdm
import py_vncorenlp
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./SimCSE/data/zalo_legal_2021", type=str, help="path to training data")
    parser.add_argument("--save_dir", default="./SimCSE/generated_data", type=str, help="path to training data")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir,exist_ok=True)
    qa_path = os.path.join(args.data_dir, "original/train_question_answer.json")
    dev_split_path = os.path.join(args.data_dir, "processed/queries_dev.json")
    output_path = os.path.join(args.save_dir, "eval_data.json")

    qa = json.load(open(qa_path, encoding='utf-8'))['items']
    dev_split = json.load(open(dev_split_path, encoding='utf-8'))
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
    
    print("Creating data file for evaluation")
    i = 2556
    question_ids = []
    questions = []
    relevant_articles = []
    
    for question_id, question in tqdm(dev_split.items()):
        question = " ".join(rdrsegmenter.word_segment(question))
        annotation = qa[i]
        assert question_id == annotation['question_id']
            
        question_ids.append(question_id)
        questions.append(question)
        relevant_articles.append(annotation['relevant_articles'])
        
        i += 1
    
    eval_data = {
        'question_ids': question_ids,
        'questions': questions,
        'relevant_articles': relevant_articles
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f)
                    