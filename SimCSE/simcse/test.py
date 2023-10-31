import pandas as pd

df = pd.read_csv("generated_data/corpus_256.csv", encoding='utf-8')
print(df.head())
