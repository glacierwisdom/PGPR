import pandas as pd
import numpy as np
import ast

df = pd.read_csv('data/shs27k_llapa/processed/random_train.tsv', sep='\t')
labels = df['label'].apply(ast.literal_eval).tolist()
labels_array = np.array(labels)
counts = labels_array.sum(axis=0)

print("Label counts per index (0-6):")
for i, count in enumerate(counts):
    print(f"Index {i}: {count}")
