import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def process_shs27k():
    print("Processing SHS27k data...")
    
    # Paths
    raw_dir = Path("data/raw/shs27k/extracted/raw_data")
    actions_file = raw_dir / "protein.actions.SHS27k.txt"
    seq_file = raw_dir / "protein.SHS27k.sequences.dictionary.tsv"
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Read data
    print("Reading files...")
    df_actions = pd.read_csv(actions_file, sep='\t')
    df_seq = pd.read_csv(seq_file, sep='\t', header=None, names=['protein_id', 'sequence'])
    
    # Create sequence map
    seq_map = pd.Series(df_seq.sequence.values, index=df_seq.protein_id).to_dict()
    
    # Prepare positive samples
    print("Preparing positive samples...")
    positives = []
    positive_pairs = set()
    
    for _, row in df_actions.iterrows():
        p1 = row['item_id_a']
        p2 = row['item_id_b']
        
        # Skip if sequence missing
        if p1 not in seq_map or p2 not in seq_map:
            continue
            
        # Store pair to check for negatives later
        # Use sorted tuple to handle undirected nature if needed, but STRING is often directed or undirected.
        # The file has 'is_directional' column.
        # But for PPI, usually we treat {A, B} same as {B, A}.
        pair = tuple(sorted((p1, p2)))
        positive_pairs.add(pair)
        
        positives.append({
            'protein_A_id': p1,
            'protein_B_id': p2,
            'sequence_A': seq_map[p1],
            'sequence_B': seq_map[p2],
            'relationship_label': 1
        })
        
    print(f"Found {len(positives)} positive samples.")
    
    # Generate negative samples
    print("Generating negative samples...")
    negatives = []
    proteins = list(seq_map.keys())
    num_negatives = len(positives)
    
    while len(negatives) < num_negatives:
        p1 = random.choice(proteins)
        p2 = random.choice(proteins)
        
        if p1 == p2:
            continue
            
        pair = tuple(sorted((p1, p2)))
        
        if pair not in positive_pairs:
            negatives.append({
                'protein_A_id': p1,
                'protein_B_id': p2,
                'sequence_A': seq_map[p1],
                'sequence_B': seq_map[p2],
                'relationship_label': 0
            })
            positive_pairs.add(pair) # Avoid duplicates in negatives too
            
    print(f"Generated {len(negatives)} negative samples.")
    
    # Combine
    all_data = positives + negatives
    random.shuffle(all_data)
    df_all = pd.DataFrame(all_data)
    
    # Split
    print("Splitting into train/val/test...")
    train_df, temp_df = train_test_split(df_all, test_size=0.3, random_state=42, stratify=df_all['relationship_label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['relationship_label'])
    
    # Save
    print("Saving files...")
    df_all.to_csv(processed_dir / "shs27k_processed.tsv", sep='\t', index=False)
    train_df.to_csv(processed_dir / "shs27k_train.csv", index=False)
    val_df.to_csv(processed_dir / "shs27k_val.csv", index=False)
    test_df.to_csv(processed_dir / "shs27k_test.csv", index=False)
    
    print("Done!")
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

if __name__ == "__main__":
    process_shs27k()
