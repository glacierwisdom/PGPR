
import pandas as pd
import json
import os
import ast
from pathlib import Path

def convert_split(mode):
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / 'data' / 'shs27k_llapa' / 'SHS27k_ml.csv'
    json_path = project_root / 'data' / 'shs27k_llapa' / f'SHS27k_{mode}.json'
    output_dir = project_root / 'data' / 'shs27k_llapa' / 'processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # LLAPA 定义的 7 种关系
    CLASS_NAMES = ['activation', 'binding', 'catalysis', 'expression', 'inhibition', 'ptmod', 'reaction']
    
    print(f"Converting {mode} split...")
    df = pd.read_csv(csv_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        split_indices = json.load(f)
    
    train_idx = split_indices['train_index']
    val_idx = split_indices['val_index']
    test_idx = split_indices['test_index']
    
    # 将 mode 字符串转换为 7 位 0/1 列表
    def mode_to_label(mode_str):
        try:
            active_modes = ast.literal_eval(mode_str)
            label = [0] * 7
            for i, name in enumerate(CLASS_NAMES):
                if name in active_modes:
                    label[i] = 1
            return label
        except:
            return [0] * 7

    # 转换列
    df['label'] = df['mode'].apply(mode_to_label)
    # 改名以匹配 GAPNPPI 预期
    df = df.rename(columns={'seq_a': 'protein_a', 'seq_b': 'protein_b'})
    
    # 提取子集
    train_df = df.iloc[train_idx][['protein_a', 'protein_b', 'label']]
    val_df = df.iloc[val_idx][['protein_a', 'protein_b', 'label']]
    test_df = df.iloc[test_idx][['protein_a', 'protein_b', 'label']]
    
    def save_tsv(data, name):
        data.to_csv(os.path.join(output_dir, f"{mode}_{name}.tsv"), sep='\t', index=False)
    
    save_tsv(train_df, "train")
    save_tsv(val_df, "val")
    save_tsv(test_df, "test")
    print(f"Finished {mode} split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

for m in ['random', 'dfs', 'bfs']:
    convert_split(m)
