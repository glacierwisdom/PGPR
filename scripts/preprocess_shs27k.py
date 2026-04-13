#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHS 数据集预处理脚本（SHS27k/SHS148k）
用于把原始文件整理为工程可直接训练/评估的 TSV，并生成 random/bfs/dfs 三种划分。
"""

import sys
import logging
import pandas as pd
from pathlib import Path
import argparse
import ast
import hashlib
import random
from collections import deque
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHSPreprocessor:
    """SHS数据集预处理类"""
    
    def __init__(self, project_root: Path, dataset_name: str):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.dataset_name = dataset_name.lower()
        self.raw_dir_candidates = [
            self.data_dir / f"{self.dataset_name}_llapa" / "raw",
            self.data_dir / "raw" / self.dataset_name,
            self.data_dir / "raw",
        ]
        self.processed_dir = self.data_dir / f"{self.dataset_name}_llapa" / "processed"
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def _stable_id(self, sequence: str) -> str:
        seq = (sequence or "").strip().upper()
        digest = hashlib.sha1(seq.encode("utf-8")).hexdigest()[:16]
        return f"SEQ_{digest}"

    def _find_default_raw_file(self) -> Path:
        candidate_names = [
            f"{self.dataset_name}.tsv",
            f"{self.dataset_name}.csv",
            f"{self.dataset_name.upper()}.tsv",
            f"{self.dataset_name.upper()}.csv",
            "SHS27k.csv",
            "SHS148k.csv",
            "SHS27K.csv",
            "SHS148K.csv",
        ]
        for raw_dir in self.raw_dir_candidates:
            for name in candidate_names:
                p = raw_dir / name
                if p.exists():
                    return p
        return Path()

    def _load_raw(self, raw_file: Path) -> pd.DataFrame:
        ext = raw_file.suffix.lower()
        if ext == ".tsv":
            return pd.read_csv(raw_file, sep="\t")
        if ext == ".csv":
            return pd.read_csv(raw_file)
        raise ValueError(f"不支持的文件格式: {raw_file}")

    def _normalize_labels(self, series: pd.Series, num_relations: int = 7) -> pd.Series:
        rel_map = {
            "activation": 0,
            "binding": 1,
            "catalysis": 2,
            "expression": 3,
            "inhibition": 4,
            "inhibitory": 4,
            "ptm": 5,
            "ptmod": 5,
            "ptmodulation": 5,
            "reaction": 6,
        }

        def to_vec(x):
            if isinstance(x, (list, tuple)):
                vec = list(x)
            elif isinstance(x, str):
                s = x.strip()
                if s.startswith("[") and s.endswith("]"):
                    vec = ast.literal_eval(s)
                else:
                    lower = s.lower()
                    if lower in rel_map:
                        vec = [lower]
                    else:
                        try:
                            vec = int(s)
                        except Exception:
                            raise ValueError(f"无法解析标签: {x}")
            else:
                vec = x

            if isinstance(vec, (int, float)):
                idx = int(vec)
                if idx < 0 or idx >= num_relations:
                    raise ValueError(f"标签超出范围(0-{num_relations-1}): {idx}")
                onehot = [0] * num_relations
                onehot[idx] = 1
                return onehot

            if isinstance(vec, (list, tuple)):
                if len(vec) == 0:
                    return [0] * num_relations

                if all(isinstance(v, str) for v in vec):
                    out = [0] * num_relations
                    for v in vec:
                        key = (v or "").strip().lower()
                        if key in rel_map:
                            out[rel_map[key]] = 1
                        else:
                            raise ValueError(f"未知关系类型: {v}")
                    return out

                vec = [int(v) for v in vec]
                if len(vec) != num_relations:
                    raise ValueError(f"多标签维度不等于{num_relations}: {len(vec)}")
                return vec

            if isinstance(vec, bool):
                return [0] * num_relations

            raise ValueError(f"无法解析标签类型: {type(x)}")

        return series.apply(to_vec)

    def _normalize_raw_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        lower_cols = {c.lower(): c for c in df.columns}

        if all(k in lower_cols for k in ["seq_a", "seq_b"]) and "canbind" in lower_cols:
            seq_a_col = lower_cols["seq_a"]
            seq_b_col = lower_cols["seq_b"]
            bind_col = lower_cols["canbind"]
            labels = df[bind_col].astype(str).str.lower().isin(["true", "1", "t", "yes", "y"])
            out = pd.DataFrame(
                {
                    "protein_a": df[seq_a_col].astype(str),
                    "protein_b": df[seq_b_col].astype(str),
                    "label": labels.apply(lambda v: [0, 0, int(bool(v)), 0, 0, 0, 0]),
                }
            )
            non_empty = out["label"].apply(lambda x: sum(int(v) for v in x) > 0)
            out = out.loc[non_empty].reset_index(drop=True)
            return out

        if all(k in lower_cols for k in ["seq_a", "seq_b"]):
            seq_a_col = lower_cols["seq_a"]
            seq_b_col = lower_cols["seq_b"]
            label_col = lower_cols.get("mode") or lower_cols.get("label") or lower_cols.get("relationship_label") or lower_cols.get("interaction")
            if label_col is None:
                raise ValueError("找不到标签列(mode/label/relationship_label/interaction)")
            out = pd.DataFrame(
                {
                    "protein_a": df[seq_a_col].astype(str),
                    "protein_b": df[seq_b_col].astype(str),
                    "label": self._normalize_labels(df[label_col]),
                }
            )
            if "id" in lower_cols:
                id_col = lower_cols["id"]
                parts = df[id_col].astype(str).str.split("-", n=1, expand=True)
                if parts.shape[1] == 2:
                    out["protein_a_id"] = parts[0].astype(str)
                    out["protein_b_id"] = parts[1].astype(str)
            non_empty = out["label"].apply(lambda x: sum(int(v) for v in x) > 0)
            out = out.loc[non_empty].reset_index(drop=True)
            return out

        if all(k in lower_cols for k in ["protein_a", "protein_b"]):
            seq_a_col = lower_cols["protein_a"]
            seq_b_col = lower_cols["protein_b"]
            label_col = lower_cols.get("label") or lower_cols.get("relationship_label") or lower_cols.get("interaction")
            if label_col is None:
                raise ValueError("找不到标签列(label/relationship_label/interaction)")
            out = pd.DataFrame(
                {
                    "protein_a": df[seq_a_col].astype(str),
                    "protein_b": df[seq_b_col].astype(str),
                    "label": self._normalize_labels(df[label_col]),
                }
            )
            return out

        if all(k in lower_cols for k in ["sequence_a", "sequence_b"]):
            seq_a_col = lower_cols["sequence_a"]
            seq_b_col = lower_cols["sequence_b"]
            label_col = lower_cols.get("label") or lower_cols.get("relationship_label") or lower_cols.get("interaction")
            if label_col is None:
                raise ValueError("找不到标签列(label/relationship_label/interaction)")
            out = pd.DataFrame(
                {
                    "protein_a": df[seq_a_col].astype(str),
                    "protein_b": df[seq_b_col].astype(str),
                    "label": self._normalize_labels(df[label_col]),
                }
            )
            return out

        if all(k in lower_cols for k in ["protein1", "protein2"]) and ("interaction" in lower_cols or "label" in lower_cols):
            p1_col = lower_cols["protein1"]
            p2_col = lower_cols["protein2"]
            label_col = lower_cols.get("interaction") or lower_cols.get("label")
            seq1_col = lower_cols.get("sequence1")
            seq2_col = lower_cols.get("sequence2")
            if seq1_col and seq2_col:
                seq_a = df[seq1_col].astype(str)
                seq_b = df[seq2_col].astype(str)
            else:
                seq_a = df[p1_col].astype(str)
                seq_b = df[p2_col].astype(str)
            out = pd.DataFrame(
                {
                    "protein_a": seq_a,
                    "protein_b": seq_b,
                    "label": self._normalize_labels(df[label_col]),
                }
            )
            return out

        raise ValueError(f"无法识别原始数据列: {list(df.columns)}")

    def _with_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "protein_a_id" not in out.columns:
            out["protein_a_id"] = out["protein_a"].apply(self._stable_id)
        if "protein_b_id" not in out.columns:
            out["protein_b_id"] = out["protein_b"].apply(self._stable_id)
        out["label"] = out["label"].apply(lambda x: str(list(x)))
        return out[["protein_a_id", "protein_b_id", "protein_a", "protein_b", "label"]]

    def _dedup_undirected(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if "protein_a_id" not in work.columns:
            work["protein_a_id"] = work["protein_a"].apply(self._stable_id)
        if "protein_b_id" not in work.columns:
            work["protein_b_id"] = work["protein_b"].apply(self._stable_id)
        work["pair_key"] = work.apply(
            lambda r: tuple(sorted([r["protein_a_id"], r["protein_b_id"]])),
            axis=1,
        )

        key_to_pos = {}
        records = []
        for row in work.itertuples(index=False):
            key = row.pair_key
            label_vec = list(row.label)
            if key not in key_to_pos:
                a_id, b_id = key
                if row.protein_a_id == a_id:
                    a_seq, b_seq = row.protein_a, row.protein_b
                else:
                    a_seq, b_seq = row.protein_b, row.protein_a
                key_to_pos[key] = len(records)
                records.append(
                    {
                        "protein_a_id": a_id,
                        "protein_b_id": b_id,
                        "protein_a": a_seq,
                        "protein_b": b_seq,
                        "label": label_vec,
                    }
                )
            else:
                pos = key_to_pos[key]
                prev = records[pos]["label"]
                records[pos]["label"] = [max(int(p), int(n)) for p, n in zip(prev, label_vec)]

        out = pd.DataFrame.from_records(records)
        return out

    def _split_random(self, df: pd.DataFrame, seed: int, test_ratio: float, val_ratio: float):
        rng = random.Random(seed)
        indices = list(range(len(df)))
        rng.shuffle(indices)
        n_test = int(round(len(df) * test_ratio))
        n_val = int(round(len(df) * val_ratio))
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]
        return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

    def _split_traversal(self, df: pd.DataFrame, seed: int, test_ratio: float, val_ratio: float, mode: str):
        rng = random.Random(seed)
        edges = list(zip(df["protein_a_id"], df["protein_b_id"]))
        nodes = list({u for u, v in edges}.union({v for u, v in edges}))
        if not nodes:
            raise ValueError("数据为空，无法划分")
        start = rng.choice(nodes)

        adj = {n: [] for n in nodes}
        for idx, (u, v) in enumerate(edges):
            adj[u].append((v, idx))
            adj[v].append((u, idx))

        for n in adj:
            rng.shuffle(adj[n])

        target_train = int(round(len(df) * (1.0 - test_ratio - val_ratio)))
        visited_nodes = set([start])
        visited_edges = set()
        if mode == "bfs":
            frontier = deque([start])
            pop = frontier.popleft
            push = frontier.append
        else:
            frontier = [start]
            pop = frontier.pop
            push = frontier.append

        while frontier and len(visited_edges) < target_train:
            cur = pop()
            for nb, eidx in adj.get(cur, []):
                if eidx not in visited_edges:
                    visited_edges.add(eidx)
                if nb not in visited_nodes:
                    visited_nodes.add(nb)
                    push(nb)
                if len(visited_edges) >= target_train:
                    break

        train_df = df.iloc[sorted(list(visited_edges))]
        remaining_idx = [i for i in range(len(df)) if i not in visited_edges]
        remaining_df = df.iloc[remaining_idx]
        if len(remaining_df) == 0:
            return train_df, remaining_df, remaining_df
        ratio_test_in_rest = test_ratio / (test_ratio + val_ratio)
        val_df, _, test_df = self._split_random(remaining_df, seed=seed + 1, test_ratio=ratio_test_in_rest, val_ratio=0.0)
        return train_df, val_df, test_df

    def _save_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, prefix: str):
        train_df.to_csv(self.processed_dir / f"{prefix}_train.tsv", sep="\t", index=False)
        val_df.to_csv(self.processed_dir / f"{prefix}_val.tsv", sep="\t", index=False)
        test_df.to_csv(self.processed_dir / f"{prefix}_test.tsv", sep="\t", index=False)
        logger.info(f"{prefix}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    def _find_official_split_dir(self, raw_file: Path) -> Path:
        candidates = [
            raw_file.parent / "processed_data",
            raw_file.parent / "STRING" / "processed_data",
            raw_file.parent.parent / "processed_data",
            raw_file.parent.parent / "STRING" / "processed_data",
        ]
        for p in candidates:
            if p.exists() and p.is_dir():
                return p
        return Path()

    def _load_official_split(self, split_dir: Path, stem: str, prefix: str) -> tuple[list[int], list[int], list[int]]:
        p = split_dir / f"{stem}_{prefix}.json"
        if not p.exists():
            raise FileNotFoundError(str(p))
        payload = json.loads(p.read_text())
        train_idx = payload.get("train_index", [])
        val_idx = payload.get("val_index", [])
        test_idx = payload.get("test_index", [])
        return train_idx, val_idx, test_idx

    def preprocess(self, raw_file: Path, seed: int = 42, test_ratio: float = 0.15, val_ratio: float = 0.15):
        logger.info(f"=== 开始预处理 {self.dataset_name.upper()} 数据集 ===")
        logger.info(f"原始数据文件: {raw_file}")

        raw_df = self._load_raw(raw_file)
        normalized = self._normalize_raw_schema(raw_df)
        normalized = self._dedup_undirected(normalized)
        with_ids = self._with_ids(normalized)

        all_path = self.processed_dir / "all.tsv"
        with_ids.to_csv(all_path, sep="\t", index=False)
        logger.info(f"保存标准化数据: {all_path} ({len(with_ids)})")

        official_dir = self._find_official_split_dir(raw_file)
        stem = raw_file.stem.replace("_ml", "").replace("_ML", "")

        if official_dir:
            logger.info(f"检测到官方划分目录: {official_dir}")
        else:
            logger.info("未检测到官方划分目录，将使用本地生成划分")

        effective_test_ratio = test_ratio
        effective_val_ratio = val_ratio

        if official_dir:
            try:
                train_idx, val_idx, test_idx = self._load_official_split(official_dir, stem, "random")
                n = len(with_ids)
                train_df = with_ids.iloc[train_idx] if train_idx else pd.DataFrame(columns=with_ids.columns)
                val_df = with_ids.iloc[val_idx] if val_idx else pd.DataFrame(columns=with_ids.columns)
                test_df = with_ids.iloc[test_idx] if test_idx else pd.DataFrame(columns=with_ids.columns)
                if len(train_df) + len(val_df) + len(test_df) > 0 and (len(train_df) == 0 or len(test_df) == 0):
                    raise ValueError("random 官方划分缺少 train/test")
                for idx_list in (train_idx, val_idx, test_idx):
                    if idx_list and (min(idx_list) < 0 or max(idx_list) >= n):
                        raise ValueError("random 官方划分索引越界")
                self._save_split(train_df, val_df, test_df, prefix="random")
                if n > 0:
                    effective_test_ratio = len(test_df) / n
                    effective_val_ratio = len(val_df) / n
            except Exception as e:
                logger.warning(f"random 官方划分不可用，回退到本地生成: {e}")
                train_df, val_df, test_df = self._split_random(with_ids, seed=seed, test_ratio=test_ratio, val_ratio=val_ratio)
                self._save_split(train_df, val_df, test_df, prefix="random")
        else:
            train_df, val_df, test_df = self._split_random(with_ids, seed=seed, test_ratio=test_ratio, val_ratio=val_ratio)
            self._save_split(train_df, val_df, test_df, prefix="random")

        bfs_train, bfs_val, bfs_test = self._split_traversal(
            with_ids,
            seed=seed,
            test_ratio=effective_test_ratio,
            val_ratio=effective_val_ratio,
            mode="bfs",
        )
        self._save_split(bfs_train, bfs_val, bfs_test, prefix="bfs")

        if official_dir:
            try:
                train_idx, val_idx, test_idx = self._load_official_split(official_dir, stem, "dfs")
                n = len(with_ids)
                dfs_train = with_ids.iloc[train_idx] if train_idx else pd.DataFrame(columns=with_ids.columns)
                dfs_val = with_ids.iloc[val_idx] if val_idx else pd.DataFrame(columns=with_ids.columns)
                dfs_test = with_ids.iloc[test_idx] if test_idx else pd.DataFrame(columns=with_ids.columns)
                for idx_list in (train_idx, val_idx, test_idx):
                    if idx_list and (min(idx_list) < 0 or max(idx_list) >= n):
                        raise ValueError("dfs 官方划分索引越界")
                self._save_split(dfs_train, dfs_val, dfs_test, prefix="dfs")
            except Exception as e:
                logger.warning(f"dfs 官方划分不可用，回退到本地生成: {e}")
                dfs_train, dfs_val, dfs_test = self._split_traversal(with_ids, seed=seed, test_ratio=test_ratio, val_ratio=val_ratio, mode="dfs")
                self._save_split(dfs_train, dfs_val, dfs_test, prefix="dfs")
        else:
            dfs_train, dfs_val, dfs_test = self._split_traversal(
                with_ids,
                seed=seed,
                test_ratio=effective_test_ratio,
                val_ratio=effective_val_ratio,
                mode="dfs",
            )
            self._save_split(dfs_train, dfs_val, dfs_test, prefix="dfs")

        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="shs27k", choices=["shs27k", "shs148k"])
    parser.add_argument("--raw-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    preprocessor = SHSPreprocessor(project_root, dataset_name=args.dataset)

    raw_file = Path(args.raw_file) if args.raw_file else preprocessor._find_default_raw_file()
    if not raw_file or not raw_file.exists():
        logger.error("找不到原始数据文件，请使用 --raw-file 指定")
        sys.exit(1)

    success = preprocessor.preprocess(raw_file=raw_file, seed=args.seed, test_ratio=args.test_ratio, val_ratio=args.val_ratio)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
