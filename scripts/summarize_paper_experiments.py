import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.metrics import PPIMetrics


@dataclass(frozen=True)
class SummaryRow:
    name: str
    micro_f1: float
    auc: float
    auprc: float
    accuracy: float
    eval_time_s: float
    path: str


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_dict(dict_path: str) -> Dict[str, str]:
    df = pd.read_csv(dict_path, sep="\t", header=None, names=["pid", "seq"])
    return dict(zip(df["seq"].astype(str), df["pid"].astype(str)))


def load_split_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df


def to_id_pairs(df: pd.DataFrame, seq2id: Dict[str, str]) -> Tuple[List[str], List[str]]:
    a = df["protein_a"].astype(str).tolist()
    b = df["protein_b"].astype(str).tolist()
    ids_a = [seq2id.get(s, f"UNK_A_{i}") for i, s in enumerate(a)]
    ids_b = [seq2id.get(s, f"UNK_B_{i}") for i, s in enumerate(b)]
    return ids_a, ids_b


def split_categories(
    train_ids: set, test_ids_a: List[str], test_ids_b: List[str]
) -> Dict[str, np.ndarray]:
    a_seen = np.array([pid in train_ids for pid in test_ids_a], dtype=bool)
    b_seen = np.array([pid in train_ids for pid in test_ids_b], dtype=bool)
    bs = a_seen & b_seen
    es = (a_seen ^ b_seen)
    ns = (~a_seen) & (~b_seen)
    return {"BS": bs, "ES": es, "NS": ns}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = PPIMetrics()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc": float(metrics.calculate_roc_auc(y_true, y_prob)),
        "auprc": float(metrics.calculate_auprc(y_true, y_prob)),
    }


def scan_summary_results(results_root: str) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    for dirpath, _, filenames in os.walk(results_root):
        if "evaluation_results_standard.json" not in filenames:
            continue
        path = os.path.join(dirpath, "evaluation_results_standard.json")
        d = load_json(path)
        if "micro_f1" not in d:
            continue
        rows.append(
            SummaryRow(
                name=os.path.relpath(dirpath, results_root).replace("\\", "/"),
                micro_f1=float(d.get("micro_f1", 0.0)),
                auc=float(d.get("auc", 0.0)),
                auprc=float(d.get("auprc", 0.0)),
                accuracy=float(d.get("accuracy", 0.0)),
                eval_time_s=float(d.get("evaluation_time", 0.0)),
                path=path,
            )
        )
    rows.sort(key=lambda r: (r.micro_f1, r.auc), reverse=True)
    return rows


def scan_detailed_results(detailed_root: str) -> List[Tuple[str, Dict[str, float], int]]:
    out: List[Tuple[str, Dict[str, float], int]] = []
    for dirpath, _, filenames in os.walk(detailed_root):
        for fn in filenames:
            if not (fn.startswith("standard_results_") and fn.endswith(".json")):
                continue
            path = os.path.join(dirpath, fn)
            d = load_json(path)
            y_pred = np.array(d["predictions"], dtype=float)
            y_true = np.array(d["labels"], dtype=float)
            y_prob = np.array(d["probabilities"], dtype=float)
            metrics = compute_metrics(y_true, y_pred, y_prob)
            out.append((path, metrics, int(y_true.shape[0])))
    return out


def match_detailed_file(
    detailed: List[Tuple[str, Dict[str, float], int]],
    target_micro_f1: float,
    target_n: int,
    tol: float = 5e-6,
) -> Optional[str]:
    best = None
    best_gap = 1e9
    for path, m, n in detailed:
        if n != target_n:
            continue
        gap = abs(m["micro_f1"] - target_micro_f1)
        if gap <= tol and gap < best_gap:
            best_gap = gap
            best = path
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="artifacts/metrics")
    ap.add_argument("--detailed-root", default="artifacts/eval_detailed")
    ap.add_argument("--data-root", default="data/shs27k_llapa/processed")
    ap.add_argument("--seq-dict", default="data/raw/shs27k/extracted/raw_data/protein.STRING.sequences.dictionary.tsv")
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    seq2id = load_label_dict(args.seq_dict)

    summaries = scan_summary_results(args.results_root)
    detailed = scan_detailed_results(args.detailed_root)

    print("== Summary results (top) ==")
    for r in summaries[: args.topk]:
        print(
            f"{r.micro_f1:.4f}\tauc={r.auc:.4f}\tauprc={r.auprc:.4f}\tacc={r.accuracy:.4f}\t"
            f"t={r.eval_time_s:.1f}s\t{r.name}"
        )

    split_files = {}
    for split in ["bfs", "dfs", "random"]:
        split_files[split] = {
            "train": os.path.join(args.data_root, f"{split}_train.tsv"),
            "val": os.path.join(args.data_root, f"{split}_val.tsv"),
            "test": os.path.join(args.data_root, f"{split}_test.tsv"),
        }

    print("\n== Dataset stats (by split) ==")
    for split, files in split_files.items():
        train_df = load_split_df(files["train"])
        test_df = load_split_df(files["test"])
        train_a, train_b = to_id_pairs(train_df, seq2id)
        train_ids = set(train_a) | set(train_b)
        test_a, test_b = to_id_pairs(test_df, seq2id)
        cats = split_categories(train_ids, test_a, test_b)
        n = len(test_df)
        frac_unseen_pair = float((~cats["BS"]).sum() / max(n, 1))
        print(
            f"{split}: |X_train|={len(train_df)}, |X_test|={len(test_df)}, "
            f"|P_v|={len(train_ids)}, unseen-pair%={100*frac_unseen_pair:.2f}%, "
            f"BS={int(cats['BS'].sum())}, ES={int(cats['ES'].sum())}, NS={int(cats['NS'].sum())}"
        )

    print("\n== Stratified metrics for selected runs ==")
    for r in summaries[: min(args.topk, len(summaries))]:
        n_guess = None
        for split, files in split_files.items():
            test_df = load_split_df(files["test"])
            n = len(test_df)
            if n_guess is None:
                n_guess = n
            detailed_path = match_detailed_file(detailed, r.micro_f1, n)
            if detailed_path:
                train_df = load_split_df(files["train"])
                test_a, test_b = to_id_pairs(test_df, seq2id)
                train_a, train_b = to_id_pairs(train_df, seq2id)
                train_ids = set(train_a) | set(train_b)
                cats = split_categories(train_ids, test_a, test_b)

                d = load_json(detailed_path)
                y_pred = np.array(d["predictions"], dtype=float)
                y_true = np.array(d["labels"], dtype=float)
                y_prob = np.array(d["probabilities"], dtype=float)

                print(f"\n{r.name}  (matched: {os.path.basename(detailed_path)}; split={split}; N={n})")
                for key in ["BS", "ES", "NS"]:
                    mask = cats[key]
                    if mask.sum() == 0:
                        continue
                    m = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
                    print(
                        f"  {key}: n={int(mask.sum())}\tmicro_f1={m['micro_f1']:.4f}\tauc={m['auc']:.4f}\tauprc={m['auprc']:.4f}\tacc={m['accuracy']:.4f}"
                    )
                break


if __name__ == "__main__":
    main()
