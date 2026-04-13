import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from utils.caching import DiskCache


@dataclass(frozen=True)
class ProteinMappingResult:
    mapped_ids: List[str]
    used_fallback: List[bool]


class ProteinSimilarityMapper:
    def __init__(
        self,
        esm_encoder,
        cache_dir: str,
        enabled: bool = True,
        batch_size: int = 16,
        method: str = "esm",
        allow_fallback_to_esm: bool = True,
        blastp_evalue: float = 1e-5,
        blastp_num_threads: int = 4,
        blastp_timeout_seconds: int = 30,
    ):
        self.enabled = bool(enabled)
        self.esm_encoder = esm_encoder
        self.batch_size = int(batch_size)
        self.method = str(method).lower()
        self.allow_fallback_to_esm = bool(allow_fallback_to_esm)
        self.blastp_evalue = float(blastp_evalue)
        self.blastp_num_threads = int(blastp_num_threads)
        self.blastp_timeout_seconds = int(blastp_timeout_seconds)
        self.cache = DiskCache(cache_dir=cache_dir, max_size=200000, ttl=None)
        self._candidate_ids: List[str] = []
        self._candidate_embs: Optional[np.ndarray] = None
        self._candidate_id_set: Set[str] = set()
        self._trained = False
        self._backend = "esm"
        self._blast_db_prefix: Optional[str] = None
        self._blast_fasta_path: Optional[str] = None

    def is_ready(self) -> bool:
        if not self._trained:
            return False
        if self._backend == "blastp":
            return bool(self._candidate_ids) and bool(self._blast_db_prefix)
        return self._candidate_embs is not None and bool(self._candidate_ids)

    def fit(
        self,
        train_protein_sequences: Dict[str, str],
        non_isolated_ids: Set[str],
    ) -> None:
        if not self.enabled:
            self._trained = True
            return

        candidate_ids = [pid for pid in train_protein_sequences.keys() if pid in non_isolated_ids]
        candidate_ids = sorted(set(candidate_ids))
        if not candidate_ids:
            self._candidate_ids = []
            self._candidate_embs = None
            self._candidate_id_set = set()
            self._trained = True
            return

        self._candidate_ids = candidate_ids
        self._candidate_id_set = set(candidate_ids)
        cache_dir = getattr(self.cache, "cache_dir", None) or getattr(self.cache, "_cache_dir", None) or None
        cache_dir = cache_dir or os.getcwd()

        if self.method == "blastp":
            blastp_ok = bool(shutil.which("blastp")) and bool(shutil.which("makeblastdb"))
            if blastp_ok:
                fasta_path = os.path.join(cache_dir, "blast_candidates.fasta")
                db_prefix = os.path.join(cache_dir, "blast_db", "train_candidates")
                os.makedirs(os.path.dirname(db_prefix), exist_ok=True)
                self._write_fasta_if_needed(fasta_path, candidate_ids, train_protein_sequences)
                self._ensure_blast_db(fasta_path, db_prefix)
                self._backend = "blastp"
                self._blast_fasta_path = fasta_path
                self._blast_db_prefix = db_prefix
                self._candidate_embs = None
                self._trained = True
                return

            if not self.allow_fallback_to_esm:
                self._backend = "esm"
                self._candidate_embs = None
                self._trained = True
                return

        sequences = [train_protein_sequences[pid] for pid in candidate_ids]
        embeddings = self._encode_sequences(sequences)
        embeddings = self._l2_normalize(embeddings)
        self._backend = "esm"
        self._candidate_embs = embeddings
        self._trained = True

    def map_batch(
        self,
        protein_ids: Sequence[str],
        protein_sequences: Sequence[str],
        non_isolated_ids: Set[str],
    ) -> ProteinMappingResult:
        if not self.enabled:
            return ProteinMappingResult(mapped_ids=list(protein_ids), used_fallback=[False] * len(protein_ids))

        if not self.is_ready():
            return ProteinMappingResult(mapped_ids=list(protein_ids), used_fallback=[False] * len(protein_ids))

        mapped_ids: List[str] = []
        used_fallback: List[bool] = []

        to_query_indices: List[int] = []
        to_query_sequences: List[str] = []
        to_query_ids: List[str] = []

        for i, (pid, seq) in enumerate(zip(protein_ids, protein_sequences)):
            if pid in self._candidate_id_set and pid in non_isolated_ids:
                mapped_ids.append(pid)
                used_fallback.append(False)
                continue

            cache_key = f"map::{pid}"
            cached = self.cache.get(cache_key)
            if isinstance(cached, str) and cached in self._candidate_id_set:
                mapped_ids.append(cached)
                used_fallback.append(True)
                continue

            mapped_ids.append("")
            used_fallback.append(True)
            to_query_indices.append(i)
            to_query_ids.append(pid)
            to_query_sequences.append(seq)

        if to_query_indices:
            if self._backend == "blastp":
                nn_ids = self._blastp_nearest_neighbors(to_query_ids, to_query_sequences)
            else:
                q_embs = self._encode_sequences(to_query_sequences)
                q_embs = self._l2_normalize(q_embs)
                nn_ids = self._nearest_neighbors(q_embs)
            for i, mapped in zip(to_query_indices, nn_ids):
                mapped_ids[i] = mapped
                self.cache.set(f"map::{protein_ids[i]}", mapped)

        for i, mid in enumerate(mapped_ids):
            if not mid:
                mapped_ids[i] = protein_ids[i]
                used_fallback[i] = False

        return ProteinMappingResult(mapped_ids=mapped_ids, used_fallback=used_fallback)

    def _encode_sequences(self, sequences: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            batch_embs = self.esm_encoder.get_batch_embeddings(list(batch), batch_size=len(batch))
            batch_np = [emb.detach().float().cpu().numpy() for emb in batch_embs]
            vectors.extend(batch_np)
        return np.stack(vectors, axis=0).astype(np.float32, copy=False)

    def _nearest_neighbors(self, query_embs: np.ndarray) -> List[str]:
        cand = self._candidate_embs
        scores = np.matmul(query_embs, cand.T)
        best_idx = np.argmax(scores, axis=1)
        return [self._candidate_ids[int(i)] for i in best_idx]

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        return x / denom

    def _write_fasta_if_needed(self, fasta_path: str, candidate_ids: Sequence[str], seqs: Dict[str, str]) -> None:
        if os.path.exists(fasta_path):
            try:
                if os.path.getsize(fasta_path) > 0:
                    return
            except OSError:
                pass
        tmp_path = f"{fasta_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for pid in candidate_ids:
                s = seqs.get(pid)
                if not s:
                    continue
                f.write(f">{pid}\n{s}\n")
        os.replace(tmp_path, fasta_path)

    def _ensure_blast_db(self, fasta_path: str, db_prefix: str) -> None:
        pin = f"{db_prefix}.pin"
        phr = f"{db_prefix}.phr"
        psq = f"{db_prefix}.psq"
        if os.path.exists(pin) and os.path.exists(phr) and os.path.exists(psq):
            return
        cmd = [
            "makeblastdb",
            "-in",
            fasta_path,
            "-dbtype",
            "prot",
            "-out",
            db_prefix,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            err = ""
            try:
                err = e.stderr.decode("utf-8", errors="replace")
            except Exception:
                err = str(e)
            raise RuntimeError(f"makeblastdb failed: {err}") from e

    def _blastp_nearest_neighbors(self, query_ids: Sequence[str], query_sequences: Sequence[str]) -> List[str]:
        if not self._blast_db_prefix:
            return [self._candidate_ids[0] for _ in query_ids]
        cache_dir = getattr(self.cache, "cache_dir", None) or getattr(self.cache, "_cache_dir", None) or os.getcwd()
        query_fasta = os.path.join(cache_dir, f"blast_query_{os.getpid()}.fasta")
        out_path = os.path.join(cache_dir, f"blast_out_{os.getpid()}.tsv")
        with open(query_fasta, "w", encoding="utf-8") as f:
            for qid, seq in zip(query_ids, query_sequences):
                if not seq:
                    continue
                f.write(f">{qid}\n{seq}\n")
        cmd = [
            "blastp",
            "-query",
            query_fasta,
            "-db",
            self._blast_db_prefix,
            "-out",
            out_path,
            "-outfmt",
            "6 qseqid sseqid evalue bitscore",
            "-evalue",
            str(self.blastp_evalue),
            "-max_target_seqs",
            "1",
            "-num_threads",
            str(max(1, self.blastp_num_threads)),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(1, self.blastp_timeout_seconds),
            )
        except Exception:
            try:
                os.remove(query_fasta)
            except OSError:
                pass
            try:
                os.remove(out_path)
            except OSError:
                pass
            if self.allow_fallback_to_esm:
                q_embs = self._encode_sequences(query_sequences)
                q_embs = self._l2_normalize(q_embs)
                return self._nearest_neighbors(q_embs)
            return [self._candidate_ids[0] for _ in query_ids]

        mapping: Dict[str, str] = {}
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    qid, sid = parts[0], parts[1]
                    if sid in self._candidate_id_set:
                        mapping[qid] = sid
        finally:
            try:
                os.remove(query_fasta)
            except OSError:
                pass
            try:
                os.remove(out_path)
            except OSError:
                pass

        default_id = self._candidate_ids[0]
        return [mapping.get(qid, default_id) for qid in query_ids]


def compute_non_isolated_ids(graph_data) -> Set[str]:
    num_nodes = int(graph_data.x.size(0))
    if not hasattr(graph_data, "edge_index") or graph_data.edge_index is None:
        return set()

    deg = torch.bincount(graph_data.edge_index[0].detach().cpu(), minlength=num_nodes)
    non_isolated = set()
    for idx, d in enumerate(deg.tolist()):
        if d > 0:
            non_isolated.add(graph_data.protein_ids[idx])
    return non_isolated
