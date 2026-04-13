import os
import sys
import subprocess
from pathlib import Path
import json
import time
import signal
import socket
import random
from typing import Dict, List, Optional, Any, cast


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pick_free_port() -> int:
    for _ in range(50):
        port = random.randint(20000, 60000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("无法找到可用的 master_port")


def run(cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("cmd=" + " ".join(cmd) + "\n")
        f.write("=" * 80 + "\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        code = p.wait()
        if code != 0:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
        return code


def ensure_gpu_ready() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    if not hasattr(torch.distributed, "is_nccl_available") or not torch.distributed.is_nccl_available():
        raise RuntimeError("当前 PyTorch 不支持 NCCL，无法 8 卡训练")
    if torch.cuda.device_count() < 8:
        raise RuntimeError(f"GPU 数量不足: {torch.cuda.device_count()}")


def ensure_splits() -> None:
    required = [
        PROJECT_ROOT / "data/shs148k_llapa/processed/random_train.tsv",
        PROJECT_ROOT / "data/shs148k_llapa/processed/random_val.tsv",
        PROJECT_ROOT / "data/shs148k_llapa/processed/random_test.tsv",
        PROJECT_ROOT / "data/shs148k_llapa/processed/bfs_train.tsv",
        PROJECT_ROOT / "data/shs148k_llapa/processed/bfs_val.tsv",
        PROJECT_ROOT / "data/shs148k_llapa/processed/bfs_test.tsv",
        PROJECT_ROOT / "data/shs148k_llapa/processed/dfs_train.tsv",
        PROJECT_ROOT / "data/shs148k_llapa/processed/dfs_val.tsv",
        PROJECT_ROOT / "data/shs148k_llapa/processed/dfs_test.tsv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少 SHS148K 处理后划分文件: " + ", ".join(missing))


def smoke_train(config_path: str) -> None:
    port = pick_free_port()
    cmd = [
        sys.executable,
        "experiments/run_training.py",
        "--config",
        config_path,
        "--config-override",
        "training.quick_run_steps=3",
        "training.quick_run_epochs=1",
        "preprocessing.similarity_mapping.enabled=false",
        f"distributed.master_port={port}",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    stem = Path(config_path).stem
    code = run(cmd, log_path=PROJECT_ROOT / "artifacts" / "logs" / f"shs148k_{stem}_smoke.log", env=env)
    if code != 0:
        raise RuntimeError(f"smoke 训练失败: {config_path}")


def full_train(config_path: str, extra_overrides: Optional[List[str]] = None) -> None:
    overrides = list(extra_overrides or [])
    overrides.append(f"distributed.master_port={pick_free_port()}")
    cmd = [
        sys.executable,
        "experiments/run_training.py",
        "--config",
        config_path,
    ]
    if overrides:
        cmd += ["--config-override", *overrides]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    stem = Path(config_path).stem
    code = run(cmd, log_path=PROJECT_ROOT / "artifacts" / "logs" / f"shs148k_{stem}_train.log", env=env)
    if code != 0:
        raise RuntimeError(f"训练失败: {config_path}")


def eval_best(config_path: str, checkpoint_dir: str, output_dir: str, test_file: str) -> Dict[str, Any]:
    ckpt = PROJECT_ROOT / checkpoint_dir / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))

    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "experiments/run_evaluation.py",
        "--config",
        config_path,
        "--checkpoint",
        str(ckpt),
        "--data-file",
        test_file,
        "--output-dir",
        str(out_dir),
        "--mode",
        "standard",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    stem = Path(config_path).stem
    code = run(cmd, log_path=PROJECT_ROOT / "artifacts" / "logs" / f"shs148k_{stem}_eval.log", env=env)
    if code != 0:
        raise RuntimeError(f"评估失败: {config_path}")

    results_path = out_dir / "evaluation_results_standard.json"
    if not results_path.exists():
        raise FileNotFoundError(str(results_path))
    return cast(Dict[str, Any], json.loads(results_path.read_text()))


def main() -> int:
    ensure_gpu_ready()
    ensure_splits()

    configs = [
        {
            "split": "random",
            "config": "configs/exp_llama3b_head_shs148k_random.yaml",
            "ckpt_dir": "artifacts/checkpoints/shs148k_llama3b_head_random",
            "results_dir": "artifacts/metrics/shs148k_llama3b_head_random",
            "test_file": "data/shs148k_llapa/processed/random_test.tsv",
        },
        {
            "split": "dfs",
            "config": "configs/exp_llama3b_head_shs148k_dfs.yaml",
            "ckpt_dir": "artifacts/checkpoints/shs148k_llama3b_head_dfs",
            "results_dir": "artifacts/metrics/shs148k_llama3b_head_dfs",
            "test_file": "data/shs148k_llapa/processed/dfs_test.tsv",
        },
        {
            "split": "bfs",
            "config": "configs/exp_llama3b_head_shs148k_bfs.yaml",
            "ckpt_dir": "artifacts/checkpoints/shs148k_llama3b_head_bfs",
            "results_dir": "artifacts/metrics/shs148k_llama3b_head_bfs",
            "test_file": "data/shs148k_llapa/processed/bfs_test.tsv",
        },
    ]

    for c in configs:
        smoke_train(c["config"])

    splits: Dict[str, Any] = {}
    summary: Dict[str, Any] = {"time": time.time(), "splits": splits}
    for c in configs:
        full_train(c["config"])
        metrics = eval_best(
            c["config"],
            c["ckpt_dir"],
            c["results_dir"],
            c["test_file"],
        )
        splits[str(c["split"])] = metrics

    out = PROJECT_ROOT / "artifacts/metrics/shs148k_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
