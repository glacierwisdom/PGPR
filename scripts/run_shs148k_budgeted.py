import os
import sys
import subprocess
from pathlib import Path
import json
import time
import signal
import socket
import random
import argparse
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


def ensure_gpu_ready(required_gpus: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    if not hasattr(torch.distributed, "is_nccl_available") or not torch.distributed.is_nccl_available():
        raise RuntimeError("当前 PyTorch 不支持 NCCL，无法多卡训练")
    if torch.cuda.device_count() < required_gpus:
        raise RuntimeError(f"GPU 数量不足: {torch.cuda.device_count()} < {required_gpus}")


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


def smoke_train(config_path: str, log_prefix: str) -> None:
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
    code = run(cmd, log_path=PROJECT_ROOT / "artifacts" / "logs" / f"{log_prefix}_smoke.log", env=env)
    if code != 0:
        raise RuntimeError(f"smoke 训练失败: {config_path}")


def full_train(config_path: str, log_prefix: str, overrides: List[str]) -> None:
    merged = list(overrides)
    merged.append(f"distributed.master_port={pick_free_port()}")
    cmd = [
        sys.executable,
        "experiments/run_training.py",
        "--config",
        config_path,
        "--config-override",
        *merged,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    code = run(cmd, log_path=PROJECT_ROOT / "artifacts" / "logs" / f"{log_prefix}_train.log", env=env)
    if code != 0:
        raise RuntimeError(f"训练失败: {config_path}")


def eval_best(
    config_path: str,
    checkpoint_dir: str,
    output_dir: str,
    test_file: str,
    log_prefix: str,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
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
    if overrides:
        cmd.extend(["--config-override", *overrides])
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    code = run(cmd, log_path=PROJECT_ROOT / "artifacts" / "logs" / f"{log_prefix}_eval.log", env=env)
    if code != 0:
        raise RuntimeError(f"评估失败: {config_path}")

    results_path = out_dir / "evaluation_results_standard.json"
    if not results_path.exists():
        raise FileNotFoundError(str(results_path))
    return cast(Dict[str, Any], json.loads(results_path.read_text()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--enable-rl", action="store_true", default=False)
    parser.add_argument("--skip-smoke", action="store_true", default=False)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--llm-max-length", type=int, default=384)
    parser.add_argument("--splits", type=str, default="random,dfs,bfs")
    parser.add_argument("--only-eval", action="store_true", default=False)
    parser.add_argument("--skip-train-if-checkpoint-exists", action="store_true", default=False)
    args = parser.parse_args()

    ensure_gpu_ready(required_gpus=int(args.world_size))
    ensure_splits()

    run_tag = f"shs148k_budget_e{int(args.epochs)}"
    configs = [
        {
            "split": "random",
            "config": "configs/exp_llama3b_head_shs148k_random.yaml",
            "test_file": "data/shs148k_llapa/processed/random_test.tsv",
        },
        {
            "split": "dfs",
            "config": "configs/exp_llama3b_head_shs148k_dfs.yaml",
            "test_file": "data/shs148k_llapa/processed/dfs_test.tsv",
        },
        {
            "split": "bfs",
            "config": "configs/exp_llama3b_head_shs148k_bfs.yaml",
            "test_file": "data/shs148k_llapa/processed/bfs_test.tsv",
        },
    ]

    splits: Dict[str, Any] = {}
    summary: Dict[str, Any] = {
        "time": time.time(),
        "tag": run_tag,
        "budget": {
            "epochs": int(args.epochs),
            "early_stopping_patience": int(args.early_stopping_patience),
            "enable_rl": bool(args.enable_rl),
            "world_size": int(args.world_size),
            "eval_batch_size": int(args.eval_batch_size),
            "val_interval": int(args.val_interval),
            "num_chains": int(args.num_chains),
            "max_steps": int(args.max_steps),
            "llm_max_length": int(args.llm_max_length),
            "splits": str(args.splits),
            "only_eval": bool(args.only_eval),
            "skip_train_if_checkpoint_exists": bool(args.skip_train_if_checkpoint_exists),
        },
        "splits": splits,
    }

    selected_splits = {s.strip() for s in str(args.splits).split(",") if s.strip()}
    for c in configs:
        split = str(c["split"])
        if selected_splits and split not in selected_splits:
            continue
        log_prefix = f"{run_tag}_{split}"

        ckpt_dir = f"artifacts/checkpoints/{run_tag}_{split}"
        results_dir = f"artifacts/metrics/{run_tag}_{split}"
        ckpt_path = PROJECT_ROOT / ckpt_dir / "best_model.pth"

        overrides = [
            f"training.epochs={int(args.epochs)}",
            f"training.early_stopping_patience={int(args.early_stopping_patience)}",
            f"training.validation.val_interval={int(args.val_interval)}",
            f"distributed.world_size={int(args.world_size)}",
            f"callbacks.checkpoint.dir={ckpt_dir}",
            f"paths.checkpoints_dir={ckpt_dir}",
            f"paths.results_dir={results_dir}",
            f"evaluation.batch_size={int(args.eval_batch_size)}",
            f"model.num_chains={int(args.num_chains)}",
            f"model.max_steps={int(args.max_steps)}",
            f"llm.max_length={int(args.llm_max_length)}",
        ]

        if not bool(args.enable_rl):
            overrides.append("reinforcement_learning.enabled=false")

        if not bool(args.skip_smoke):
            smoke_train(c["config"], log_prefix=log_prefix)

        if not bool(args.only_eval):
            if bool(args.skip_train_if_checkpoint_exists) and ckpt_path.exists():
                pass
            else:
                full_train(c["config"], log_prefix=log_prefix, overrides=overrides)

        eval_overrides = [
            f"model.num_chains={int(args.num_chains)}",
            f"model.max_steps={int(args.max_steps)}",
            f"llm.max_length={int(args.llm_max_length)}",
            f"evaluation.batch_size={int(args.eval_batch_size)}",
        ]
        metrics = eval_best(
            c["config"],
            checkpoint_dir=ckpt_dir,
            output_dir=results_dir,
            test_file=str(c["test_file"]),
            log_prefix=log_prefix,
            overrides=eval_overrides,
        )
        splits[split] = metrics

    out = PROJECT_ROOT / f"artifacts/metrics/{run_tag}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
