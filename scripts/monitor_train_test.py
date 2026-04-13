import argparse
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RunSpec:
    gpus: int
    world_size: int
    use_distributed: bool


def _run(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def _submit_job(script_path: str, export_env: Dict[str, str], sbatch_overrides: Optional[List[str]] = None) -> int:
    env_kv = ",".join([f"{k}={v}" for k, v in export_env.items()])
    cmd = ["sbatch", "--export", f"ALL,{env_kv}"]
    if sbatch_overrides:
        cmd.extend(sbatch_overrides)
    cmd.append(script_path)
    rc, out = _run(cmd)
    if rc != 0:
        raise RuntimeError(out.strip())
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(out.strip())
    return int(m.group(1))


def _job_state(job_id: int) -> str:
    rc, out = _run(["squeue", "-j", str(job_id), "-h", "-o", "%T"])
    if rc != 0:
        return "UNKNOWN"
    state = out.strip()
    if not state:
        return "GONE"
    return state


def _log_paths(job_id: int, job_name_prefix: str, log_dir: str) -> Tuple[Path, Path]:
    log_dir_path = Path(log_dir)
    return (
        log_dir_path / f"{job_name_prefix}_{job_id}.log",
        log_dir_path / f"{job_name_prefix}_{job_id}.err",
    )


def _tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 8192
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                step = block if size >= block else size
                size -= step
                f.seek(size)
                data = f.read(step) + data
            lines = data.decode("utf-8", errors="replace").splitlines()[-n:]
            return "\n".join(lines)
    except Exception:
        return ""


def _detect_failure(err_text: str, log_text: str) -> Optional[str]:
    text = "\n".join([err_text, log_text])
    patterns = [
        ("ddp_missing_method", r"DistributedDataParallel.*has no attribute 'batch_generate_chains'"),
        ("weights_only_unpickling", r"Weights only load failed|WeightsUnpickler error"),
        ("makeblastdb_failed", r"makeblastdb failed|Command \[\'makeblastdb\'"),
        ("dtype_mismatch", r"mat1 and mat2 must have the same dtype"),
        ("ppo_backward_twice", r"Trying to backward through the graph a second time"),
        ("qos_gpu_limit", r"QOSGrpGRES"),
        ("nccl_store_broken_pipe", r"Broken pipe|recvValue failed|TCPStore"),
    ]
    for code, pat in patterns:
        if re.search(pat, text):
            return code
    return None


def _done_marker(done_file: str) -> bool:
    return Path(done_file).exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="sbatch script path (e.g. run_bfs_blastp_full.sh)")
    ap.add_argument("--job-name-prefix", default="bfs_blastp_full", help="log filename prefix")
    ap.add_argument("--log-dir", default="artifacts/logs")
    ap.add_argument("--done-file", default="artifacts/metrics/bfs_blastp_full/DONE")
    ap.add_argument("--poll-seconds", type=int, default=20)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--start-from-job", type=int, default=0, help="monitor existing job id if >0")
    args = ap.parse_args()

    script_path = os.path.abspath(args.script)
    base_env = {}

    specs = [
        RunSpec(gpus=4, world_size=4, use_distributed=True),
        RunSpec(gpus=2, world_size=2, use_distributed=True),
        RunSpec(gpus=1, world_size=1, use_distributed=False),
    ]

    attempt = 0
    job_id = args.start_from_job if args.start_from_job > 0 else 0

    while True:
        if _done_marker(args.done_file):
            print(f"[monitor] DONE file exists: {args.done_file}")
            return

        if job_id <= 0:
            spec = specs[min(attempt, len(specs) - 1)]
            export_env = dict(base_env)
            export_env["WORLD_SIZE_OVERRIDE"] = str(spec.world_size)
            export_env["USE_DISTRIBUTED_OVERRIDE"] = "true" if spec.use_distributed else "false"
            export_env["TRAIN_EPOCHS_OVERRIDE"] = os.environ.get("TRAIN_EPOCHS_OVERRIDE", "")
            sbatch_overrides = [
                f"--gres=gpu:rtx_3090:{spec.gpus}",
                "--cpus-per-task=32",
            ]
            print(f"[monitor] submitting job (attempt={attempt+1}) gpus={spec.gpus} world_size={spec.world_size} use_distributed={spec.use_distributed}")
            job_id = _submit_job(script_path, export_env, sbatch_overrides=sbatch_overrides)
            print(f"[monitor] submitted job_id={job_id}")

        log_path, err_path = _log_paths(job_id, args.job_name_prefix, args.log_dir)

        while True:
            if _done_marker(args.done_file):
                print(f"[monitor] DONE file exists: {args.done_file}")
                return

            state = _job_state(job_id)
            if state in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING"):
                log_tail = _tail(log_path, n=10)
                err_tail = _tail(err_path, n=10)
                if log_tail:
                    print(f"[monitor] job_id={job_id} state={state}\n{log_tail}\n")
                elif err_tail:
                    print(f"[monitor] job_id={job_id} state={state}\n{err_tail}\n")
                time.sleep(args.poll_seconds)
                continue

            log_tail = _tail(log_path, n=80)
            err_tail = _tail(err_path, n=120)
            failure_code = _detect_failure(err_tail, log_tail)

            if _done_marker(args.done_file):
                print(f"[monitor] DONE file exists: {args.done_file}")
                return

            print(f"[monitor] job_id={job_id} state={state} done_file_missing failure_code={failure_code}")
            if attempt >= args.max_retries:
                raise SystemExit(f"[monitor] exceeded max retries. last job_id={job_id} failure_code={failure_code}")

            attempt += 1
            job_id = 0
            time.sleep(5)
            break


if __name__ == "__main__":
    main()
