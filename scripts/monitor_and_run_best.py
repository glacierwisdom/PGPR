import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Metrics:
    micro_f1: Optional[float] = None
    auc: Optional[float] = None
    auprc: Optional[float] = None

    @staticmethod
    def from_json_path(p: Path) -> "Metrics":
        if not p.exists():
            return Metrics()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return Metrics()
        return Metrics(
            micro_f1=data.get("micro_f1"),
            auc=data.get("auc"),
            auprc=data.get("auprc"),
        )


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def list_user_jobs(user: str) -> List[Tuple[str, str, str]]:
    code, out = run_cmd(["squeue", "-u", user, "-h", "-o", "%i|%j|%T"])
    if code != 0:
        return []
    jobs = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        jobs.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return jobs


def wait_until_no_running_pipeline_jobs(poll_seconds: int, log) -> None:
    user = os.environ.get("USER", "")
    if not user:
        return
    while True:
        running = []
        for job_id, job_name, state in list_user_jobs(user):
            if state != "RUNNING":
                continue
            if job_name.startswith(("bfs_", "dfs_", "random_", "eval_")):
                running.append((job_id, job_name))
        if not running:
            return
        log("检测到仍在运行的训练/评估作业，等待其结束后再提交新作业: " + ",".join([f"{jid}:{jn}" for jid, jn in running]))
        time.sleep(poll_seconds)


def squeue_has_job(job_id: str) -> bool:
    code, out = run_cmd(["squeue", "-j", job_id, "-h", "-o", "%i"])
    if code != 0:
        return False
    return any(line.strip() == job_id for line in out.splitlines())

def get_squeue_state(job_id: str) -> Optional[Tuple[str, str]]:
    code, out = run_cmd(["squeue", "-j", job_id, "-h", "-o", "%T|%R"])
    if code != 0:
        return None
    line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    if not line:
        return None
    parts = line.split("|", 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def cancel_job(job_id: str) -> None:
    run_cmd(["scancel", job_id])



def wait_job(job_id: str, poll_seconds: int) -> None:
    while True:
        if not squeue_has_job(job_id):
            return
        time.sleep(poll_seconds)


def results_dir(split: str, variant: str) -> Path:
    return PROJECT_ROOT / "results" / f"{split}_blastp_full_fix_{variant}"


def results_json(split: str, variant: str) -> Path:
    return results_dir(split, variant) / "evaluation_results_standard.json"


def results_done(split: str, variant: str) -> bool:
    return (results_dir(split, variant) / "DONE").exists() and results_json(split, variant).exists()

def run_stage_until_done(
    *,
    title: str,
    split: str,
    variant: str,
    script: Path,
    poll_seconds: int,
    log,
    export_env: Optional[Dict[str, str]] = None,
    max_attempts: int = 2,
) -> Metrics:
    if results_done(split, variant):
        m = read_split_results(split, variant)
        log(f"{title} 已完成: {fmt_metrics(m)}")
        return m

    last = Metrics()
    for attempt in range(1, max_attempts + 1):
        log(f"{title} 开始执行 (attempt {attempt}/{max_attempts})")
        wait_until_no_running_pipeline_jobs(poll_seconds, log)
        jid = submit_sbatch(script, export_env=export_env)
        log(f"{title} job={jid}")
        wait_job(jid, poll_seconds)
        if results_done(split, variant):
            last = read_split_results(split, variant)
            log(f"{title} 完成: {fmt_metrics(last)}")
            return last
        last = read_split_results(split, variant)
        log(f"{title} 未产出结果，当前读到: {fmt_metrics(last)}")

    return last


def submit_sbatch(script_path: Path, dependency: Optional[str] = None, export_env: Optional[Dict[str, str]] = None) -> str:
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    if export_env:
        export_parts = ["ALL"]
        for k, v in export_env.items():
            export_parts.append(f"{k}={v}")
        cmd.append(f"--export={','.join(export_parts)}")
    cmd.append(str(script_path))
    code, out = run_cmd(cmd, cwd=PROJECT_ROOT)
    if code != 0:
        raise RuntimeError(f"sbatch failed for {script_path}: {out}")
    for tok in out.split():
        if tok.isdigit():
            return tok
    raise RuntimeError(f"cannot parse job id from: {out}")


def read_split_results(split: str, variant: str) -> Metrics:
    return Metrics.from_json_path(results_json(split, variant))


def fmt_metrics(m: Metrics) -> str:
    return f"micro_f1={m.micro_f1} auc={m.auc} auprc={m.auprc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-job-ids", nargs="*", default=["12534", "12535"])
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--log-file", default=str(PROJECT_ROOT / "logs" / "pipeline_monitor.log"))
    parser.add_argument("--max-dependency-wait-minutes", type=int, default=30)
    args = parser.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        print(line, end="")
        log_path.open("a", encoding="utf-8").write(line)

    wait_ids = [jid for jid in args.wait_job_ids if jid]
    if wait_ids:
        log(f"等待当前作业结束: {','.join(wait_ids)}")
        dependency_wait_start: Dict[str, float] = {}
        while True:
            alive: List[str] = []
            for jid in wait_ids:
                st = get_squeue_state(jid)
                if st is None:
                    continue
                state, reason = st
                alive.append(jid)
                if state == "PENDING" and ("DependencyNeverSatisfied" in reason):
                    if jid not in dependency_wait_start:
                        dependency_wait_start[jid] = time.time()
                    waited = time.time() - dependency_wait_start[jid]
                    if waited > args.max_dependency_wait_minutes * 60:
                        log(f"作业 {jid} 长时间处于依赖等待({reason})，自动取消以避免卡死")
                        cancel_job(jid)
            if not alive:
                break
            log(f"仍在运行/排队: {','.join(alive)}")
            time.sleep(args.poll_seconds)
        log("当前作业已结束（或已自动清理依赖卡住的作业），开始执行后续编排任务")

    bfs_stage1 = read_split_results("bfs", "rl_off_10ep")
    bfs_stage2 = read_split_results("bfs", "with_ppo_stage2")
    log(f"BFS stage1(rl_off_10ep): {fmt_metrics(bfs_stage1)}")
    log(f"BFS stage2(with_ppo_stage2): {fmt_metrics(bfs_stage2)}")

    if not results_done("bfs", "with_ppo_stage2"):
        run_stage_until_done(
            title="BFS stage2 eval",
            split="bfs",
            variant="with_ppo_stage2",
            script=PROJECT_ROOT / "run_eval_bfs_blastp_full_fix_with_ppo_stage2.sh",
            poll_seconds=args.poll_seconds,
            log=log,
            max_attempts=2,
        )
        bfs_stage2 = read_split_results("bfs", "with_ppo_stage2")
        log(f"BFS stage2(with_ppo_stage2): {fmt_metrics(bfs_stage2)}")

    if bfs_stage1.micro_f1 is not None and bfs_stage2.micro_f1 is not None and bfs_stage2.micro_f1 + 0.005 < bfs_stage1.micro_f1:
        log("BFS stage2 没有超过 stage1，提交 tuned stage2 训练+测试")
        jid = submit_sbatch(PROJECT_ROOT / "run_bfs_blastp_full_fix_with_ppo_stage2_tuned_fulltest.sh")
        log(f"BFS tuned stage2 job={jid}")
        wait_job(jid, args.poll_seconds)
        tuned = Metrics.from_json_path(PROJECT_ROOT / "results" / "bfs_blastp_full_fix_with_ppo_stage2_tuned" / "evaluation_results_standard.json")
        log(f"BFS tuned stage2: {fmt_metrics(tuned)}")
    else:
        log("BFS 维持当前 stage2 结果，不额外 rerun tuned")

    bfs_opt = run_stage_until_done(
        title="BFS stage2-opt (PPO optimized)",
        split="bfs",
        variant="with_ppo_stage2_opt",
        script=PROJECT_ROOT / "run_bfs_blastp_full_fix_with_ppo_stage2_opt_fulltest.sh",
        poll_seconds=args.poll_seconds,
        log=log,
        max_attempts=1,
    )

    candidates = [("stage1", bfs_stage1), ("stage2", bfs_stage2), ("stage2_opt", bfs_opt)]
    scored = [(tag, m) for tag, m in candidates if isinstance(m.micro_f1, (int, float))]
    if scored:
        best_tag, best_m = max(scored, key=lambda x: x[1].micro_f1)
        log(f"BFS BEST={best_tag} {fmt_metrics(best_m)}")

    for split in ["dfs", "random"]:
        m1 = run_stage_until_done(
            title=f"{split.upper()} stage1 (RL-off 10ep)",
            split=split,
            variant="rl_off_10ep",
            script=PROJECT_ROOT / f"run_{split}_blastp_full_fix_rl_off_10ep_fulltest.sh",
            poll_seconds=args.poll_seconds,
            log=log,
            max_attempts=2,
        )

        m2 = run_stage_until_done(
            title=f"{split.upper()} stage2 (PPO stage2)",
            split=split,
            variant="with_ppo_stage2",
            script=PROJECT_ROOT / f"run_{split}_blastp_full_fix_with_ppo_stage2_fulltest.sh",
            poll_seconds=args.poll_seconds,
            log=log,
            max_attempts=2,
        )

        m2_opt = run_stage_until_done(
            title=f"{split.upper()} stage2-opt (PPO optimized)",
            split=split,
            variant="with_ppo_stage2_opt",
            script=PROJECT_ROOT / f"run_{split}_blastp_full_fix_with_ppo_stage2_opt_fulltest.sh",
            poll_seconds=args.poll_seconds,
            log=log,
            max_attempts=1,
        )

        if m1.micro_f1 is not None and m2.micro_f1 is not None and m2.micro_f1 + 0.005 < m1.micro_f1:
            log(f"{split.upper()} stage2 不如 stage1，提交 tuned stage2 rerun")
            m3 = run_stage_until_done(
                title=f"{split.upper()} tuned stage2 (PPO stronger)",
                split=split,
                variant="with_ppo_stage2",
                script=PROJECT_ROOT / f"run_{split}_blastp_full_fix_with_ppo_stage2_fulltest.sh",
                poll_seconds=args.poll_seconds,
                log=log,
                export_env={"TRAIN_EPOCHS": "20", "PPO_EPOCHS": "4", "PPO_BATCH_SIZE": "32"},
                max_attempts=1,
            )
            log(f"{split.upper()} tuned stage2 results: {fmt_metrics(m3)}")

        candidates = [("stage1", m1), ("stage2", m2), ("stage2_opt", m2_opt)]
        scored = [(tag, m) for tag, m in candidates if isinstance(m.micro_f1, (int, float))]
        if scored:
            best_tag, best_m = max(scored, key=lambda x: x[1].micro_f1)
            log(f"{split.upper()} BEST={best_tag} {fmt_metrics(best_m)}")

    log("全部编排任务完成")


if __name__ == "__main__":
    main()
