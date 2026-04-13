import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.now().isoformat()} =====\n")
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
        return p.wait()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    experiments = [
        ("llama3b_head_random", "configs/exp_llama3b_head_random.yaml"),
        ("llama3b_head_bfs", "configs/exp_llama3b_head_bfs.yaml"),
        ("llama3b_head_dfs", "configs/exp_llama3b_head_dfs.yaml"),
        ("llama3b_text_random", "configs/exp_llama3b_text_random.yaml"),
        ("llama3b_text_bfs", "configs/exp_llama3b_text_bfs.yaml"),
        ("llama3b_text_dfs", "configs/exp_llama3b_text_dfs.yaml"),
    ]

    for name, cfg in experiments:
        cfg_path = (repo_root / cfg).resolve()
        log_path = repo_root / "logs" / f"exp_{name}.log"

        ckpt_dir = repo_root / "output" / "checkpoints" / name
        best_ckpt = ckpt_dir / "best_model.pth"

        if best_ckpt.exists():
            continue

        cmd = [
            python,
            str(repo_root / "main.py"),
            "train",
            "--config",
            str(cfg_path),
            "--model-config",
            str(repo_root / "configs" / "model.yaml"),
            "--data-config",
            str(repo_root / "configs" / "data.yaml"),
        ]

        code = run(cmd, log_path)
        if code != 0:
            return code

        if not best_ckpt.exists():
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
