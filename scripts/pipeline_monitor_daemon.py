import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


ERROR_PATTERNS: List[Tuple[str, str]] = [
    ("traceback", r"Traceback \(most recent call last\)"),
    ("python_error", r"^\w*(Error|Exception)\b"),
    ("process_raised_exception", r"ProcessRaisedException"),
    ("cuda_error", r"CUDA error:|CUDNN_STATUS_|NCCL error|nccl"),
    ("oom", r"out of memory|CUDA out of memory"),
    ("killed", r"Killed\b|SIGKILL|SIGTERM"),
]


METRIC_LINE_RE = re.compile(r"验证结果\s*-\s*(?P<body>.*)$")
EPOCH_RE = re.compile(r"Epoch\s+(?P<cur>\d+)\s*/\s*(?P<tot>\d+)")
BATCH_RE = re.compile(r"Batch\s+(?P<cur>\d+)\s*/\s*(?P<tot>\d+)\s*\((?P<pct>[\d.]+)%\)")


@dataclass
class MonitorState:
    start_ts: float
    last_log_path: Optional[str] = None
    last_log_offset: int = 0
    last_seen_epoch: Optional[Tuple[int, int]] = None
    last_seen_batch: Optional[Tuple[int, int, float]] = None
    last_metrics: Optional[Dict[str, float]] = None
    last_checkpoint_path: Optional[str] = None
    last_eval_json_path: Optional[str] = None
    last_heartbeat_ts: Optional[float] = None
    last_error_fingerprint: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_path(path: Path) -> "MonitorState":
        if not path.exists():
            return MonitorState(start_ts=time.time())
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            s = MonitorState(start_ts=float(data.get("start_ts", time.time())))
            for k, v in data.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            return s
        except Exception:
            return MonitorState(start_ts=time.time())


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_pid(pid_path: Path) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")


def pid_is_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_singleton(pid_path: Path, *, force: bool) -> None:
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            pid = 0
        if pid_is_alive(pid) and not force:
            raise SystemExit(f"[{now_ts()}] monitor already running pid={pid} (use --force to override)")
    write_pid(pid_path)


def find_latest_log(log_dir: Path) -> Optional[Path]:
    candidates = list(log_dir.glob("root_*.log"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def read_new_lines(path: Path, offset: int, max_bytes: int) -> Tuple[int, List[str]]:
    if not path.exists():
        return offset, []
    size = path.stat().st_size
    if offset > size:
        offset = 0
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(max_bytes)
        new_offset = f.tell()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return new_offset, lines


def parse_metrics(body: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    parts = [p.strip() for p in body.split("|") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def detect_error_fingerprint(lines: Iterable[str]) -> Optional[str]:
    text = "\n".join(lines[-200:])
    if not text.strip():
        return None
    for code, pat in ERROR_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE):
            sample = "\n".join(lines[-50:])
            fp = f"{code}:{hash(sample)}"
            return fp
    return None


def newest_checkpoint(checkpoints_root: Path) -> Optional[Path]:
    if not checkpoints_root.exists():
        return None
    best = None
    best_mtime = -1.0
    for p in checkpoints_root.rglob("*.pth"):
        try:
            st = p.stat()
        except Exception:
            continue
        if st.st_mtime > best_mtime:
            best = p
            best_mtime = st.st_mtime
    return best


def newest_checkpoint_since(checkpoints_root: Path, since_ts: float) -> Optional[Path]:
    if not checkpoints_root.exists():
        return None
    best = None
    best_mtime = -1.0
    for p in checkpoints_root.rglob("*.pth"):
        try:
            st = p.stat()
        except Exception:
            continue
        if st.st_mtime < since_ts:
            continue
        if st.st_mtime > best_mtime:
            best = p
            best_mtime = st.st_mtime
    return best


def newest_eval_json(results_root: Path) -> Optional[Path]:
    if not results_root.exists():
        return None
    best = None
    best_mtime = -1.0
    for p in results_root.rglob("evaluation_results_standard.json"):
        try:
            st = p.stat()
        except Exception:
            continue
        if st.st_mtime > best_mtime:
            best = p
            best_mtime = st.st_mtime
    return best


def newest_eval_json_since(results_root: Path, since_ts: float) -> Optional[Path]:
    if not results_root.exists():
        return None
    best = None
    best_mtime = -1.0
    for p in results_root.rglob("evaluation_results_standard.json"):
        try:
            st = p.stat()
        except Exception:
            continue
        if st.st_mtime < since_ts:
            continue
        if st.st_mtime > best_mtime:
            best = p
            best_mtime = st.st_mtime
    return best


def fmt_state(state: MonitorState) -> str:
    parts: List[str] = []
    if state.last_seen_epoch:
        cur, tot = state.last_seen_epoch
        parts.append(f"epoch={cur}/{tot}")
    if state.last_seen_batch:
        cur, tot, pct = state.last_seen_batch
        parts.append(f"batch={cur}/{tot}({pct:.1f}%)")
    if state.last_metrics:
        keys = ["val_f1_score", "val_loss", "micro_f1", "macro_f1"]
        picked = [(k, state.last_metrics.get(k)) for k in keys if k in state.last_metrics]
        if picked:
            parts.append("metrics=" + ",".join([f"{k}={v:.6f}" for k, v in picked if isinstance(v, (int, float))]))
    if state.last_checkpoint_path:
        parts.append(f"ckpt={Path(state.last_checkpoint_path).name}")
    if state.last_eval_json_path:
        parts.append(f"eval={Path(state.last_eval_json_path).parent.name}")
    return " ".join(parts) if parts else "no_progress_yet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=int, default=10)
    ap.add_argument("--log-dir", default=str(PROJECT_ROOT / "logs"))
    ap.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--checkpoints-dir", default=str(PROJECT_ROOT / "output" / "checkpoints"))
    ap.add_argument("--pid-file", default=str(PROJECT_ROOT / "logs" / "pipeline_monitor_daemon.pid"))
    ap.add_argument("--state-file", default=str(PROJECT_ROOT / "logs" / "pipeline_monitor_daemon.state.json"))
    ap.add_argument("--out-log", default=str(PROJECT_ROOT / "logs" / "pipeline_monitor_daemon.log"))
    ap.add_argument("--max-read-bytes", type=int, default=256 * 1024)
    ap.add_argument("--stale-seconds", type=int, default=300)
    ap.add_argument("--reset-start", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    pid_file = Path(args.pid_file)
    state_file = Path(args.state_file)
    out_log = Path(args.out_log)
    out_log.parent.mkdir(parents=True, exist_ok=True)

    acquire_singleton(pid_file, force=bool(args.force))
    state = MonitorState.from_path(state_file)
    if bool(args.reset_start):
        state = MonitorState(start_ts=time.time())

    def emit(line: str) -> None:
        msg = f"[{now_ts()}] {line}"
        print(msg, flush=True)
        out_log.open("a", encoding="utf-8").write(msg + "\n")

    emit("monitor started")

    while True:
        current_log: Optional[Path] = Path(state.last_log_path) if state.last_log_path else None
        latest_log = find_latest_log(log_dir)
        chosen_log: Optional[Path] = None

        if current_log and current_log.exists():
            chosen_log = current_log
            if latest_log and latest_log.exists():
                try:
                    if latest_log.stat().st_mtime > current_log.stat().st_mtime:
                        chosen_log = latest_log
                except Exception:
                    pass
        else:
            chosen_log = latest_log

        if chosen_log is None:
            emit("waiting for root_*.log ...")
            time.sleep(args.poll_seconds)
            continue

        log_path_str = str(chosen_log)
        if state.last_log_path != log_path_str:
            state.last_log_path = log_path_str
            state.last_log_offset = 0
            state.last_seen_epoch = None
            state.last_seen_batch = None
            state.last_metrics = None
            state.last_error_fingerprint = None
            emit(f"tracking log={chosen_log.name}")

        new_offset, lines = read_new_lines(chosen_log, state.last_log_offset, int(args.max_read_bytes))
        if lines:
            state.last_log_offset = new_offset
            state.last_heartbeat_ts = time.time()

            for ln in lines[-200:]:
                m = EPOCH_RE.search(ln)
                if m:
                    state.last_seen_epoch = (int(m.group("cur")), int(m.group("tot")))
                m = BATCH_RE.search(ln)
                if m:
                    state.last_seen_batch = (int(m.group("cur")), int(m.group("tot")), float(m.group("pct")))
                mm = METRIC_LINE_RE.search(ln)
                if mm:
                    metrics = parse_metrics(mm.group("body"))
                    if metrics:
                        state.last_metrics = metrics

            fp = detect_error_fingerprint(lines)
            if fp and fp != state.last_error_fingerprint:
                state.last_error_fingerprint = fp
                tail = "\n".join(lines[-40:])
                emit("ALERT detected_error\n" + tail)

        ckpt = newest_checkpoint_since(checkpoints_dir, state.start_ts)
        if ckpt and str(ckpt) != state.last_checkpoint_path:
            state.last_checkpoint_path = str(ckpt)
            emit(f"new checkpoint: {ckpt}")

        eval_json = newest_eval_json_since(results_dir, state.start_ts)
        if eval_json and str(eval_json) != state.last_eval_json_path:
            state.last_eval_json_path = str(eval_json)
            emit(f"new eval json: {eval_json}")

        hb = state.last_heartbeat_ts or state.start_ts
        if time.time() - hb > int(args.stale_seconds):
            emit(f"ALERT no_log_update_for>{args.stale_seconds}s ({fmt_state(state)})")
            state.last_heartbeat_ts = time.time()

        state_file.write_text(state.to_json(), encoding="utf-8")
        emit(f"status {fmt_state(state)}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
