from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


SERVER_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT = SERVER_ROOT / "code" / "run_video_pipeline.py"
RUNS_ROOT = SERVER_ROOT / "runs_ui"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _read_kv_sections(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    root_vals: dict[str, str] = {}
    counts: dict[str, str] = {}
    if not path.exists():
        return root_vals, counts
    section = "root"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "[counts]":
            section = "counts"
            continue
        if line.startswith("[") and line.endswith("]"):
            section = "other"
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if section == "counts":
            counts[key.strip()] = value.strip()
        elif section == "root":
            root_vals[key.strip()] = value.strip()
    return root_vals, counts


def _read_team_counts(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        row: dict[str, Any] = {"frame_key": parts[0]}
        for token in parts[1:]:
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            try:
                row[k] = int(v)
            except ValueError:
                row[k] = v
        out.append(row)
    return out


def _video_meta(video_path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"original_fps": 0.0, "duration_sec": 0.0, "frame_count": 0.0}
    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS), 0.0)
    frame_count = _safe_float(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0.0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {"original_fps": fps, "duration_sec": duration, "frame_count": frame_count}


def _to_file_url(path: Path) -> str:
    rel = path.resolve().relative_to(RUNS_ROOT.resolve())
    return f"/files/{rel.as_posix()}"


def _frame_key_to_src_idx(frame_key: str) -> int | None:
    m = re.match(r"^frame_(\d+)$", frame_key)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _build_frame_index(run: "RunState") -> list[dict[str, Any]]:
    frames_root = run.out_dir / "frames"
    if not frames_root.exists():
        return []
    frame_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir() and p.name.startswith("frame_")], key=lambda p: p.name)
    rows: list[dict[str, Any]] = []
    for fd in frame_dirs:
        frame_key = fd.name
        src_idx = _frame_key_to_src_idx(frame_key)
        original_img = fd / f"{frame_key}.jpg"
        field_img = fd / f"{frame_key}_field.jpg"
        pocket_img = run.out_dir / "logs" / "pocket_vis" / f"{frame_key}_pocket.jpg"
        txt_dir = fd / "txt"
        ts = (src_idx / run.original_fps) if (src_idx is not None and run.original_fps > 0) else None
        rows.append(
            {
                "frame_key": frame_key,
                "source_frame_index": src_idx,
                "timestamp_sec": ts,
                "original_image_url": _to_file_url(original_img) if original_img.exists() else None,
                "field_image_url": _to_file_url(field_img) if field_img.exists() else None,
                "pocket_image_url": _to_file_url(pocket_img) if pocket_img.exists() else None,
                "overlay_txt_urls": {
                    "players": _to_file_url(txt_dir / "players.txt") if (txt_dir / "players.txt").exists() else None,
                    "pocket": _to_file_url(txt_dir / "pocket.txt") if (txt_dir / "pocket.txt").exists() else None,
                    "yardline_line": _to_file_url(txt_dir / "yardline_line.txt") if (txt_dir / "yardline_line.txt").exists() else None,
                    "harshmark_line": _to_file_url(txt_dir / "harshmark_line.txt") if (txt_dir / "harshmark_line.txt").exists() else None,
                },
            }
        )
    return rows


@dataclass
class RunState:
    run_id: str
    video_path: Path
    process_fps: float
    out_dir: Path
    run_dir: Path
    created_at: str
    original_fps: float = 0.0
    status: str = "queued"
    error: str | None = None
    pid: int | None = None
    completed_at: str | None = None
    expected_sampled_frames: int = 0
    process: subprocess.Popen[str] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def log_path(self) -> Path:
        return self.run_dir / "pipeline.log"

    @property
    def notes_path(self) -> Path:
        return self.run_dir / "notes.json"

    @property
    def meta_path(self) -> Path:
        return self.run_dir / "run_meta.json"

    def to_json(self) -> dict[str, Any]:
        frame_count = len(_build_frame_index(self))
        progress = 0.0
        if self.status == "completed":
            progress = 1.0
        elif self.expected_sampled_frames > 0:
            progress = min(1.0, frame_count / self.expected_sampled_frames)
        return {
            "run_id": self.run_id,
            "status": self.status,
            "video_path": str(self.video_path),
            "process_fps": self.process_fps,
            "original_fps": self.original_fps,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "out_dir": str(self.out_dir),
            "pid": self.pid,
            "processed_sampled_frames": frame_count,
            "expected_sampled_frames": self.expected_sampled_frames,
            "progress": progress,
        }

    def persist_meta(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")
        if not self.notes_path.exists():
            self.notes_path.write_text("[]\n", encoding="utf-8")


class RunCreateRequest(BaseModel):
    video_path: str = Field(..., description="Absolute or project-relative path on server machine.")
    process_fps: float = Field(..., gt=0)


class NoteCreateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    scope: str = Field(..., pattern="^(overall|frame)$")
    frame_key: str | None = None
    timestamp_sec: float | None = None


app = FastAPI(title="AutoScout API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=str(RUNS_ROOT), check_dir=False), name="files")

runs_lock = threading.Lock()
runs: dict[str, RunState] = {}
active_run_id: str | None = None


def get_run_or_404(run_id: str) -> RunState:
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


def _worker(run: RunState) -> None:
    global active_run_id
    with run._lock:
        run.status = "running"
        run.persist_meta()

    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--video",
        str(run.video_path),
        "--fps",
        str(run.process_fps),
        "--out-dir",
        str(run.out_dir),
        "--save-team-counts",
        "--formation-analysis",
    ]

    with run.log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(SERVER_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        with run._lock:
            run.process = proc
            run.pid = proc.pid
            run.persist_meta()
        code = proc.wait()

    with run._lock:
        run.completed_at = utc_now_iso()
        if code == 0:
            run.status = "completed"
            run.error = None
        else:
            run.status = "failed"
            run.error = f"Pipeline exited with code {code}"
        run.persist_meta()

    with runs_lock:
        if active_run_id == run.run_id:
            active_run_id = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs")
def create_run(payload: RunCreateRequest) -> dict[str, Any]:
    global active_run_id

    video_path = Path(payload.video_path).expanduser()
    if not video_path.is_absolute():
        video_path = (SERVER_ROOT / video_path).resolve()
    if not video_path.exists():
        raise HTTPException(status_code=400, detail=f"Video path not found: {video_path}")

    with runs_lock:
        if active_run_id is not None:
            active = runs.get(active_run_id)
            if active and active.status in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="Another run is currently in progress.")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        run_dir = RUNS_ROOT / run_id
        out_dir = run_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = _video_meta(video_path)
        expected = int(max(1.0, round(meta["duration_sec"] * payload.process_fps))) if meta["duration_sec"] > 0 else 0

        run = RunState(
            run_id=run_id,
            video_path=video_path.resolve(),
            process_fps=float(payload.process_fps),
            out_dir=out_dir.resolve(),
            run_dir=run_dir.resolve(),
            created_at=utc_now_iso(),
            original_fps=meta["original_fps"],
            expected_sampled_frames=expected,
        )
        run.persist_meta()
        runs[run_id] = run
        active_run_id = run_id

    thread = threading.Thread(target=_worker, args=(run,), daemon=True)
    thread.start()
    return run.to_json()


@app.get("/api/runs/active")
def get_active_run() -> dict[str, Any]:
    if active_run_id is None:
        return {"active_run_id": None}
    run = runs.get(active_run_id)
    if run is None:
        return {"active_run_id": None}
    return {"active_run_id": run.run_id, "run": run.to_json()}


@app.get("/api/runs/{run_id}/status")
def get_run_status(run_id: str) -> dict[str, Any]:
    run = get_run_or_404(run_id)
    return run.to_json()


@app.get("/api/runs/{run_id}/frames")
def get_run_frames(run_id: str) -> dict[str, Any]:
    run = get_run_or_404(run_id)
    frames = _build_frame_index(run)
    return {"run_id": run_id, "count": len(frames), "frames": frames}


@app.get("/api/runs/{run_id}/summary")
def get_run_summary(run_id: str) -> dict[str, Any]:
    run = get_run_or_404(run_id)

    formation_path = run.out_dir / "formation" / "formation_summary.txt"
    root_vals, counts = _read_kv_sections(formation_path)
    team_counts_path = run.out_dir / "logs" / "team_counts.txt"
    team_rows = _read_team_counts(team_counts_path)

    selected_vals = [int(row.get("selected_total", 0)) for row in team_rows if isinstance(row.get("selected_total", 0), int)]
    avg_selected = (sum(selected_vals) / len(selected_vals)) if selected_vals else 0.0

    return {
        "run": run.to_json(),
        "formation": {
            "summary_url": _to_file_url(formation_path) if formation_path.exists() else None,
            "offense_team": root_vals.get("offense_team"),
            "defense_team": root_vals.get("defense_team"),
            "x_los": _safe_float(root_vals.get("x_los"), 0.0),
            "y_mid": _safe_float(root_vals.get("y_mid"), 0.0),
            "counts": {k: int(_safe_float(v, 0.0)) for k, v in counts.items()},
        },
        "team_counts": team_rows,
        "stats": {
            "avg_selected": avg_selected,
            "team_count_rows": len(team_rows),
        },
    }


@app.get("/api/runs/{run_id}/assets")
def get_run_assets(run_id: str) -> dict[str, Any]:
    run = get_run_or_404(run_id)
    formation_img = None
    formation_dir = run.out_dir / "formation"
    if formation_dir.exists():
        imgs = sorted(formation_dir.glob("*_formation.jpg"))
        if imgs:
            formation_img = _to_file_url(imgs[0])
    field_video = None
    video_dir = run.out_dir / "videos"
    if video_dir.exists():
        vids = sorted(video_dir.glob("*.mp4"))
        if vids:
            field_video = _to_file_url(vids[0])
    return {
        "run_id": run_id,
        "formation_image_url": formation_img,
        "field_video_url": field_video,
    }


@app.get("/api/runs/{run_id}/notes")
def get_notes(run_id: str) -> dict[str, Any]:
    run = get_run_or_404(run_id)
    if not run.notes_path.exists():
        return {"run_id": run_id, "notes": []}
    notes = json.loads(run.notes_path.read_text(encoding="utf-8"))
    return {"run_id": run_id, "notes": notes}


@app.post("/api/runs/{run_id}/notes")
def create_note(run_id: str, payload: NoteCreateRequest) -> dict[str, Any]:
    run = get_run_or_404(run_id)
    notes: list[dict[str, Any]]
    if run.notes_path.exists():
        notes = json.loads(run.notes_path.read_text(encoding="utf-8"))
    else:
        notes = []
    note = {
        "id": uuid.uuid4().hex[:10],
        "created_at": utc_now_iso(),
        "scope": payload.scope,
        "text": payload.text.strip(),
        "frame_key": payload.frame_key,
        "timestamp_sec": payload.timestamp_sec,
    }
    if payload.scope == "overall":
        # Keep exactly one overall note; submitting again updates it.
        existing = next((n for n in notes if n.get("scope") == "overall"), None)
        if existing is not None:
            existing["text"] = note["text"]
            existing["created_at"] = note["created_at"]
            existing["frame_key"] = None
            existing["timestamp_sec"] = None
            note = existing
        else:
            notes.append(note)
    else:
        notes.append(note)
    run.notes_path.write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    return {"run_id": run_id, "note": note}


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    items = sorted(runs.values(), key=lambda r: r.created_at, reverse=True)
    return {"runs": [r.to_json() for r in items]}


if __name__ == "__main__":
    try:
        import uvicorn  # type: ignore
    except Exception as exc:
        raise SystemExit("uvicorn is required. Install with: pip install fastapi uvicorn") from exc
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
