#!/usr/bin/env python3
"""LTX-2.5 video generation web server (Intel Arc B70 / XPU).

Serves a demo UI to trigger LTX-2.5 generation via the bundled ltx-gen.sh
subprocess. All state lives server-side; the browser is a thin polling client.

Endpoints:
    GET  /                   -> static/index.html
    POST /api/generate       -> start a job {mode: "8"|"50", prompt}
    GET  /api/status         -> current server/job state
    GET  /api/log?since=<n>  -> incremental log lines (counter-based)
    GET  /api/videos         -> list of generated videos
    GET  /output/<file>      -> serve video files
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
GEN_SCRIPT = ROOT / "ltx-gen.sh"
STATIC_DIR = ROOT / "static"
OUTPUT_DIR = ROOT / "output"
# StaticFiles raises at import time if the directory is missing, and `output/` is
# not tracked in git, so a fresh clone would fail to start the server.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LOG_LINES = 3000
LOG_POLL = 0.5

app = FastAPI(title="LTX-2.5 Generation Server")


class ServerState:
    """All mutable server-side state. Locked by an asyncio.Lock-equivalent mutex."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "idle"  # idle | running | done | error
        self.mode = "50"
        self.prompt = ""
        self.frames = None
        self.job_id = ""
        self.video = None  # filename relative to OUTPUT_DIR
        self.error_msg = None
        self.started_at = None
        self.finished_at = None
        self._log = []  # list[str]  (completed lines, in order)
        self._seq = 0  # total completed lines ever appended (monotonic counter)
        self._progress = ""  # current in-progress line (updated via \r)

    def append_log(self, text: str) -> None:
        with self._lock:
            for line in text.splitlines():
                self._log.append(line)
                self._seq += 1
            if len(self._log) > MAX_LOG_LINES:
                self._log = self._log[-MAX_LOG_LINES:]

    def update_progress(self, line: str) -> None:
        """Replace the current in-progress (carriage-return updated) line."""
        with self._lock:
            self._progress = line

    def clear_progress(self) -> None:
        with self._lock:
            self._progress = ""

    def reset_log(self) -> None:
        """Start a fresh log for a new job (clears history and counters)."""
        with self._lock:
            self._log = []
            self._seq = 0
            self._progress = ""

    def get_log(self, since: int) -> dict:
        with self._lock:
            # If the client's counter is behind the trimmed window, hand it
            # everything we still hold and tell it the new baseline.
            head = max(0, self._seq - len(self._log))
            start = max(since, head)
            lines = self._log[start - head:]
            return {
                "seq": self._seq,
                "head": head,
                "lines": lines,
                "progress": self._progress,
            }

    def state(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "mode": self.mode,
                "prompt": self.prompt,
                "frames": self.frames,
                "job_id": self.job_id,
                "video": self.video,
                "error": self.error_msg,
                "running": self.status == "running",
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "log_seq": self._seq,
            }


STATE = ServerState()


class GenerateRequest(BaseModel):
    mode: str = Field(pattern="^(8|50)$")
    prompt: str = Field(min_length=1, max_length=2000)
    frames: int | None = Field(default=None, ge=1, le=1024)


def snap_frames_to_grid(frames: int) -> int:
    """Snap a frame count to the LTX-2 VAE temporal grid (num_frames = 8k + 1).

    Makes user-supplied frame counts valid for the diffusion model; the raw
    value is floored to the nearest 8k+1 point (min 1).
    """
    return ((max(frames, 1) - 1) // 8) * 8 + 1


def _stream_lines(proc: asyncio.subprocess.Process, tag: str) -> None:
    """Read a subprocess stream and push completed lines plus live progress.

    tqdm progress bars are redrawn in place with `\r` (no newline). To show
    them live, a `\r` replaces the current in-progress line (STATE._progress)
    while `\n` finalises it as a completed log line.
    """

    async def _drain(stream) -> None:
        buf = bytearray()
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                # Finalise everything up to a newline as a completed line.
                nl = buf.find(b"\n")
                if nl == -1:
                    break
                line = bytes(buf[:nl])
                del buf[: nl + 1]
                text = line.decode("utf-8", errors="replace").rstrip("\r")
                STATE.append_log(text + "\n")
            # Any remaining \r-separated segment is live progress.
            if buf:
                cr = buf.rfind(b"\r")
                if cr != -1:
                    segment = bytes(buf[cr + 1 :])
                else:
                    segment = bytes(buf)
                del buf[:]
                if segment:
                    STATE.update_progress(
                        segment.decode("utf-8", errors="replace")
                    )
        STATE.clear_progress()

    asyncio.ensure_future(_drain(getattr(proc, tag)))


async def _run_job(mode: str, prompt: str, out_name: str, job_id: str, frames: int | None = None) -> None:
    """Run a single generation job on the server's event loop."""
    args = []
    if mode == "8":
        args = ["--distilled"]
    args += ["--prompt", prompt, "--output-path", str(OUTPUT_DIR / out_name)]
    if frames is not None:
        args += ["--num-frames", str(frames)]

    STATE.append_log(f"\n=== JOB {job_id} :: {mode}-step mode ===")
    STATE.append_log(f"prompt: {prompt}\n")
    if frames is not None:
        STATE.append_log(f"frames: {frames} ({frames / 24.0:.2f}s @ 24fps)\n")

    try:
        proc = await asyncio.create_subprocess_exec(
            str(GEN_SCRIPT),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ROOT),
        )
    except Exception as exc:  # e.g. script not executable
        STATE.status = "error"
        STATE.error_msg = f"Failed to launch ltx-gen.sh: {exc}"
        STATE.append_log(f"ERROR: {exc}\n")
        return

    _stream_lines(proc, "stdout")
    _stream_lines(proc, "stderr")

    rc = await proc.wait()
    if rc == 0:
        STATE.status = "done"
        STATE.video = out_name
        STATE.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        STATE.append_log(f"\n=== DONE ({mode}-step) -> {out_name} ===\n")
    else:
        STATE.status = "error"
        STATE.error_msg = f"Generation failed with exit code {rc}"
        STATE.append_log(f"\n=== ERROR: exit code {rc} ===\n")


_job_lock = asyncio.Lock()  # one generation at a time


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if not GEN_SCRIPT.exists():
        raise HTTPException(500, "ltx-gen.sh not found")
    if _job_lock.locked():
        raise HTTPException(409, "A generation job is already running")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    job_id = f"{stamp}-{os.getpid()}"
    out_name = f"ltx_{stamp}.mp4"

    # Snap any user-supplied frame count to the VAE's 8k+1 grid so it is
    # always a valid length for the diffusion model.
    frames = snap_frames_to_grid(req.frames) if req.frames is not None else None

    # Mark state as running immediately, then run the job in a background task
    # so the request returns at once and the UI can flip idle -> running instantly.
    STATE.reset_log()  # fresh log + counters for this job
    STATE.status = "running"
    STATE.mode = req.mode
    STATE.prompt = req.prompt
    STATE.frames = frames
    STATE.job_id = job_id
    STATE.video = None
    STATE.error_msg = None
    STATE.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    STATE.finished_at = None
    STATE.append_log(
        f"[{time.strftime('%H:%M:%S')}] job {job_id} queued ({req.mode}-step)\n"
    )

    asyncio.create_task(
        _run_job_locked(req.mode, req.prompt, out_name, job_id, frames)
    )

    return {"job_id": job_id, "output": out_name, "frames": frames}


async def _run_job_locked(mode: str, prompt: str, out_name: str, job_id: str, frames: int | None = None) -> None:
    """Acquire the job lock and run the generation as a background task."""
    async with _job_lock:
        await _run_job(mode, prompt, out_name, job_id, frames)


@app.get("/api/status")
def status():
    return STATE.state()


@app.get("/api/log")
def log(since: int = 0):
    return STATE.get_log(since)


@app.get("/api/videos")
def videos():
    if not OUTPUT_DIR.exists():
        return []
    files = sorted(
        (p.name for p in OUTPUT_DIR.iterdir() if p.suffix.lower() == ".mp4"),
        reverse=True,
    )
    return files


# Expose the output directory for <video> playback and downloads.
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="LTX-2.5 generation web server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
