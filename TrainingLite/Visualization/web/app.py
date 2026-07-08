#!/usr/bin/env python3
"""FastAPI application for the state comparison visualization webapp."""

import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(VIS_DIR)
sys.path.insert(0, PARENT_DIR)

from browser_session import touch_browser_heartbeat
from visualization_service import VisualizationService

app = FastAPI(title="State Comparison Visualizer")
service = VisualizationService()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.on_event("startup")
def _open_browser_when_ready() -> None:
    """Open browser only after the server is accepting connections."""
    url = os.environ.pop("VIZ_OPEN_BROWSER", None)
    if url:
        import webbrowser
        webbrowser.open(url, new=0)


@app.post("/api/browser/heartbeat")
def browser_heartbeat() -> Dict[str, str]:
    touch_browser_heartbeat()
    return {"status": "ok"}


class SettingsUpdate(BaseModel):
    csv_file_path: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    horizon_steps: Optional[int] = None
    steering_delay_steps: Optional[int] = None
    acceleration_delay_steps: Optional[int] = None
    enable_comparison: Optional[bool] = None
    show_controls: Optional[bool] = None
    show_delta_state: Optional[bool] = None
    show_motor_sensors: Optional[bool] = None
    show_all_comparisons: Optional[bool] = None
    sync_scales: Optional[bool] = None
    show_metrics: Optional[bool] = None
    state_name: Optional[str] = None
    selected_other_data: Optional[List[str]] = None
    comparison_start_index: Optional[int] = None
    default_car_model: Optional[str] = None
    default_car_parameters: Optional[str] = None
    theme: Optional[str] = None


class CsvLoadRequest(BaseModel):
    path: str


class SingleComparisonRequest(BaseModel):
    start_index: Optional[int] = None


class ReplaySnapshotRequest(BaseModel):
    index: int
    half_window: int = 150
    columns: Optional[List[str]] = None


class ReplayFrameRequest(BaseModel):
    index: int
    include_heavy: bool = True


def _handle(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    return service.settings.to_dict()


@app.put("/api/config")
def put_config(update: SettingsUpdate) -> Dict[str, Any]:
    try:
        return service.update_settings(update.model_dump(exclude_unset=True)).to_dict()
    except Exception as exc:
        raise _handle(exc)


@app.get("/api/options")
def get_options() -> Dict[str, Any]:
    return service.get_options()


@app.get("/api/session")
def get_session() -> Dict[str, Any]:
    return service.get_session_info()


@app.get("/api/csv/browse")
def browse_csv(path: str = "") -> Dict[str, Any]:
    try:
        return service.browse_csv(path)
    except Exception as exc:
        raise _handle(exc)


@app.post("/api/csv/load")
def load_csv(req: CsvLoadRequest) -> Dict[str, Any]:
    try:
        return service.load_csv_path(req.path)
    except Exception as exc:
        raise _handle(exc)


@app.post("/api/csv/upload")
async def upload_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        content = await file.read()
        return service.load_csv_upload(file.filename or "upload.csv", content)
    except Exception as exc:
        raise _handle(exc)


@app.post("/api/csv/reload")
def reload_csv() -> Dict[str, Any]:
    try:
        return service.reload_csv()
    except Exception as exc:
        raise _handle(exc)


@app.post("/api/comparison/single")
def run_single_comparison(req: SingleComparisonRequest) -> Dict[str, Any]:
    try:
        return service.run_single_comparison(req.start_index)
    except Exception as exc:
        raise _handle(exc)


@app.post("/api/comparison/full")
def run_full_comparison(force: bool = False) -> Dict[str, str]:
    try:
        job_id = service.start_full_comparison(force=force)
        return {"job_id": job_id}
    except Exception as exc:
        raise _handle(exc)


@app.get("/api/comparison/status/{job_id}")
def comparison_status(job_id: str) -> Dict[str, Any]:
    try:
        return service.get_job_status(job_id)
    except Exception as exc:
        raise _handle(exc)


@app.post("/api/comparison/clear")
def clear_comparisons() -> Dict[str, str]:
    service.clear_comparisons()
    return {"status": "ok"}


@app.get("/api/plot/data")
def plot_data() -> Dict[str, Any]:
    try:
        return service.get_plot_data()
    except Exception as exc:
        raise _handle(exc)


@app.get("/api/plot/bundle")
def plot_bundle() -> Dict[str, Any]:
    try:
        return service.get_plot_bundle()
    except Exception as exc:
        raise _handle(exc)


@app.get("/api/metrics")
def metrics() -> Optional[Dict[str, float]]:
    return service.get_metrics()


@app.get("/api/replay/ui-config")
def replay_ui_config() -> Dict[str, Any]:
    try:
        return service.get_replay_ui_config()
    except Exception as exc:
        raise _handle(exc)


@app.get("/api/replay/map")
def replay_map() -> Dict[str, Any]:
    try:
        return service.get_replay_map_info()
    except Exception as exc:
        raise _handle(exc)


@app.get("/api/replay/map-image")
def replay_map_image():
    try:
        return FileResponse(service.get_replay_map_image_path())
    except Exception as exc:
        raise _handle(exc)


@app.get("/api/replay/track")
def replay_track(columns: Optional[str] = None) -> Dict[str, Any]:
    try:
        cols = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        return service.get_replay_track(cols)
    except Exception as exc:
        raise _handle(exc)


@app.post("/api/replay/frame")
def replay_frame(req: ReplayFrameRequest) -> Dict[str, Any]:
    try:
        return service.get_replay_frame(req.index, include_heavy=req.include_heavy)
    except Exception as exc:
        raise _handle(exc)


@app.post("/api/replay/snapshot")
def replay_snapshot(req: ReplaySnapshotRequest) -> Dict[str, Any]:
    try:
        return service.get_replay_snapshot(req.index, req.half_window, req.columns)
    except Exception as exc:
        raise _handle(exc)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
