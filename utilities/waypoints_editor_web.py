#!/usr/bin/env python3
"""
Browser-based waypoint editor for F1TENTH maps.

Launch from the repository root:
    python utilities/waypoints_editor_web.py
    python utilities/waypoints_editor_web.py --map RCA2 --port 8766

Opens Settings.MAP_NAME by default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import splev, splprep

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utilities.Settings import Settings

MAPS_ROOT = REPO_ROOT / "utilities" / "maps"
DEFAULT_SCALE = 20.0

HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Waypoints Editor</title>
  <style>
    html, body { margin: 0; height: 100%; background: #0b1a33; color: #eef3ff; font-family: sans-serif; }
    #root { display: grid; grid-template-rows: auto 1fr; height: 100%; }
    #toolbar {
      display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
      padding: 10px 12px; background: #132a55; border-bottom: 1px solid #2a4f8f;
    }
    #toolbar label { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }
    #toolbar select, #toolbar input[type="text"] {
      background: #0b1a33; color: #eef3ff; border: 1px solid #3d6cb5; border-radius: 4px; padding: 4px 6px;
    }
    #toolbar button {
      background: #2f6fed; color: #fff; border: none; border-radius: 6px;
      padding: 7px 12px; font-size: 13px; cursor: pointer;
    }
    #toolbar button.secondary { background: #3a4f73; }
    #toolbar button.warn { background: #b85c1f; }
    #status { flex: 1 1 220px; font-size: 13px; opacity: 0.92; min-width: 180px; }
    #canvas-wrap { position: relative; min-height: 0; }
    canvas {
      width: 100%; height: 100%; display: block; background: #091f3f;
      touch-action: none; cursor: crosshair;
    }
  </style>
</head>
<body>
  <div id="root">
    <div id="toolbar">
      <label>Map
        <select id="map-select"></select>
      </label>
      <label>Waypoints
        <select id="file-select"></select>
      </label>
      <button id="btn-load" type="button">Load</button>
      <button id="btn-save" type="button">Save</button>
      <button id="btn-save-as" type="button" class="secondary">Save as…</button>
      <button id="btn-undo" type="button" class="secondary">Undo</button>
      <button id="btn-recalc" type="button" class="warn">Recalculate race line</button>
      <label>Gaussian scale
        <input id="scale-slider" type="range" min="1" max="50" step="0.1" value="20" />
        <span id="scale-value">20.0</span>
      </label>
      <span id="status">Loading…</span>
    </div>
    <div id="canvas-wrap"><canvas id="view"></canvas></div>
  </div>
  <script>
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const mapSelectEl = document.getElementById("map-select");
    const fileSelectEl = document.getElementById("file-select");
    const scaleSliderEl = document.getElementById("scale-slider");
    const scaleValueEl = document.getElementById("scale-value");

    let mapMeta = null;
    let mapImg = null;
    let initialX = [];
    let initialY = [];
    let x = [];
    let y = [];
    let history = [];
    let scale = 20.0;
    let zoom = 55.0;
    let cameraCenter = [0, 0];
    let dragging = false;
    let dragIndex = null;
    let panning = false;
    let panLast = [0, 0];
    let currentMap = "";
    let currentFile = "";

    const ZOOM_MIN = 10;
    const ZOOM_MAX = 400;
    const ZOOM_FACTOR = 1.12;
    const PICK_RADIUS_M = 0.35;

    function setStatus(msg) { statusEl.textContent = msg; }

    function resizeCanvas() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function worldToScreen(wx, wy) {
      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;
      const sx = (wx - cameraCenter[0]) * zoom + 0.5 * w;
      const sy = h - ((wy - cameraCenter[1]) * zoom + 0.5 * h);
      return [sx, sy];
    }

    function screenToWorld(sx, sy) {
      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;
      const wx = cameraCenter[0] + (sx - 0.5 * w) / zoom;
      const wy = cameraCenter[1] + (0.5 * h - sy) / zoom;
      return [wx, wy];
    }

    function drawPolyline(points, color, width, dashed=false) {
      if (!points || points.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dashed ? [6, 5] : []);
      ctx.beginPath();
      points.forEach((p, i) => {
        const [sx, sy] = worldToScreen(p[0], p[1]);
        if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    function drawPoints(points, color, radiusPx) {
      ctx.fillStyle = color;
      for (const p of points) {
        const [sx, sy] = worldToScreen(p[0], p[1]);
        ctx.beginPath();
        ctx.arc(sx, sy, radiusPx, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function drawMapBackground() {
      if (!mapMeta || !mapImg || !mapImg.complete) return;
      const mapWm = mapMeta.width * mapMeta.resolution;
      const mapHm = mapMeta.height * mapMeta.resolution;
      const [sx, syTop] = worldToScreen(mapMeta.origin[0], mapMeta.origin[1] + mapHm);
      const drawW = mapWm * zoom;
      const drawH = mapHm * zoom;
      ctx.globalAlpha = 0.88;
      ctx.drawImage(mapImg, sx, syTop, drawW, drawH);
      ctx.globalAlpha = 1.0;
    }

    function draw() {
      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#091f3f";
      ctx.fillRect(0, 0, w, h);
      drawMapBackground();
      const initialPts = initialX.map((xi, i) => [xi, initialY[i]]);
      const currentPts = x.map((xi, i) => [xi, y[i]]);
      drawPolyline(initialPts, "rgba(80,140,255,0.75)", 1.5, true);
      drawPolyline(currentPts, "rgba(70,220,120,0.95)", 2.0, false);
      drawPoints(currentPts, "#ff4d4d", 5);
    }

    function pushHistory() {
      if (history.length > 0) {
        const last = history[history.length - 1];
        if (last.x.length === x.length && last.x.every((v, i) => v === x[i]) &&
            last.y.every((v, i) => v === y[i])) return;
      }
      history.push({ x: x.slice(), y: y.slice() });
      if (history.length > 20) history.shift();
    }

    function undo() {
      if (history.length <= 1) {
        setStatus("Nothing to undo.");
        return;
      }
      history.pop();
      const prev = history[history.length - 1];
      x = prev.x.slice();
      y = prev.y.slice();
      draw();
      setStatus("Undid last change.");
    }

    function gaussianWeight(i, center, n) {
      const d = Math.min(Math.abs(i - center), Math.abs(i - center + n), Math.abs(i - center - n));
      return Math.exp(-0.5 * Math.pow(d / scale, 2));
    }

    function applyWeightedMove(dx, dy) {
      const n = x.length;
      for (let i = 0; i < n; i++) {
        const w = gaussianWeight(i, dragIndex, n);
        x[i] += dx * w;
        y[i] += dy * w;
      }
    }

    function nearestWaypointIndex(wx, wy) {
      let best = -1;
      let bestD = PICK_RADIUS_M;
      for (let i = 0; i < x.length; i++) {
        const d = Math.hypot(wx - x[i], wy - y[i]);
        if (d < bestD) { bestD = d; best = i; }
      }
      return best;
    }

    function fitCameraToWaypoints() {
      if (!x.length) return;
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (let i = 0; i < x.length; i++) {
        minX = Math.min(minX, x[i]); maxX = Math.max(maxX, x[i]);
        minY = Math.min(minY, y[i]); maxY = Math.max(maxY, y[i]);
      }
      cameraCenter = [0.5 * (minX + maxX), 0.5 * (minY + maxY)];
      const span = Math.max(maxX - minX, maxY - minY, 1.0);
      const rect = canvas.getBoundingClientRect();
      zoom = clamp(0.75 * Math.min(rect.width, rect.height) / span, ZOOM_MIN, ZOOM_MAX);
    }

    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

    async function fetchJson(url, options) {
      const r = await fetch(url, options);
      const payload = await r.json();
      if (!r.ok) throw new Error(payload.error || `HTTP ${r.status}`);
      return payload;
    }

    async function loadConfig() {
      const cfg = await fetchJson("/api/config");
      if (cfg.default_map) currentMap = cfg.default_map;
    }

    async function loadMapList() {
      const maps = (await fetchJson("/api/maps")).maps;
      mapSelectEl.innerHTML = "";
      for (const name of maps) {
        const opt = document.createElement("option");
        opt.value = name; opt.textContent = name;
        mapSelectEl.appendChild(opt);
      }
      if (maps.length) {
        currentMap = maps.includes(currentMap) ? currentMap : maps[0];
        mapSelectEl.value = currentMap;
        await refreshWaypointFiles();
      }
    }

    async function refreshWaypointFiles() {
      currentMap = mapSelectEl.value;
      const files = (await fetchJson(`/api/waypoint-files?map=${encodeURIComponent(currentMap)}`)).files;
      fileSelectEl.innerHTML = "";
      for (const file of files) {
        const opt = document.createElement("option");
        opt.value = file; opt.textContent = file;
        fileSelectEl.appendChild(opt);
      }
      if (files.length) {
        const preferred = `${currentMap}_wp.csv`;
        currentFile = files.includes(currentFile) ? currentFile :
          (files.includes(preferred) ? preferred : files[0]);
        fileSelectEl.value = currentFile;
      }
    }

    async function loadSession() {
      currentMap = mapSelectEl.value;
      currentFile = fileSelectEl.value;
      const payload = await fetchJson(
        `/api/session?map=${encodeURIComponent(currentMap)}&file=${encodeURIComponent(currentFile)}`
      );
      mapMeta = payload.map;
      initialX = payload.initial_x;
      initialY = payload.initial_y;
      x = payload.x.slice();
      y = payload.y.slice();
      scale = payload.scale || 20.0;
      scaleSliderEl.value = String(scale);
      scaleValueEl.textContent = scale.toFixed(1);
      history = [{ x: x.slice(), y: y.slice() }];
      mapImg = new Image();
      mapImg.onload = () => { fitCameraToWaypoints(); draw(); };
      mapImg.src = `/api/map-image?map=${encodeURIComponent(currentMap)}&t=${Date.now()}`;
      fitCameraToWaypoints();
      draw();
      setStatus(`Loaded ${currentMap}/${currentFile} (${x.length} waypoints). Drag points to edit.`);
    }

    async function saveWaypoints(filename) {
      const body = {
        map: currentMap,
        file: filename || currentFile,
        x, y,
      };
      const payload = await fetchJson("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (payload.file) {
        currentFile = payload.file;
        await refreshWaypointFiles();
        fileSelectEl.value = currentFile;
      }
      setStatus(payload.message || "Saved.");
    }

    async function recalculateRaceLine() {
      const payload = await fetchJson("/api/recalculate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ map: currentMap, file: currentFile, x, y }),
      });
      x = payload.x;
      y = payload.y;
      pushHistory();
      draw();
      setStatus(payload.message || "Race line recalculated.");
    }

    mapSelectEl.addEventListener("change", refreshWaypointFiles);
    document.getElementById("btn-load").addEventListener("click", loadSession);
    document.getElementById("btn-save").addEventListener("click", () => saveWaypoints(null));
    document.getElementById("btn-save-as").addEventListener("click", async () => {
      const name = prompt("Save as filename (inside map folder):", currentFile);
      if (!name) return;
      await saveWaypoints(name.trim());
    });
    document.getElementById("btn-undo").addEventListener("click", undo);
    document.getElementById("btn-recalc").addEventListener("click", recalculateRaceLine);
    scaleSliderEl.addEventListener("input", () => {
      scale = Number(scaleSliderEl.value);
      scaleValueEl.textContent = scale.toFixed(1);
    });

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [wx0, wy0] = screenToWorld(sx, sy);
      const factor = e.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
      zoom = clamp(zoom * factor, ZOOM_MIN, ZOOM_MAX);
      const [wx1, wy1] = screenToWorld(sx, sy);
      cameraCenter = [cameraCenter[0] + wx0 - wx1, cameraCenter[1] + wy0 - wy1];
      draw();
    }, { passive: false });

    canvas.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [wx, wy] = screenToWorld(sx, sy);
      const idx = nearestWaypointIndex(wx, wy);
      if (idx >= 0) {
        dragging = true;
        dragIndex = idx;
      } else {
        panning = true;
        panLast = [e.clientX, e.clientY];
      }
    });
    window.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      if (dragging && dragIndex !== null) {
        const [wx, wy] = screenToWorld(sx, sy);
        const dx = wx - x[dragIndex];
        const dy = wy - y[dragIndex];
        applyWeightedMove(dx, dy);
        draw();
      } else if (panning) {
        const dxPx = e.clientX - panLast[0];
        const dyPx = e.clientY - panLast[1];
        panLast = [e.clientX, e.clientY];
        cameraCenter = [cameraCenter[0] - dxPx / zoom, cameraCenter[1] + dyPx / zoom];
        draw();
      }
    });
    window.addEventListener("mouseup", () => {
      if (dragging) pushHistory();
      dragging = false;
      dragIndex = null;
      panning = false;
    });

    window.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        saveWaypoints(null);
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        undo();
      }
    });

    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
    loadConfig().then(loadMapList).then(loadSession).catch((err) => setStatus(String(err)));
  </script>
</body>
</html>
"""


def list_maps() -> List[str]:
    if not MAPS_ROOT.is_dir():
        return []
    maps = []
    for entry in sorted(MAPS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / f"{entry.name}.yaml").is_file() and (entry / f"{entry.name}.png").is_file():
            maps.append(entry.name)
    return maps


def list_waypoint_files(map_name: str) -> List[str]:
    map_dir = MAPS_ROOT / map_name
    if not map_dir.is_dir():
        return []
    files = sorted(p.name for p in map_dir.glob("*.csv") if "_wp" in p.name or p.name.endswith("_wp.csv"))
    if not files:
        files = sorted(p.name for p in map_dir.glob("*.csv"))
    return files


def load_map_meta(map_name: str) -> Dict[str, Any]:
    yaml_path = MAPS_ROOT / map_name / f"{map_name}.yaml"
    png_path = MAPS_ROOT / map_name / f"{map_name}.png"
    if not yaml_path.is_file() or not png_path.is_file():
        raise FileNotFoundError(f"Map assets missing for {map_name}")
    with open(yaml_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    from PIL import Image

    with Image.open(png_path) as img:
        width, height = img.size
    origin = cfg.get("origin", [0.0, 0.0, 0.0])
    return {
        "name": map_name,
        "resolution": float(cfg["resolution"]),
        "origin": [float(origin[0]), float(origin[1])],
        "width": int(width),
        "height": int(height),
    }


def waypoint_csv_path(map_name: str, filename: str) -> Path:
    map_dir = (MAPS_ROOT / map_name).resolve()
    target = (map_dir / filename).resolve()
    if map_dir not in target.parents and target != map_dir:
        raise ValueError("Invalid waypoint file path")
    if target.suffix.lower() != ".csv":
        raise ValueError("Waypoint file must be a .csv")
    return target


def read_waypoints_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()
    return df


def write_waypoints_csv(path: Path, df: pd.DataFrame) -> None:
    """Write waypoints in the same format as utilities/maps/RCA1/RCA1_wp.csv."""
    columns = list(df.columns)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in df.itertuples(index=False, name=None):
            handle.write(",".join(f"{float(value):.6f}" for value in row) + "\n")


def backup_existing_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}_backup_{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return str(backup_path.relative_to(REPO_ROOT))


def compute_path_geometry(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    x_ext = np.concatenate([x, [x[0]]])
    y_ext = np.concatenate([y, [y[0]]])
    dx = np.diff(x_ext)
    dy = np.diff(y_ext)
    ds = np.hypot(dx, dy)
    ds = np.maximum(ds, 1e-9)

    s = np.zeros(n, dtype=np.float64)
    s[1:] = np.cumsum(ds[:-1])

    heading_seg = np.arctan2(dy, dx)
    psi_csv = heading_seg - 0.5 * np.pi

    heading_ext = np.concatenate([heading_seg, [heading_seg[0]]])
    dheading = np.diff(heading_ext)
    dheading = (dheading + np.pi) % (2.0 * np.pi) - np.pi
    kappa = dheading / ds
    return s, psi_csv, kappa


def interpolate_column_by_s(old_s: np.ndarray, old_values: np.ndarray, new_s: np.ndarray) -> np.ndarray:
    total = old_s[-1] if len(old_s) else 0.0
    if total <= 1e-9:
        return np.full(len(new_s), old_values[0] if len(old_values) else 0.0)
    old_s_ext = np.concatenate([old_s, [total]])
    old_values_ext = np.concatenate([old_values, [old_values[0]]])
    return np.interp(new_s, old_s_ext, old_values_ext)


def build_waypoint_dataframe(
    x: np.ndarray,
    y: np.ndarray,
    template: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s, psi_csv, kappa = compute_path_geometry(x, y)

    if template is None:
        return pd.DataFrame(
            {
                "s_m": s,
                "x_m": x,
                "y_m": y,
                "psi_rad": psi_csv,
                "kappa_radpm": kappa,
            }
        )

    data = template.copy()
    old_s = data["s_m"].to_numpy(dtype=np.float64) if "s_m" in data.columns else np.linspace(0, 1, len(data))
    n = len(x)
    if len(data) != n:
        resized = {col: interpolate_column_by_s(old_s, data[col].to_numpy(dtype=np.float64), s) for col in data.columns}
        data = pd.DataFrame(resized)

    data["s_m"] = s
    data["x_m"] = x
    data["y_m"] = y
    data["psi_rad"] = psi_csv
    data["kappa_radpm"] = kappa
    return data


def _is_closed_loop(x: np.ndarray, y: np.ndarray) -> bool:
    gap = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    if len(x) < 2:
        return False
    chords = np.hypot(np.diff(x), np.diff(y))
    mean_chord = float(np.mean(chords)) if len(chords) else gap
    return gap < max(0.5, 0.25 * mean_chord)


def _resample_polyline_equal_spacing(
    x_dense: np.ndarray,
    y_dense: np.ndarray,
    n_out: int,
    closed: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    xd = np.asarray(x_dense, dtype=np.float64)
    yd = np.asarray(y_dense, dtype=np.float64)
    if closed:
        xd = np.concatenate([xd, [xd[0]]])
        yd = np.concatenate([yd, [yd[0]]])
    seg = np.hypot(np.diff(xd), np.diff(yd))
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total_length = float(s_cum[-1])
    if total_length <= 1e-9:
        raise ValueError("Degenerate race line")

    targets = np.linspace(0.0, total_length, n_out, endpoint=False)
    x_new = np.interp(targets, s_cum, xd)
    y_new = np.interp(targets, s_cum, yd)
    return x_new, y_new


def recalculate_raceline(x: np.ndarray, y: np.ndarray, num_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth the path with a local cubic B-spline through the waypoints, then
    resample at equal arc-length spacing.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_out = int(num_points or len(x))
    if n_out < 3:
        raise ValueError("Need at least 3 waypoints")
    if len(x) < 4:
        raise ValueError("Need at least 4 waypoints for spline fitting")

    closed = _is_closed_loop(x, y)
    xc = x.copy()
    yc = y.copy()
    if closed and np.hypot(xc[-1] - xc[0], yc[-1] - yc[0]) > 1e-6:
        xc = np.concatenate([xc, [xc[0]]])
        yc = np.concatenate([yc, [yc[0]]])

    spline_order = min(3, len(xc) - 1)
    tck, _ = splprep([xc, yc], s=0, per=closed, k=spline_order)

    dense_count = max(4000, n_out * 40)
    u_dense = np.linspace(0.0, 1.0, dense_count, endpoint=not closed)
    x_dense, y_dense = splev(u_dense, tck)
    return _resample_polyline_equal_spacing(x_dense, y_dense, n_out, closed=closed)


class WaypointsEditorServer:
    def __init__(self, host: str, port: int, default_map: str, default_scale: float):
        self.host = host
        self.port = port
        self.default_map = default_map
        self.default_scale = default_scale
        self._httpd: Optional[ThreadingHTTPServer] = None

    def start(self, open_browser: bool = True) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send_bytes(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
                body = json.dumps(payload).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8", status)

            def _read_json(self) -> Dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                return json.loads(raw.decode("utf-8"))

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)

                if path in ("/", "/index.html"):
                    self._send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                    return

                if path == "/api/config":
                    self._send_json(
                        {
                            "default_map": server.default_map,
                            "scale": server.default_scale,
                        }
                    )
                    return

                if path == "/api/maps":
                    self._send_json({"maps": list_maps()})
                    return

                if path == "/api/waypoint-files":
                    map_name = query.get("map", [server.default_map])[0]
                    self._send_json({"files": list_waypoint_files(map_name)})
                    return

                if path == "/api/session":
                    map_name = query.get("map", [server.default_map])[0]
                    filename = query.get("file", [f"{map_name}_wp.csv"])[0]
                    try:
                        csv_path = waypoint_csv_path(map_name, filename)
                        df = read_waypoints_csv(csv_path)
                        x = df["x_m"].to_numpy(dtype=np.float64)
                        y = df["y_m"].to_numpy(dtype=np.float64)
                        payload = {
                            "map": load_map_meta(map_name),
                            "file": filename,
                            "x": x.tolist(),
                            "y": y.tolist(),
                            "initial_x": x.tolist(),
                            "initial_y": y.tolist(),
                            "scale": server.default_scale,
                            "columns": list(df.columns),
                        }
                        self._send_json(payload)
                    except Exception as exc:
                        self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return

                if path == "/api/map-image":
                    map_name = query.get("map", [server.default_map])[0]
                    image_path = MAPS_ROOT / map_name / f"{map_name}.png"
                    if not image_path.is_file():
                        self._send_json({"error": "Map image not found"}, HTTPStatus.NOT_FOUND)
                        return
                    body = image_path.read_bytes()
                    self._send_bytes(body, "image/png")
                    return

                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    body = self._read_json()
                except json.JSONDecodeError:
                    self._send_json({"error": "Invalid JSON"}, HTTPStatus.BAD_REQUEST)
                    return

                if path == "/api/save":
                    try:
                        map_name = body["map"]
                        filename = body.get("file") or f"{map_name}_wp.csv"
                        x = np.asarray(body["x"], dtype=np.float64)
                        y = np.asarray(body["y"], dtype=np.float64)
                        csv_path = waypoint_csv_path(map_name, filename)
                        template = read_waypoints_csv(csv_path) if csv_path.is_file() else None
                        backup = backup_existing_file(csv_path)
                        df = build_waypoint_dataframe(x, y, template)
                        write_waypoints_csv(csv_path, df)
                        message = f"Saved {csv_path.relative_to(REPO_ROOT)} ({len(df)} waypoints)."
                        if backup:
                            message += f" Backup: {backup}."
                        self._send_json({"message": message, "file": filename, "backup": backup})
                    except Exception as exc:
                        self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return

                if path == "/api/recalculate":
                    try:
                        map_name = body["map"]
                        filename = body.get("file") or f"{map_name}_wp.csv"
                        x = np.asarray(body["x"], dtype=np.float64)
                        y = np.asarray(body["y"], dtype=np.float64)
                        x_new, y_new = recalculate_raceline(x, y, num_points=len(x))
                        csv_path = waypoint_csv_path(map_name, filename)
                        template = read_waypoints_csv(csv_path) if csv_path.is_file() else None
                        df = build_waypoint_dataframe(x_new, y_new, template)
                        spacing = float(np.mean(np.diff(df["s_m"].to_numpy()))) if len(df) > 1 else 0.0
                        self._send_json(
                            {
                                "x": x_new.tolist(),
                                "y": y_new.tolist(),
                                "spacing_m": spacing,
                                "message": (
                                    f"Recalculated {len(x_new)} waypoints with local spline "
                                    f"(~{spacing:.4f} m spacing)."
                                ),
                            }
                        )
                    except Exception as exc:
                        self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return

                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        url = f"http://{self.host}:{self.port}/"
        print(f"Waypoints editor running at {url}")
        if open_browser:
            webbrowser.open(url)
        self._httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Web-based waypoint editor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--map",
        default=Settings.MAP_NAME,
        help=f"Default map to open (default: Settings.MAP_NAME = {Settings.MAP_NAME})",
    )
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Initial Gaussian drag scale")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    server = WaypointsEditorServer(args.host, args.port, args.map, args.scale)
    server.start(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
