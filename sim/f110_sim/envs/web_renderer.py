import json
import os
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import yaml
from PIL import Image
from utilities.state_utilities import POSE_THETA_IDX, POSE_X_IDX, POSE_Y_IDX


HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>F1TENTH Web Renderer</title>
  <style>
    html, body { margin: 0; height: 100%; background: #091f56; color: #fff; font-family: sans-serif; }
    #root { display: grid; grid-template-rows: 72px 1fr; height: 100%; }
    #topbar { display: flex; align-items: center; gap: 16px; padding: 0 12px; background: #0d2f7b; }
    #status { font-size: 14px; opacity: 0.95; white-space: pre-line; }
    #controls { display: flex; align-items: center; gap: 14px; margin-left: auto; font-size: 13px; }
    #controls label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
    canvas { width: 100%; height: 100%; display: block; background: #092057; }
  </style>
</head>
<body>
  <div id="root">
    <div id="topbar">
      <strong>F1TENTH Web Renderer</strong>
      <span id="status">Connecting...</span>
      <div id="controls">
        <label><input id="toggle-map" type="checkbox" checked />Show Map</label>
        <label><input id="toggle-history" type="checkbox" />Show position history</label>
        <label><input id="toggle-car-info" type="checkbox" />Show car state info</label>
      </div>
    </div>
    <canvas id="view"></canvas>
  </div>
  <script>
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const toggleMapEl = document.getElementById("toggle-map");
    const toggleHistoryEl = document.getElementById("toggle-history");
    const toggleCarInfoEl = document.getElementById("toggle-car-info");
    let mapMeta = null;
    let mapImg = null;
    let latest = null;
    let latestSimulationTime = 0.0;
    let latestSimAtPoll = 0.0;
    let latestPollWallTimeS = 0.0;
    let prevPollSimTime = null;
    let prevPollWallTimeS = null;
    let simTimeRate = 1.0; // simulated seconds per wall second
    let observedFetchIntervalS = 0.1;
    let lastFetchCompleteWallS = null;
    let lastFrameId = 0;
    let staticOverlay = {};
    let frameBuffer = [];
    let playbackSimTime = null;
    let lastDrawWallTimeS = null;
    let cameraFollowEgo = true;
    let cameraCenter = null;
    const BUFFER_WINDOW_S = 2.5;
    const TARGET_BUFFER_DELAY_S = 1.8;
    const MIN_BUFFER_DELAY_S = 1.2;
    const HISTORY_POLL_MS = 100;
    const REQUEST_OVERLAP_FRAMES = 12;
    const MAX_INFLIGHT_FETCHES = 2;
    let inflightFetches = 0;
    let clientSessionId = null;
    let zoom = 60.0; // pixels per meter
    const ZOOM_MIN = 15.0;
    const ZOOM_MAX = 300.0;
    const ZOOM_FACTOR = 1.15;
    let isDragging = false;
    let dragLastX = 0.0;
    let dragLastY = 0.0;
    const viewerId = `viewer-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
    let showMap = true;
    let showPositionHistory = false;
    let showCarStateInfo = false;
    let browserPositionHistory = [];
    let browserHistoryLastFrameId = 0;
    const MAX_BROWSER_HISTORY_POINTS = 20000;

    function resizeCanvas() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function worldToScreen(x, y, cx, cy) {
      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;
      const sx = (x - cx) * zoom + 0.5 * w;
      const sy = h - ((y - cy) * zoom + 0.5 * h);
      return [sx, sy];
    }

    function clamp(v, lo, hi) {
      return Math.max(lo, Math.min(hi, v));
    }

    function rgb(color, fallback) {
      if (Array.isArray(color) && color.length >= 3) {
        return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
      }
      return fallback;
    }

    function lerp(a, b, t) {
      return a + (b - a) * t;
    }

    function angleLerp(a, b, t) {
      let d = b - a;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      return a + d * t;
    }

    function angleDelta(a, b) {
      let d = b - a;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      return d;
    }

    function interpolateState(a, b, t) {
      const overlaySource = t < 0.5 ? (a.web_overlay || {}) : (b.web_overlay || {});
      const poses = [];
      const n = Math.min(a.poses?.length || 0, b.poses?.length || 0);
      for (let i = 0; i < n; i++) {
        const pa = a.poses[i];
        const pb = b.poses[i];
        poses.push([
          lerp(pa[0], pb[0], t),
          lerp(pa[1], pb[1], t),
          angleLerp(pa[2], pb[2], t),
        ]);
      }
      return {
        ego_idx: b.ego_idx ?? a.ego_idx ?? 0,
        simulation_time: lerp(a.simulation_time || 0.0, b.simulation_time || 0.0, t),
        poses: poses,
        web_overlay: overlaySource,
      };
    }

    function extrapolateState(prev, last, targetTime) {
      if (!prev || !last) return last;
      const dt = Math.max(1e-6, last.simulation_time - prev.simulation_time);
      const extra = Math.max(0.0, targetTime - last.simulation_time);
      // Short extrapolation only; beyond this we prefer holding last sample.
      const extraClamped = Math.min(extra, 0.15);

      const poses = [];
      const n = Math.min(prev.poses?.length || 0, last.poses?.length || 0);
      for (let i = 0; i < n; i++) {
        const p0 = prev.poses[i];
        const p1 = last.poses[i];
        const vx = (p1[0] - p0[0]) / dt;
        const vy = (p1[1] - p0[1]) / dt;
        const w = angleDelta(p0[2], p1[2]) / dt;
        poses.push([
          p1[0] + vx * extraClamped,
          p1[1] + vy * extraClamped,
          p1[2] + w * extraClamped,
        ]);
      }
      return {
        ego_idx: last.ego_idx ?? 0,
        simulation_time: last.simulation_time + extraClamped,
        poses: poses,
        web_overlay: last.web_overlay || {},
      };
    }

    function stateForRender() {
      if (frameBuffer.length === 0) return null;
      if (frameBuffer.length === 1) return frameBuffer[0];

      const nowS = performance.now() / 1000.0;
      const oldestT = frameBuffer[0].simulation_time;
      const newestT = frameBuffer[frameBuffer.length - 1].simulation_time;

      if (playbackSimTime === null) {
        playbackSimTime = newestT - TARGET_BUFFER_DELAY_S;
        if (playbackSimTime < oldestT) playbackSimTime = oldestT;
        lastDrawWallTimeS = nowS;
      } else {
        const dtWall = Math.max(0.0, Math.min(0.1, nowS - (lastDrawWallTimeS ?? nowS)));
        lastDrawWallTimeS = nowS;

        // Keep horizon near target; gentle PLL-like correction.
        const currentHorizon = newestT - playbackSimTime;
        const horizonError = currentHorizon - TARGET_BUFFER_DELAY_S;
        const correction = Math.max(-0.08, Math.min(0.08, 0.04 * horizonError));
        const playbackRate = Math.max(0.2, simTimeRate * (1.0 + correction));

        if (currentHorizon > MIN_BUFFER_DELAY_S) {
          playbackSimTime += dtWall * playbackRate;
        }

        // Never run playback backward; keep inside available buffer.
        playbackSimTime = Math.max(playbackSimTime, oldestT);
        playbackSimTime = Math.min(playbackSimTime, newestT + 0.1);
      }
      const target = playbackSimTime;
      if (target <= frameBuffer[0].simulation_time) return frameBuffer[0];

      for (let i = 1; i < frameBuffer.length; i++) {
        const a = frameBuffer[i - 1];
        const b = frameBuffer[i];
        if (target <= b.simulation_time) {
          const dt = Math.max(1e-6, b.simulation_time - a.simulation_time);
          const t = Math.max(0.0, Math.min(1.0, (target - a.simulation_time) / dt));
          return interpolateState(a, b, t);
        }
      }
      const last = frameBuffer[frameBuffer.length - 1];
      const prev = frameBuffer.length > 1 ? frameBuffer[frameBuffer.length - 2] : null;
      return extrapolateState(prev, last, target);
    }

    function drawPoints(points, color, radius, cx, cy) {
      if (!Array.isArray(points)) return;
      ctx.fillStyle = color;
      for (const p of points) {
        if (!Array.isArray(p) || p.length < 2) continue;
        const [sx, sy] = worldToScreen(p[0], p[1], cx, cy);
        ctx.beginPath();
        ctx.arc(sx, sy, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function drawSinglePoint(pointList, color, radius, cx, cy) {
      if (!Array.isArray(pointList) || pointList.length === 0) return;
      const p = pointList[0];
      if (!Array.isArray(p) || p.length < 2) return;
      const [sx, sy] = worldToScreen(p[0], p[1], cx, cy);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawLine(points, color, lineWidth, cx, cy) {
      if (!Array.isArray(points) || points.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      points.forEach((p, idx) => {
        if (!Array.isArray(p) || p.length < 2) return;
        const [sx, sy] = worldToScreen(p[0], p[1], cx, cy);
        if (idx === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
    }

    function drawTrajectories(trajectories, color, radius, cx, cy) {
      if (!Array.isArray(trajectories)) return;
      ctx.fillStyle = color;
      for (const traj of trajectories) {
        if (!Array.isArray(traj)) continue;
        for (const p of traj) {
          if (!Array.isArray(p) || p.length < 2) continue;
          const [sx, sy] = worldToScreen(p[0], p[1], cx, cy);
          ctx.beginPath();
          ctx.arc(sx, sy, radius, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    async function loadMap() {
      try {
        const r = await fetch("/map");
        mapMeta = await r.json();
        mapImg = new Image();
        mapImg.src = "/map-image";
      } catch (e) {
        console.error(e);
      }
    }

    async function loadStaticOverlay() {
      try {
        const r = await fetch("/overlay-static");
        staticOverlay = await r.json();
      } catch (e) {
        // Keep last static overlay on errors.
      }
    }

    async function fetchStateHistoryTick() {
      if (inflightFetches >= MAX_INFLIGHT_FETCHES) {
        return;
      }
      inflightFetches += 1;
      try {
        const since = Math.max(0, lastFrameId - REQUEST_OVERLAP_FRAMES);
        const r = await fetch(`/state-history?since=${since}`);
        const payload = await r.json();
        const history = Array.isArray(payload.history) ? payload.history : [];
        const serverSessionId = payload.session_id || null;
        const incomingLatestFrameId = Number(payload.latest_frame_id || 0);
        if (serverSessionId !== null && clientSessionId !== null && serverSessionId !== clientSessionId) {
          // New backend instance detected: reset client-side state and continue seamlessly.
          frameBuffer = [];
          latest = null;
          latestSimulationTime = 0.0;
          latestSimAtPoll = 0.0;
          latestPollWallTimeS = 0.0;
          prevPollSimTime = null;
          prevPollWallTimeS = null;
          simTimeRate = 1.0;
          observedFetchIntervalS = 0.1;
          lastFetchCompleteWallS = null;
          lastFrameId = 0;
          playbackSimTime = null;
          lastDrawWallTimeS = null;
          browserPositionHistory = [];
          browserHistoryLastFrameId = 0;
          clientSessionId = serverSessionId;
          loadMap();
          loadStaticOverlay();
          statusEl.textContent = "Detected new experiment backend. Reconnected.";
        }
        if (clientSessionId === null && serverSessionId !== null) {
          clientSessionId = serverSessionId;
        }
        appendHistorySamples(history);
        // Ignore stale response snapshots that are older than what we already processed.
        if (incomingLatestFrameId + REQUEST_OVERLAP_FRAMES < lastFrameId) {
          return;
        }
        const newLatestSimulationTime = Number(payload.latest_simulation_time || 0.0);
        const nowWall = performance.now() / 1000.0;
        if (prevPollSimTime !== null && prevPollWallTimeS !== null) {
          const dSim = newLatestSimulationTime - prevPollSimTime;
          const dWall = Math.max(1e-6, nowWall - prevPollWallTimeS);
          const instRate = Math.max(0.1, Math.min(50.0, dSim / dWall));
          // Smooth estimate to avoid jitter in playback speed.
          simTimeRate = 0.8 * simTimeRate + 0.2 * instRate;
        }
        prevPollSimTime = newLatestSimulationTime;
        prevPollWallTimeS = nowWall;
        latestSimulationTime = newLatestSimulationTime;
        lastFrameId = Math.max(lastFrameId, incomingLatestFrameId);
        latestSimAtPoll = latestSimulationTime;
        latestPollWallTimeS = nowWall;

        const cutoff = latestSimulationTime - BUFFER_WINDOW_S;
        const mergedByFrameId = new Map();
        for (const s of [...frameBuffer, ...history]) {
          if (!s || !Number.isFinite(s.simulation_time) || !Number.isFinite(s.frame_id)) continue;
          if (s.simulation_time < cutoff) continue;
          const fid = Number(s.frame_id);
          const existing = mergedByFrameId.get(fid);
          if (!existing || s.simulation_time >= existing.simulation_time) {
            mergedByFrameId.set(fid, s);
          }
        }

        const merged = Array.from(mergedByFrameId.values()).sort((a, b) => {
          const ta = Number(a.simulation_time);
          const tb = Number(b.simulation_time);
          if (Math.abs(ta - tb) > 1e-9) return ta - tb;
          return Number(a.frame_id) - Number(b.frame_id);
        });

        // Ensure monotonic simulation_time in buffer (drop regressions).
        const monotonic = [];
        let lastT = -Infinity;
        for (const s of merged) {
          const t = Number(s.simulation_time);
          if (t + 1e-9 >= lastT) {
            monotonic.push(s);
            lastT = t;
          }
        }
        frameBuffer = monotonic;
        latest = frameBuffer.length > 0 ? frameBuffer[frameBuffer.length - 1] : null;

        const labels = latest?.web_overlay?.label_dict || {};
        const labelText = showCarStateInfo && Object.keys(labels).length > 0
          ? " | " + Object.entries(labels).map(([k, v]) => `${k}: ${v}`).join(" | ")
          : "";
        const camMode = cameraFollowEgo ? "follow-car" : "free";
        statusEl.textContent = `Sim time: ${(latestSimulationTime || 0).toFixed(2)} s | cars: ${(latest?.poses?.length || 0)} | cam: ${camMode} | zoom: ${zoom.toFixed(1)}${labelText}`;
      } catch (e) {
        statusEl.textContent = "Waiting for simulation state...";
      } finally {
        const now = performance.now() / 1000.0;
        if (lastFetchCompleteWallS !== null) {
          const interval = Math.max(0.01, now - lastFetchCompleteWallS);
          observedFetchIntervalS = 0.8 * observedFetchIntervalS + 0.2 * interval;
        }
        lastFetchCompleteWallS = now;
        inflightFetches = Math.max(0, inflightFetches - 1);
      }
    }

    function startHistoryPolling() {
      fetchStateHistoryTick();
      setInterval(fetchStateHistoryTick, HISTORY_POLL_MS);
    }

    function draw() {
      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#092057";
      ctx.fillRect(0, 0, w, h);

      const frame = stateForRender();
      if (frame && frame.poses && frame.poses.length > 0) {
        const ego = frame.poses[frame.ego_idx || 0];
        if (cameraFollowEgo || cameraCenter === null) {
          cameraCenter = [ego[0], ego[1]];
        }
        const cx = cameraCenter[0];
        const cy = cameraCenter[1];
        const overlay = Object.assign({}, staticOverlay || {}, frame.web_overlay || {});
        const colors = overlay.colors || {};

        if (showMap && mapMeta && mapImg && mapImg.complete) {
          const mapWm = mapMeta.width * mapMeta.resolution;
          const mapHm = mapMeta.height * mapMeta.resolution;
          const [sx, syTop] = worldToScreen(mapMeta.origin[0], mapMeta.origin[1] + mapHm, cx, cy);
          const drawW = mapWm * zoom;
          const drawH = mapHm * zoom;
          ctx.globalAlpha = 0.85;
          ctx.drawImage(mapImg, sx, syTop, drawW, drawH);
          ctx.globalAlpha = 1.0;
        }

        // Base/static overlays first
        const globalWaypoints = overlay.global_waypoints || overlay.waypoints_global || overlay.waypoints;
        drawPoints(globalWaypoints, rgb(colors.waypoints, "rgb(64, 190, 255)"), 2.0, cx, cy);
        drawPoints(overlay.waypoints_alternative, "rgb(170,170,170)", 1.5, cx, cy);
        drawPoints(overlay.next_waypoints_alternative, rgb(colors.next_waypoints_alternative, "rgb(127,0,127)"), 2.0, cx, cy);
        drawPoints(overlay.next_waypoints, rgb(colors.next_waypoints, "rgb(0,127,0)"), 2.2, cx, cy);
        drawPoints(overlay.lidar_border_points, rgb(colors.lidar, "rgb(255,0,255)"), 2.6, cx, cy);
        drawPoints(overlay.track_border_points, rgb(colors.track_border, "rgb(255,0,0)"), 1.2, cx, cy);
        drawPoints(overlay.obstacles, rgb(colors.obstacles, "rgb(255,0,0)"), 2.0, cx, cy);

        // Histories
        if (showPositionHistory) {
          drawTrajectories(browserPositionHistory, "rgba(70,160,255,0.55)", 1.8, cx, cy);
          drawPoints(overlay.past_car_states_alternative, rgb(colors.history_alt, "rgb(255,255,0)"), 2.0, cx, cy);
          drawPoints(overlay.past_car_states_gt, rgb(colors.history_gt, "rgb(0,128,255)"), 2.0, cx, cy);
          drawPoints(overlay.past_car_states_prior, rgb(colors.history_prior, "rgb(255,255,255)"), 2.0, cx, cy);
          drawPoints(overlay.past_car_states_prior_full, rgb(colors.history_prior_full, "rgb(0,255,255)"), 2.0, cx, cy);
        }

        // MPC overlays
        drawTrajectories(overlay.rollout_trajectory, rgb(colors.mppi, "rgb(250,25,30)"), 1.3, cx, cy);
        drawTrajectories(overlay.optimal_trajectory, rgb(colors.optimal, "rgb(255,165,0)"), 1.5, cx, cy);

        // Gap / target
        drawSinglePoint(overlay.largest_gap_middle_point, rgb(colors.gap, "rgb(0,255,0)"), 5.0, cx, cy);
        drawSinglePoint(overlay.target_point, rgb(colors.target, "rgb(255,204,0)"), 7.0, cx, cy);

        // Steering arrow and emergency slowdown lines
        if (overlay.steering_arrow && overlay.steering_arrow.start && overlay.steering_arrow.end) {
          drawLine([overlay.steering_arrow.start, overlay.steering_arrow.end], "rgb(0,204,0)", 2.0, cx, cy);
        }
        if (overlay.emergency_slowdown) {
          const s = overlay.emergency_slowdown;
          const factor = Math.max(0, Math.min(1, s.speed_reduction_factor ?? 1.0));
          const red = Math.floor((1 - factor) * 255);
          const green = Math.floor(factor * 255);
          const dynamicColor = `rgb(${red}, ${green}, 0)`;
          drawLine(s.left_line, dynamicColor, 2.0, cx, cy);
          drawLine(s.right_line, dynamicColor, 2.0, cx, cy);
          drawLine(s.stop_line, "rgb(255,0,0)", 2.0, cx, cy);
          if (Array.isArray(s.display_position) && s.display_position.length > 0 && factor < 1.0) {
            const p = s.display_position[0];
            const [sx, sy] = worldToScreen(p[0], p[1], cx, cy);
            ctx.fillStyle = dynamicColor;
            ctx.font = "12px sans-serif";
            ctx.fillText(factor.toFixed(2), sx, sy);
          }
        }

        frame.poses.forEach((p, i) => {
          const [sx, sy] = worldToScreen(p[0], p[1], cx, cy);
          const heading = p[2];
          const carLen = 0.58 * zoom;
          const carWid = 0.31 * zoom;
          ctx.save();
          ctx.translate(sx, sy);
          ctx.rotate(-heading);
          ctx.fillStyle = i === (frame.ego_idx || 0) ? "#ac61b9" : "#63345e";
          ctx.fillRect(-carLen / 2, -carWid / 2, carLen, carWid);
          ctx.restore();
        });
      }
      requestAnimationFrame(draw);
    }

    function startDrag(clientX, clientY) {
      isDragging = true;
      dragLastX = clientX;
      dragLastY = clientY;
      cameraFollowEgo = false;
      if (cameraCenter === null && latest?.poses?.length > 0) {
        const ego = latest.poses[latest.ego_idx || 0];
        cameraCenter = [ego[0], ego[1]];
      }
    }

    function updateDrag(clientX, clientY) {
      if (!isDragging || cameraCenter === null) return;
      const dxPx = clientX - dragLastX;
      const dyPx = clientY - dragLastY;
      dragLastX = clientX;
      dragLastY = clientY;
      cameraCenter = [
        cameraCenter[0] - dxPx / zoom,
        cameraCenter[1] + dyPx / zoom,
      ];
    }

    function stopDrag() {
      isDragging = false;
    }

    function ensureHistoryCapacity(carCount) {
      while (browserPositionHistory.length < carCount) {
        browserPositionHistory.push([]);
      }
    }

    function appendHistorySamples(samples) {
      if (!Array.isArray(samples)) return;
      for (const s of samples) {
        if (!s || !Array.isArray(s.poses)) continue;
        const fid = Number(s.frame_id || 0);
        if (fid <= browserHistoryLastFrameId) continue;
        ensureHistoryCapacity(s.poses.length);
        for (let i = 0; i < s.poses.length; i++) {
          const p = s.poses[i];
          if (!Array.isArray(p) || p.length < 2) continue;
          const track = browserPositionHistory[i];
          track.push([Number(p[0]), Number(p[1])]);
          if (track.length > MAX_BROWSER_HISTORY_POINTS) {
            track.splice(0, track.length - MAX_BROWSER_HISTORY_POINTS);
          }
        }
        browserHistoryLastFrameId = fid;
      }
    }

    async function sendViewerHeartbeat() {
      try {
        await fetch(`/viewer-heartbeat?viewer_id=${encodeURIComponent(viewerId)}`, { cache: "no-store" });
      } catch (e) {
        // Ignore transient heartbeat errors; polling loop will retry.
      }
    }

    window.addEventListener("resize", resizeCanvas);
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        cameraFollowEgo = !cameraFollowEgo;
      }
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? ZOOM_FACTOR : (1.0 / ZOOM_FACTOR);
      zoom = clamp(zoom * factor, ZOOM_MIN, ZOOM_MAX);
    }, { passive: false });
    canvas.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      startDrag(e.clientX, e.clientY);
    });
    window.addEventListener("mousemove", (e) => {
      updateDrag(e.clientX, e.clientY);
    });
    window.addEventListener("mouseup", () => {
      stopDrag();
    });
    canvas.addEventListener("mouseleave", () => {
      stopDrag();
    });
    toggleMapEl.addEventListener("change", () => {
      showMap = !!toggleMapEl.checked;
    });
    toggleHistoryEl.addEventListener("change", () => {
      showPositionHistory = !!toggleHistoryEl.checked;
    });
    toggleCarInfoEl.addEventListener("change", () => {
      showCarStateInfo = !!toggleCarInfoEl.checked;
    });
    sendViewerHeartbeat();
    setInterval(sendViewerHeartbeat, 2000);
    resizeCanvas();
    loadMap();
    loadStaticOverlay();
    startHistoryPolling();
    requestAnimationFrame(draw);
  </script>
</body>
</html>
"""

WEB_RENDERER_DIR = Path(__file__).resolve().parent / "WebRenderer"
WEB_RENDERER_HTML_PATH = WEB_RENDERER_DIR / "index.html"


class WebEnvRenderer:
    """
    HTTP-based renderer backend for browser visualization.

    Runtime data flow:
    - simulator calls `render(render_obs)` at control cadence
    - backend stores compact state/history in-memory
    - browser client pulls `/state-history` + `/overlay-static` and renders locally

    The browser client source-of-truth is `sim/f110_sim/envs/WebRenderer/index.html`.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "ego_idx": 0,
            "simulation_time": 0.0,
            "poses": [],
        }
        self._session_id = f"{int(time.time() * 1000)}-{os.getpid()}"
        self._state_history = []
        # Keep a larger server-side history so client buffer length
        # is independent from per-request chunk size.
        self._state_history_max = 600
        self._history_chunk_size = 50
        self._publish_rate_hz = 50.0
        self._last_published_sim_time: Optional[float] = None
        self._frame_id = 0
        self._static_overlay: Dict[str, Any] = {}
        self._static_overlay_keys = {
            "waypoints",
            "waypoints_alternative",
            "track_border_points",
            "colors",
        }
        self._static_max_points = {
            "waypoints": 4000,
            "waypoints_alternative": 4000,
            "track_border_points": 6000,
        }
        self._dynamic_max_points = {
            "lidar_border_points": 320,
            "next_waypoints": 80,
            "next_waypoints_alternative": 80,
            "past_car_states_alternative": 180,
            "past_car_states_gt": 180,
            "past_car_states_prior": 180,
            "past_car_states_prior_full": 220,
            "obstacles": 120,
        }
        self._max_rollout_trajectories = 8
        self._max_rollout_points_per_trajectory = 24
        self._max_optimal_trajectories = 2
        self._max_optimal_points_per_trajectory = 64
        self._float_precision_digits = 3
        self._map_meta: Dict[str, Any] = {
            "resolution": 1.0,
            "origin": [0.0, 0.0],
            "width": 0,
            "height": 0,
        }
        self._map_image_path: Optional[str] = None
        self._viewer_last_seen: Dict[str, float] = {}
        self._viewer_timeout_s = 6.0
        self._auto_open_cooldown_s = 12.0
        self._last_auto_open_attempt_s = 0.0
        self._auto_open_startup_grace_s = 5.0
        self._created_at_s = time.time()

        renderer = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload: Dict[str, Any], status: int = 200):
                encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)

                if path == "/" or path == "/index.html":
                    body = renderer._load_html_page().encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path == "/state":
                    with renderer._lock:
                        payload = dict(renderer._state)
                    self._send_json(payload)
                    return

                if path == "/state-history":
                    since_raw = query.get("since", ["0"])[0]
                    try:
                        since_id = int(since_raw)
                    except ValueError:
                        since_id = 0
                    with renderer._lock:
                        history = [s for s in renderer._state_history if int(s.get("frame_id", 0)) > since_id]
                        if len(history) > renderer._history_chunk_size:
                            history = history[-renderer._history_chunk_size :]
                        payload = {
                            "history": history,
                            "latest_simulation_time": float(renderer._state.get("simulation_time", 0.0)),
                            "latest_frame_id": int(renderer._state.get("frame_id", 0)),
                            "session_id": renderer._session_id,
                        }
                    self._send_json(payload)
                    return

                if path == "/overlay-static":
                    with renderer._lock:
                        payload = dict(renderer._static_overlay)
                        payload["session_id"] = renderer._session_id
                    self._send_json(payload)
                    return

                if path == "/map":
                    with renderer._lock:
                        payload = dict(renderer._map_meta)
                    self._send_json(payload)
                    return

                if path == "/map-image":
                    with renderer._lock:
                        path = renderer._map_image_path
                    if path is None or (not os.path.exists(path)):
                        self.send_response(HTTPStatus.NOT_FOUND)
                        self.end_headers()
                        return
                    with open(path, "rb") as f:
                        body = f.read()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path == "/viewer-heartbeat":
                    viewer_id = query.get("viewer_id", [""])[0].strip()
                    now = time.time()
                    with renderer._lock:
                        if viewer_id:
                            renderer._viewer_last_seen[viewer_id] = now
                        renderer._prune_inactive_viewers_locked(now)
                        payload = {
                            "ok": True,
                            "active_viewers": len(renderer._viewer_last_seen),
                            "session_id": renderer._session_id,
                        }
                    self._send_json(payload)
                    return

                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()

            def log_message(self, format: str, *args):
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        print(f"Web renderer listening on http://{self.host}:{self.port}")
        # Give already-open tabs a chance to reconnect and heartbeat first.
        # Without this grace period, restarting experiments can spuriously open
        # a duplicate tab before the existing tab pings the new backend.
        self._maybe_autolaunch_browser()

    def _prune_inactive_viewers_locked(self, now_s: Optional[float] = None):
        now_s = time.time() if now_s is None else now_s
        stale_ids = [
            vid
            for vid, last_seen in self._viewer_last_seen.items()
            if (now_s - float(last_seen)) > self._viewer_timeout_s
        ]
        for vid in stale_ids:
            self._viewer_last_seen.pop(vid, None)

    def _has_active_viewer(self, now_s: Optional[float] = None) -> bool:
        with self._lock:
            self._prune_inactive_viewers_locked(now_s)
            return len(self._viewer_last_seen) > 0

    def _maybe_autolaunch_browser(self):
        now_s = time.time()
        if (now_s - self._created_at_s) < self._auto_open_startup_grace_s:
            return
        if self._has_active_viewer(now_s):
            return
        with self._lock:
            if (now_s - self._last_auto_open_attempt_s) < self._auto_open_cooldown_s:
                return
            self._last_auto_open_attempt_s = now_s
        try:
            webbrowser.open(f"http://{self.host}:{self.port}", new=0, autoraise=False)
        except Exception:
            # Auto-open is best-effort and should never break simulation.
            pass

    def _load_html_page(self) -> str:
        # Prefer external client file so frontend iteration does not require
        # touching Python code. Keep string fallback for resilience.
        if WEB_RENDERER_HTML_PATH.exists():
            return WEB_RENDERER_HTML_PATH.read_text(encoding="utf-8")
        return HTML_PAGE

    def update_map(self, map_path: str, map_ext: str):
        yaml_path = map_path + ".yaml"
        img_path = map_path + map_ext
        with open(yaml_path, "r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
        img = np.array(Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM))
        with self._lock:
            self._map_meta = {
                "resolution": float(metadata["resolution"]),
                "origin": [float(metadata["origin"][0]), float(metadata["origin"][1])],
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
            }
            self._map_image_path = img_path

    @staticmethod
    def _downsample_points(points: Any, max_points: int):
        if not isinstance(points, list):
            return points
        n = len(points)
        if n <= max_points or max_points <= 0:
            return points
        stride = max(1, int(np.ceil(n / max_points)))
        return points[::stride]

    @staticmethod
    def _round_floats(obj: Any, digits: int):
        if isinstance(obj, float):
            return round(obj, digits)
        if isinstance(obj, list):
            return [WebEnvRenderer._round_floats(v, digits) for v in obj]
        if isinstance(obj, dict):
            return {k: WebEnvRenderer._round_floats(v, digits) for k, v in obj.items()}
        return obj

    def _downsample_trajectories(self, trajectories: Any, max_traj: int, max_points: int):
        if not isinstance(trajectories, list):
            return trajectories
        if len(trajectories) > max_traj:
            step = max(1, int(np.ceil(len(trajectories) / max_traj)))
            trajectories = trajectories[::step]
        trimmed = []
        for traj in trajectories:
            if isinstance(traj, list):
                trimmed.append(self._downsample_points(traj, max_points))
            else:
                trimmed.append(traj)
        return trimmed

    def _compress_dynamic_overlay(self, overlay: Dict[str, Any]) -> Dict[str, Any]:
        for key, max_pts in self._dynamic_max_points.items():
            if key in overlay:
                overlay[key] = self._downsample_points(overlay[key], max_pts)

        if "rollout_trajectory" in overlay:
            overlay["rollout_trajectory"] = self._downsample_trajectories(
                overlay["rollout_trajectory"],
                self._max_rollout_trajectories,
                self._max_rollout_points_per_trajectory,
            )
        if "optimal_trajectory" in overlay:
            overlay["optimal_trajectory"] = self._downsample_trajectories(
                overlay["optimal_trajectory"],
                self._max_optimal_trajectories,
                self._max_optimal_points_per_trajectory,
            )

        # Label text can become large; keep first 24 keys.
        labels = overlay.get("label_dict")
        if isinstance(labels, dict) and len(labels) > 24:
            keys = list(labels.keys())[:24]
            overlay["label_dict"] = {k: labels[k] for k in keys}

        return self._round_floats(overlay, self._float_precision_digits)

    def render(self, render_obs: Dict[str, Any]):
        car_states = render_obs.get("car_states")
        poses = []
        if car_states is not None:
            for state in car_states:
                poses.append(
                    [
                        float(state[POSE_X_IDX]),
                        float(state[POSE_Y_IDX]),
                        float(state[POSE_THETA_IDX]),
                    ]
                )
        else:
            poses_x = render_obs.get("poses_x", [])
            poses_y = render_obs.get("poses_y", [])
            poses_theta = render_obs.get("poses_theta", [])
            for x, y, theta in zip(poses_x, poses_y, poses_theta):
                poses.append([float(x), float(y), float(theta)])

        overlay = dict(render_obs.get("web_overlay", {}) or {})
        static_overlay = {}
        for key in list(overlay.keys()):
            if key in self._static_overlay_keys:
                static_overlay[key] = overlay.pop(key)
        for key, max_pts in self._static_max_points.items():
            if key in static_overlay:
                static_overlay[key] = self._downsample_points(static_overlay[key], max_pts)
        overlay = self._compress_dynamic_overlay(overlay)

        simulation_time = float(render_obs.get("simulation_time", 0.0))
        self._maybe_autolaunch_browser()

        with self._lock:
            if static_overlay:
                self._static_overlay = static_overlay

            if self._last_published_sim_time is not None:
                min_dt = 1.0 / self._publish_rate_hz
                if (simulation_time - self._last_published_sim_time) < min_dt:
                    return
            self._last_published_sim_time = simulation_time
            self._frame_id += 1
            self._state = {
                "frame_id": self._frame_id,
                "ego_idx": int(render_obs.get("ego_idx", 0)),
                "simulation_time": round(simulation_time, self._float_precision_digits),
                "poses": self._round_floats(poses, self._float_precision_digits),
                "web_overlay": overlay,
            }
            self._state_history.append(self._state)
            if len(self._state_history) > self._state_history_max:
                self._state_history = self._state_history[-self._state_history_max :]

    def close(self):
        self._server.shutdown()
        self._server.server_close()
