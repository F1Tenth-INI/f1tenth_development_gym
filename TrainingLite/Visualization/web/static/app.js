/**
 * State Comparison Visualizer — frontend
 */

const GRADIENT_COLORS = [
  [82, 0, 245],
  [255, 0, 191],
  [255, 0, 0],
];

const OTHER_DATA_COLORS = [
  "#008000", "#FF00FF", "#0000FF", "#FFA500", "#00CED1",
  "#9400D3", "#FFD700", "#FF1493", "#00FF00", "#8A2BE2",
];

const CTRL_STEERING_COLOR = "#e74c3c";
const CTRL_ACCEL_COLOR = "#3498db";

const CTRL_COLOR_BY_COLUMN = {
  angular_control_executed: CTRL_STEERING_COLOR,
  translational_control_executed: CTRL_ACCEL_COLOR,
  angular_control: CTRL_STEERING_COLOR,
  translational_control: CTRL_ACCEL_COLOR,
  angular_control_calculated: "#c0392b",
  translational_control_calculated: "#2980b9",
};

function isSteeringControlColumn(col) {
  return col.includes("angular");
}

function isAccelControlColumn(col) {
  return col.includes("translational");
}

function controlOverlayYAxis(baseY) {
  return overlayTraceYAxis(baseY);
}

let settings = {};
let session = null;
let browsePath = "AnalyseData";
let sliderTimer = null;
let heartbeatTimer = null;
let loadingCount = 0;
const PLOT_UIREVISION = "state-viz-v1";
const COMPARISON_CANCELLED = "ComparisonCancelled";
let comparisonRunId = 0;
let activeJobWait = null;

function syncLoadingUi() {
  const bar = document.getElementById("global-loading");
  const chart = document.getElementById("chart-container");
  if (loadingCount > 0) {
    bar.classList.remove("hidden");
    chart?.classList.add("is-loading");
  } else {
    bar.classList.add("hidden");
    chart?.classList.remove("is-loading");
  }
}

function resetLoading() {
  loadingCount = 0;
  syncLoadingUi();
}

function setLoading(active, message) {
  loadingCount += active ? 1 : -1;
  if (loadingCount < 0) {
    console.warn("Loading count underflow; resetting");
    loadingCount = 0;
  }
  const msgEl = document.getElementById("global-loading-msg");
  if (message && (active || loadingCount > 0)) msgEl.textContent = message;
  syncLoadingUi();
}

function updateLoadingMessage(message) {
  if (loadingCount > 0 && message) {
    document.getElementById("global-loading-msg").textContent = message;
  }
}

async function withLoading(message, fn) {
  setLoading(true, message);
  try {
    return await fn();
  } finally {
    setLoading(false);
    if (loadingCount === 0) syncLoadingUi();
  }
}

function isComparisonCancelled(err) {
  return err?.message === COMPARISON_CANCELLED;
}

function cancelActiveJobWait() {
  if (!activeJobWait) return;
  const state = activeJobWait;
  state.cancelled = true;
  if (state.timer) clearTimeout(state.timer);
  activeJobWait = null;
  state.reject(new Error(COMPARISON_CANCELLED));
}

function capturePlotLayoutState() {
  const el = document.getElementById("chart-container");
  if (!el?.layout) return null;
  const src = el.layout;
  const saved = {};
  Object.keys(src).forEach((key) => {
    if (!key.startsWith("xaxis") && !key.startsWith("yaxis")) return;
    const axis = src[key];
    if (!axis) return;
    saved[key] = {};
    if (Array.isArray(axis.range)) saved[key].range = axis.range.slice();
    if (axis.autorange === false) saved[key].autorange = false;
  });
  return Object.keys(saved).length ? saved : null;
}

function applyPlotLayoutState(layout, saved, identity) {
  if (!saved) {
    layout.uirevision = identity
      ? `${PLOT_UIREVISION}-${identity}`
      : `${PLOT_UIREVISION}-autoscale`;
    Object.keys(layout).forEach((key) => {
      if (!key.startsWith("xaxis") && !key.startsWith("yaxis")) return;
      layout[key] = { ...(layout[key] || {}), autorange: true };
    });
    return layout;
  }
  layout.uirevision = PLOT_UIREVISION;
  Object.entries(saved).forEach(([key, axisState]) => {
    layout[key] = { ...(layout[key] || {}), ...axisState };
  });
  return layout;
}

function getChartTheme() {
  const light = settings.theme === "light";
  return {
    paper: light ? "#ffffff" : "#1a1d27",
    plot: light ? "#f8f9fb" : "#0f1117",
    text: light ? "#1a1d2e" : "#e4e8f0",
    grid: light ? "#dce0e8" : "#2d3348",
    gtLine: light ? "#1a1d2e" : "#e4e8f0",
    legendBg: light ? "rgba(255,255,255,0.85)" : "rgba(0,0,0,0.3)",
  };
}

function applyTheme(theme) {
  const t = theme === "light" ? "light" : "dark";
  settings.theme = t;
  document.documentElement.setAttribute("data-theme", t);
  document.getElementById("theme-dark").classList.toggle("active", t === "dark");
  document.getElementById("theme-light").classList.toggle("active", t === "light");
}

function startBrowserHeartbeat() {
  const ping = () => {
    fetch("/api/browser/heartbeat", { method: "POST" }).catch(() => {});
  };
  ping();
  if (heartbeatTimer) clearInterval(heartbeatTimer);
  heartbeatTimer = setInterval(ping, 5000);
}

// ------------------------------------------------------------------ API helpers
async function api(method, path, body, trackLoading = false) {
  if (trackLoading) setLoading(true);
  try {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    if (res.status === 204) return null;
    return res.json();
  } finally {
    if (trackLoading) setLoading(false);
  }
}

// ------------------------------------------------------------------ gradient traces
function lerpColor(t) {
  const n = GRADIENT_COLORS.length - 1;
  const seg = Math.min(Math.floor(t * n), n - 1);
  const local = t * n - seg;
  const c0 = GRADIENT_COLORS[seg];
  const c1 = GRADIENT_COLORS[seg + 1];
  const r = Math.round(c0[0] + (c1[0] - c0[0]) * local);
  const g = Math.round(c0[1] + (c1[1] - c0[1]) * local);
  const b = Math.round(c0[2] + (c1[2] - c0[2]) * local);
  return `rgb(${r},${g},${b})`;
}

function gradientLineTraces(time, values, label, alphaBase = 0.6, showInLegend = true) {
  if (!time || time.length === 0) return [];
  const traces = [];
  const n = values.length;
  if (n === 1) {
    traces.push({
      x: [time[0]], y: [values[0]], mode: "markers",
      marker: { color: lerpColor(0), size: 4, opacity: alphaBase },
      name: label, showlegend: showInLegend && !!label,
    });
    return traces;
  }
  for (let i = 0; i < n - 1; i++) {
    const t = i / Math.max(1, n - 2);
    traces.push({
      x: [time[i], time[i + 1]], y: [values[i], values[i + 1]],
      mode: "lines+markers",
      line: { color: lerpColor(t), width: 2 },
      marker: { color: lerpColor(t), size: 3, opacity: alphaBase * (1 - 0.3 * t) },
      name: i === 0 ? label : undefined,
      showlegend: showInLegend && i === 0 && !!label,
      legendgroup: label || "pred",
    });
  }
  return traces;
}

// ------------------------------------------------------------------ plot rendering
function traceXAxisId(rowIndex) {
  return rowIndex === 0 ? "x" : `x${rowIndex + 1}`;
}

function traceYAxisId(rowIndex) {
  return rowIndex === 0 ? "y" : `y${rowIndex + 1}`;
}

function layoutXAxisKey(rowIndex) {
  return rowIndex === 0 ? "xaxis" : `xaxis${rowIndex + 1}`;
}

function layoutYAxisKey(rowIndex) {
  return rowIndex === 0 ? "yaxis" : `yaxis${rowIndex + 1}`;
}

function layoutYAxisKeyFromTrace(traceY) {
  return traceY === "y" ? "yaxis" : `yaxis${traceY.slice(1)}`;
}

function subplotAxisIds(rowIndex) {
  return {
    x: traceXAxisId(rowIndex),
    y: traceYAxisId(rowIndex),
    xLayout: layoutXAxisKey(rowIndex),
    yLayout: layoutYAxisKey(rowIndex),
  };
}

function overlayTraceYAxis(baseTraceY) {
  const n = baseTraceY === "y" ? 2 : parseInt(baseTraceY.slice(1), 10) + 1;
  return `y${n}`;
}

function computeRowDomains(rowCount, rowHeights, gap = 0.06) {
  const totalWeight = rowHeights.reduce((a, b) => a + b, 0);
  const usable = 1 - gap * Math.max(0, rowCount - 1);
  const domains = [];
  let cursor = 0;
  rowHeights.forEach((w) => {
    const h = (w / totalWeight) * usable;
    domains.push({ x: [0, 1], y: [1 - cursor - h, 1 - cursor] });
    cursor += h + gap;
  });
  return domains;
}

function configureStackedAxes(layout, rowDefs, chartTheme) {
  const weights = rowDefs.map((r) => r.weight);
  const domains = computeRowDomains(rowDefs.length, weights);
  return rowDefs.map((row, i) => {
    const ids = subplotAxisIds(i);
    const isLast = i === rowDefs.length - 1;
    layout[ids.xLayout] = {
      title: isLast ? "Time" : "",
      gridcolor: chartTheme.grid,
      domain: domains[i].x,
      anchor: ids.y,
      showticklabels: isLast,
      color: chartTheme.text,
      ...(i > 0 ? { matches: "x" } : {}),
    };
    layout[ids.yLayout] = {
      title: row.yTitle,
      gridcolor: chartTheme.grid,
      domain: domains[i].y,
      anchor: ids.x,
      color: chartTheme.text,
    };
    return ids;
  });
}

function buildSubplotLayout(showDelta, showControls) {
  const rowCount = 1 + (showDelta ? 1 : 0) + (showControls ? 1 : 0);
  const weights = rowCount === 1 ? [1] : rowCount === 2 ? [0.62, 0.38] : [0.5, 0.25, 0.25];
  return { rowCount, weights };
}

async function refreshPlot({
  preserveZoom = true,
  showLoading = false,
  loadingMessage = "Updating plot…",
} = {}) {
  const render = async () => {
    if (!session || session.row_count === 0) return;

    const savedLayout = preserveZoom ? capturePlotLayoutState() : null;

    let plotData;
    let metrics = null;
    try {
      [plotData, metrics] = await Promise.all([
        api("GET", "/api/plot/data", undefined, false),
        settings.show_metrics
          ? api("GET", "/api/metrics", undefined, false).catch(() => null)
          : Promise.resolve(null),
      ]);
    } catch (e) {
      console.warn("Plot data:", e.message);
      return;
    }

    const { traces, layout, structureKey } = buildPlotFromData(plotData);
    applyPlotLayoutState(layout, savedLayout, preserveZoom ? null : plotData.state_name);
    layout.uirevision = `${layout.uirevision || PLOT_UIREVISION}-${structureKey}`;

    const container = document.getElementById("chart-container");
    const chartHeight = Math.max(400, container.clientHeight || 500);

    if (container.data && container.dataset.structureKey !== structureKey) {
      Plotly.purge(container);
    }
    container.dataset.structureKey = structureKey;

    await Plotly.react(
      "chart-container",
      traces,
      { ...layout, height: chartHeight },
      { responsive: true, displayModeBar: true }
    );

    applyMetricsToDom(metrics);
  };

  if (showLoading) return withLoading(loadingMessage, render);
  return render();
}

function buildPlotFromData(plotData) {
  const showDelta = settings.show_delta_state;
  const showControls = settings.show_controls && Object.keys(plotData.controls || {}).length > 0;
  const multiPred = settings.show_all_comparisons && plotData.predictions.length > 1;
  const showPredLegend = !multiPred;

  const { rowCount, weights } = buildSubplotLayout(showDelta, showControls);
  const traces = [];
  const chartTheme = getChartTheme();

  const rowDefs = [{ weight: weights[0], yTitle: plotData.state_name }];
  if (showDelta) rowDefs.push({ weight: weights[rowDefs.length], yTitle: `Δ ${plotData.state_name}` });
  if (showControls) rowDefs.push({ weight: weights[rowDefs.length], yTitle: "" });

  const layout = {
    paper_bgcolor: chartTheme.paper,
    plot_bgcolor: chartTheme.plot,
    font: { color: chartTheme.text },
    margin: { t: 50, r: 20, b: 40, l: 60 },
    showlegend: true,
    legend: { bgcolor: chartTheme.legendBg, font: { color: chartTheme.text } },
  };

  const axisIds = configureStackedAxes(layout, rowDefs, chartTheme);
  const mainIds = axisIds[0];
  let deltaIds = showDelta ? axisIds[1] : null;
  let controlsIds = showControls ? axisIds[axisIds.length - 1] : null;

  // --- Main state row
  traces.push({
    x: plotData.time, y: plotData.ground_truth,
    mode: "lines", name: "Ground Truth",
    line: { color: chartTheme.gtLine, width: 2 },
    xaxis: mainIds.x, yaxis: mainIds.y,
  });

  if (settings.enable_comparison && plotData.predictions.length > 0) {
    plotData.predictions.forEach((pred) => {
      const predTraces = gradientLineTraces(
        pred.time, pred.values,
        showPredLegend ? "Model Prediction" : null,
        0.6, showPredLegend
      );
      predTraces.forEach((t) => { t.xaxis = mainIds.x; t.yaxis = mainIds.y; });
      traces.push(...predTraces);
    });
  }

  const otherCols = Object.keys(plotData.other_data || {});
  if (otherCols.length > 0) {
    const useTwinAxis = !settings.sync_scales && rowCount === 1;
    if (settings.sync_scales || rowCount > 1) {
      otherCols.forEach((col, i) => {
        traces.push({
          x: plotData.time, y: plotData.other_data[col],
          mode: "lines", name: col,
          line: { color: OTHER_DATA_COLORS[i % OTHER_DATA_COLORS.length], width: 1.5, dash: "dot" },
          xaxis: mainIds.x, yaxis: mainIds.y,
        });
      });
    } else if (useTwinAxis) {
      layout.yaxis2 = {
        title: "Other Data", overlaying: "y", side: "right",
        gridcolor: chartTheme.grid, showgrid: false,
        color: chartTheme.text,
      };
      otherCols.forEach((col, i) => {
        traces.push({
          x: plotData.time, y: plotData.other_data[col],
          mode: "lines", name: col, yaxis: "y2",
          line: { color: OTHER_DATA_COLORS[i % OTHER_DATA_COLORS.length], width: 1.5 },
          xaxis: mainIds.x,
        });
      });
    }
  }

  // --- Delta row
  if (showDelta && deltaIds) {
    if (plotData.delta && plotData.delta.ground_truth) {
      traces.push({
        x: plotData.delta.ground_truth.time,
        y: plotData.delta.ground_truth.values,
        mode: "lines", name: "Ground Truth Delta",
        line: { color: chartTheme.gtLine, width: 2 },
        xaxis: deltaIds.x, yaxis: deltaIds.y,
      });
    }
    if (settings.enable_comparison) {
      plotData.predictions.forEach((pred, idx) => {
        if (pred.delta) {
          const showLegend = showPredLegend && idx === 0;
          const deltaTraces = gradientLineTraces(
            pred.delta.time, pred.delta.values,
            showLegend ? "Model Prediction Delta" : null,
            0.6, showLegend
          );
          deltaTraces.forEach((t) => { t.xaxis = deltaIds.x; t.yaxis = deltaIds.y; });
          traces.push(...deltaTraces);
        }
      });
    }
  }

  // --- Controls row (always below main plot, dual y-axes for different scales)
  if (showControls && controlsIds) {
    const overlayTraceY = controlOverlayYAxis(controlsIds.y);
    const overlayLayoutY = layoutYAxisKeyFromTrace(overlayTraceY);
    let hasSteering = false;
    let hasAccel = false;

    const controlEntries = Object.entries(plotData.controls).sort(([a], [b]) => {
      if (isSteeringControlColumn(a) && !isSteeringControlColumn(b)) return -1;
      if (!isSteeringControlColumn(a) && isSteeringControlColumn(b)) return 1;
      return 0;
    });

    controlEntries.forEach(([col, vals]) => {
      const steering = isSteeringControlColumn(col);
      const accel = isAccelControlColumn(col);
      const useOverlay = accel && !steering;
      const color = CTRL_COLOR_BY_COLUMN[col] || (steering ? CTRL_STEERING_COLOR : CTRL_ACCEL_COLOR);

      if (steering) hasSteering = true;
      if (accel) hasAccel = true;

      traces.push({
        x: plotData.time,
        y: vals,
        mode: "lines",
        name: col,
        line: { color, width: 1.5 },
        xaxis: controlsIds.x,
        yaxis: useOverlay ? overlayTraceY : controlsIds.y,
      });
    });

    if (hasSteering) {
      layout[controlsIds.yLayout] = {
        ...layout[controlsIds.yLayout],
        title: { text: "Steering", font: { color: CTRL_STEERING_COLOR } },
        tickfont: { color: CTRL_STEERING_COLOR },
        linecolor: CTRL_STEERING_COLOR,
        gridcolor: chartTheme.grid,
      };
    } else if (hasAccel) {
      layout[controlsIds.yLayout] = {
        ...layout[controlsIds.yLayout],
        title: { text: "Acceleration", font: { color: CTRL_ACCEL_COLOR } },
        tickfont: { color: CTRL_ACCEL_COLOR },
        linecolor: CTRL_ACCEL_COLOR,
        gridcolor: chartTheme.grid,
      };
    }

    if (hasSteering && hasAccel) {
      layout[overlayLayoutY] = {
        title: { text: "Acceleration", font: { color: CTRL_ACCEL_COLOR } },
        tickfont: { color: CTRL_ACCEL_COLOR },
        linecolor: CTRL_ACCEL_COLOR,
        overlaying: controlsIds.y,
        anchor: controlsIds.x,
        side: "right",
        showgrid: false,
      };
    }
  }

  const structureKey = `rows-${rowCount}-ctrl-${showControls}-delta-${showDelta}`;

  layout.title = { text: plotData.state_name, font: { size: 14 } };
  if (rowCount > 1) {
    layout.title = { text: plotData.state_name, font: { size: 14 }, y: 0.98 };
    if (showControls) layout.margin = { ...layout.margin, r: 70 };
  }

  return { traces, layout, structureKey };
}

function applyMetricsToDom(m) {
  const ids = ["metric-mean", "metric-max", "metric-std", "metric-rmse"];
  if (!settings.show_metrics || !m) {
    ids.forEach((id) => { document.getElementById(id).textContent = "N/A"; });
    return;
  }
  document.getElementById("metric-mean").textContent = m.mean_error.toFixed(4);
  document.getElementById("metric-max").textContent = m.max_error.toFixed(4);
  document.getElementById("metric-std").textContent = m.error_std.toFixed(4);
  document.getElementById("metric-rmse").textContent = m.rmse.toFixed(4);
}

async function refreshMetrics() {
  applyMetricsToDom(await api("GET", "/api/metrics", undefined, false).catch(() => null));
}

async function waitForComparisonJob(jobId, { timeoutMs = 600000 } = {}) {
  cancelActiveJobWait();

  const statusEl = document.getElementById("job-status");
  statusEl.textContent = "Starting comparison…";
  statusEl.className = "job-status running";
  const deadline = Date.now() + timeoutMs;

  return new Promise((resolve, reject) => {
    const state = { cancelled: false, timer: null, resolve, reject };

    const finish = (fn, value) => {
      if (state.cancelled) return;
      state.cancelled = true;
      if (state.timer) clearTimeout(state.timer);
      if (activeJobWait === state) activeJobWait = null;
      fn(value);
    };

    activeJobWait = state;

    const poll = async () => {
      if (state.cancelled) return;

      if (Date.now() > deadline) {
        statusEl.className = "job-status error";
        statusEl.textContent = "Comparison timed out";
        finish(reject, new Error("Comparison timed out"));
        return;
      }

      try {
        const job = await api("GET", `/api/comparison/status/${jobId}`, undefined, false);
        if (state.cancelled) return;

        statusEl.textContent = job.message || job.status;
        if (job.status === "running") {
          statusEl.className = "job-status running";
          state.timer = setTimeout(poll, 400);
          return;
        }
        if (job.status === "completed") {
          statusEl.className = "job-status done";
          setTimeout(() => { statusEl.textContent = ""; }, 3000);
          finish(resolve, job);
          return;
        }
        if (job.status === "failed") {
          statusEl.className = "job-status error";
          statusEl.textContent = job.error || "Comparison failed";
          finish(reject, new Error(job.error || "Comparison failed"));
          return;
        }

        // Unknown status — keep polling but don't hang forever silently
        state.timer = setTimeout(poll, 400);
      } catch (e) {
        if (state.cancelled || isComparisonCancelled(e)) return;
        statusEl.className = "job-status error";
        statusEl.textContent = e.message;
        finish(reject, e);
      }
    };

    poll();
  });
}

async function runFullComparisonAndRefresh({
  preserveZoom = true,
  message = "Running comparison…",
  pushFirst = true,
  wrapLoading = true,
} = {}) {
  if (!session?.row_count) return;

  const runId = ++comparisonRunId;

  const run = async () => {
    if (pushFirst) await pushSettings(undefined, false);
    if (runId !== comparisonRunId) return;

    await api("POST", "/api/comparison/clear", undefined, false);
    if (runId !== comparisonRunId) return;

    const { job_id } = await api("POST", "/api/comparison/full", undefined, false);
    if (runId !== comparisonRunId) return;

    try {
      await waitForComparisonJob(job_id);
    } catch (e) {
      if (isComparisonCancelled(e)) return;
      throw e;
    }
    if (runId !== comparisonRunId) return;

    session = await api("GET", "/api/session", undefined, false);
    settings = session.settings;
    applySettingsToForm(settings);
    updateSliderRange(session.comparison_slider);
    await refreshPlot({ preserveZoom });
  };

  if (wrapLoading) {
    try {
      return await withLoading(message, run);
    } catch (e) {
      if (isComparisonCancelled(e)) return;
      throw e;
    }
  }
  updateLoadingMessage(message);
  return run();
}

// ------------------------------------------------------------------ settings sync
function readFormSettings() {
  const endVal = document.getElementById("end-index").value;
  return {
    start_index: parseInt(document.getElementById("start-index").value, 10) || 0,
    end_index: endVal === "" ? null : parseInt(endVal, 10),
    horizon_steps: parseInt(document.getElementById("horizon-steps").value, 10) || 50,
    steering_delay_steps: parseInt(document.getElementById("steering-delay").value, 10) || 0,
    acceleration_delay_steps: parseInt(document.getElementById("accel-delay").value, 10) || 0,
    enable_comparison: document.getElementById("enable-comparison").checked,
    show_controls: document.getElementById("show-controls").checked,
    show_delta_state: document.getElementById("show-delta").checked,
    show_all_comparisons: document.getElementById("show-all-comparisons").checked,
    sync_scales: document.getElementById("sync-scales").checked,
    show_metrics: document.getElementById("show-metrics").checked,
    state_name: document.getElementById("state-select").value,
    selected_other_data: getSelectedOtherData(),
    comparison_start_index: parseInt(document.getElementById("comparison-slider").value, 10) || 0,
    default_car_model: document.getElementById("model-select").value,
    default_car_parameters: document.getElementById("params-select").value,
    theme: settings.theme === "light" ? "light" : "dark",
  };
}

function applySettingsToForm(s) {
  settings = s;
  applyTheme(s.theme || "dark");
  document.getElementById("start-index").value = s.start_index ?? 0;
  document.getElementById("end-index").value = s.end_index ?? "";
  document.getElementById("horizon-steps").value = s.horizon_steps ?? 50;
  document.getElementById("steering-delay").value = s.steering_delay_steps ?? 0;
  document.getElementById("accel-delay").value = s.acceleration_delay_steps ?? 0;
  document.getElementById("enable-comparison").checked = s.enable_comparison ?? true;
  document.getElementById("show-controls").checked = s.show_controls ?? false;
  document.getElementById("show-delta").checked = s.show_delta_state ?? false;
  document.getElementById("show-all-comparisons").checked = s.show_all_comparisons ?? false;
  document.getElementById("sync-scales").checked = s.sync_scales ?? false;
  document.getElementById("show-metrics").checked = s.show_metrics ?? true;
  if (s.default_car_model) document.getElementById("model-select").value = s.default_car_model;
  if (s.default_car_parameters) document.getElementById("params-select").value = s.default_car_parameters;
}

async function pushSettings(extra = {}, trackLoading = false) {
  const data = { ...readFormSettings(), ...extra };
  settings = await api("PUT", "/api/config", data, trackLoading);
  return settings;
}

function getSelectedOtherData() {
  const items = document.querySelectorAll("#other-data-list li");
  return Array.from(items).map((li) => li.dataset.col);
}

function renderOtherDataList(cols) {
  const ul = document.getElementById("other-data-list");
  ul.innerHTML = "";
  cols.forEach((col) => {
    const li = document.createElement("li");
    li.dataset.col = col;
    li.innerHTML = `<span>${col}</span><button type="button" title="Remove">×</button>`;
    li.querySelector("button").addEventListener("click", async () => {
      const remaining = getSelectedOtherData().filter((c) => c !== col);
      await pushSettings({ selected_other_data: remaining }, false);
      renderOtherDataList(remaining);
      await refreshPlot({ preserveZoom: true, showLoading: true, loadingMessage: "Updating plot…" });
    });
    ul.appendChild(li);
  });
}

function updateSliderRange(sliderInfo) {
  const slider = document.getElementById("comparison-slider");
  slider.min = sliderInfo.min;
  slider.max = sliderInfo.max;
  const val = Math.max(sliderInfo.min, Math.min(settings.comparison_start_index ?? sliderInfo.min, sliderInfo.max));
  slider.value = val;
  document.getElementById("comparison-index-label").textContent = `Index: ${val}`;
}

function updateFileStatus() {
  const nameEl = document.getElementById("loaded-file-name");
  const pathEl = document.getElementById("loaded-file-path");
  if (session && session.filename) {
    nameEl.textContent = `${session.filename} (${session.row_count} rows)`;
    pathEl.textContent = session.csv_file_path || "";
    pathEl.title = session.csv_file_path || "";
  } else {
    nameEl.textContent = "No file loaded";
    pathEl.textContent = "";
    pathEl.title = "";
  }
}

function openDataModal() {
  const modal = document.getElementById("data-modal");
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
  loadBrowse(browsePath || "AnalyseData");
}

function closeDataModal() {
  const modal = document.getElementById("data-modal");
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
}

function populateStateSelect(states, selected) {
  const sel = document.getElementById("state-select");
  sel.innerHTML = "";
  states.forEach((s) => {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    if (s === selected) opt.selected = true;
    sel.appendChild(opt);
  });
}

function populateOtherDataSelect(columns, exclude) {
  const sel = document.getElementById("other-data-select");
  sel.innerHTML = '<option value="">— add column —</option>';
  columns.filter((c) => !exclude.includes(c)).forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  });
}

// ------------------------------------------------------------------ file browser
async function loadBrowse(path) {
  browsePath = path || "AnalyseData";
  const data = await api("GET", `/api/csv/browse?path=${encodeURIComponent(browsePath)}`);
  document.getElementById("browse-path").textContent = data.current_path || "(root)";
  const ul = document.getElementById("file-list");
  ul.innerHTML = "";
  if (data.parent_path !== undefined && data.current_path) {
    const li = document.createElement("li");
    li.className = "dir";
    li.textContent = "..";
    li.addEventListener("click", () => loadBrowse(data.parent_path ?? ""));
    ul.appendChild(li);
  }
  data.entries.forEach((entry) => {
    const li = document.createElement("li");
    li.className = entry.type;
    li.textContent = entry.name;
    li.addEventListener("click", async () => {
      if (entry.type === "dir") {
        loadBrowse(entry.path);
      } else {
        try {
          await withLoading("Loading CSV…", async () => {
            session = await api("POST", "/api/csv/load", { path: entry.path }, false);
            settings = session.settings;
            applySettingsToForm(settings);
            populateStateSelect(session.available_states, settings.state_name);
            populateOtherDataSelect(session.columns, getSelectedOtherData());
            renderOtherDataList(settings.selected_other_data || []);
            updateSliderRange(session.comparison_slider);
            updateFileStatus();
            closeDataModal();
            updateLoadingMessage("Running comparison…");
            await runFullComparisonAndRefresh({
              preserveZoom: false,
              pushFirst: false,
              message: "Running comparison…",
              wrapLoading: false,
            });
          });
        } catch (e) {
          alert(e.message);
        }
      }
    });
    ul.appendChild(li);
  });
}

async function onModelOrParamsChanged() {
  try {
    await pushSettings(undefined, false);
    await runFullComparisonAndRefresh({
      preserveZoom: true,
      pushFirst: false,
      message: "Running comparison…",
    });
  } catch (e) {
    if (!isComparisonCancelled(e)) alert(e.message);
  }
}

// ------------------------------------------------------------------ init
async function init() {
  await withLoading("Initializing…", async () => {
    const options = await api("GET", "/api/options", undefined, false);

    const modelSel = document.getElementById("model-select");
    options.models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.label;
      opt.textContent = m.label;
      modelSel.appendChild(opt);
    });

    const paramsSel = document.getElementById("params-select");
    options.car_parameters.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p;
      paramsSel.appendChild(opt);
    });

    settings = await api("GET", "/api/config", undefined, false);
    applySettingsToForm(settings);
    renderOtherDataList(settings.selected_other_data || []);

    try {
      session = await api("GET", "/api/session", undefined, false);
      if (session.row_count > 0) {
        populateStateSelect(session.available_states, settings.state_name);
        populateOtherDataSelect(session.columns, getSelectedOtherData());
        updateSliderRange(session.comparison_slider);
        updateFileStatus();
      }
    } catch (e) {
      console.warn("Session:", e.message);
    }

    if (session?.row_count > 0 && settings.enable_comparison !== false) {
      try {
        await runFullComparisonAndRefresh({
          preserveZoom: false,
          pushFirst: false,
          message: "Running comparison…",
          wrapLoading: false,
        });
      } catch (e) {
        if (isComparisonCancelled(e)) return;
        console.warn("Initial comparison:", e.message);
        await refreshPlot({ preserveZoom: false });
      }
    } else if (session?.row_count > 0) {
      updateLoadingMessage("Loading plot…");
      await refreshPlot({ preserveZoom: false });
    }
  });

  bindEventListeners();
}

function bindCollapsibleUi() {
  const layout = document.getElementById("layout");
  const shell = document.getElementById("sidebar-shell");
  const sidebar = document.getElementById("sidebar");
  const toggleBtn = document.getElementById("toggle-sidebar");
  const SIDEBAR_KEY = "viz-sidebar-collapsed";
  const PANELS_KEY = "viz-panel-collapsed";
  const SLIDE_MS = 220;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let resizeAfterAnimTimer = null;
  let animEndTimer = null;
  let collapsed = false;

  const resizeChartDeferred = () => {
    clearTimeout(resizeAfterAnimTimer);
    resizeAfterAnimTimer = setTimeout(() => {
      const chart = document.getElementById("chart-container");
      if (!chart?.data) return;
      const run = () => Plotly.Plots.resize(chart);
      if (typeof requestIdleCallback === "function") {
        requestIdleCallback(run, { timeout: 800 });
      } else {
        setTimeout(run, 0);
      }
    }, 50);
  };

  const clearAnimClasses = () => {
    layout.classList.remove("sidebar-animating", "sidebar-hiding", "sidebar-showing");
    shell?.classList.remove("sidebar-slide-active");
    clearTimeout(animEndTimer);
  };

  const applyCollapsedUi = (isCollapsed) => {
    collapsed = isCollapsed;
    layout.classList.toggle("sidebar-collapsed", isCollapsed);
    toggleBtn.setAttribute("aria-expanded", String(!isCollapsed));
    toggleBtn.textContent = isCollapsed ? "▶" : "◀";
    toggleBtn.title = isCollapsed ? "Expand settings panel" : "Collapse settings panel";
    try {
      localStorage.setItem(SIDEBAR_KEY, isCollapsed ? "1" : "0");
    } catch (_) { /* ignore */ }
  };

  const finishAnim = () => {
    clearAnimClasses();
    resizeChartDeferred();
  };

  const waitForSidebarTransition = (onDone) => {
    if (!sidebar || reducedMotion) {
      onDone();
      return;
    }
    const handler = (e) => {
      if (e.target !== sidebar || e.propertyName !== "transform") return;
      sidebar.removeEventListener("transitionend", handler);
      clearTimeout(animEndTimer);
      onDone();
    };
    sidebar.addEventListener("transitionend", handler);
    animEndTimer = setTimeout(() => {
      sidebar.removeEventListener("transitionend", handler);
      onDone();
    }, SLIDE_MS + 80);
  };

  const waitForShellTransition = (onDone) => {
    if (!shell || reducedMotion) {
      onDone();
      return;
    }
    const handler = (e) => {
      if (e.target !== shell || (e.propertyName !== "width" && e.propertyName !== "max-height")) return;
      shell.removeEventListener("transitionend", handler);
      clearTimeout(animEndTimer);
      onDone();
    };
    shell.addEventListener("transitionend", handler);
    animEndTimer = setTimeout(() => {
      shell.removeEventListener("transitionend", handler);
      onDone();
    }, SLIDE_MS + 80);
  };

  const setSidebarCollapsed = (targetCollapsed, { animate = true } = {}) => {
    if (targetCollapsed === collapsed && animate) return;

    if (!animate || reducedMotion) {
      clearAnimClasses();
      applyCollapsedUi(targetCollapsed);
      layout.classList.add("sidebar-no-transition");
      requestAnimationFrame(() => {
        layout.classList.remove("sidebar-no-transition");
        resizeChartDeferred();
      });
      return;
    }

    clearAnimClasses();
    layout.classList.add("sidebar-animating");
    shell?.classList.add("sidebar-slide-active");

    if (targetCollapsed) {
      layout.classList.add("sidebar-hiding");
      waitForSidebarTransition(() => {
        layout.classList.remove("sidebar-hiding");
        applyCollapsedUi(true);
        waitForShellTransition(finishAnim);
      });
    } else {
      applyCollapsedUi(false);
      waitForShellTransition(finishAnim);
    }
  };

  try {
    if (localStorage.getItem(SIDEBAR_KEY) === "1") {
      collapsed = true;
      applyCollapsedUi(true);
      layout.classList.add("sidebar-no-transition");
      requestAnimationFrame(() => layout.classList.remove("sidebar-no-transition"));
    }
  } catch (_) { /* ignore */ }

  toggleBtn.addEventListener("click", () => {
    setSidebarCollapsed(!collapsed);
  });

  let panelState = {};
  try {
    panelState = JSON.parse(localStorage.getItem(PANELS_KEY) || "{}") || {};
  } catch (_) {
    panelState = {};
  }

  const savePanelState = () => {
    try {
      localStorage.setItem(PANELS_KEY, JSON.stringify(panelState));
    } catch (_) { /* ignore */ }
  };

  document.querySelectorAll(".panel-heading").forEach((btn) => {
    const panel = btn.closest(".panel");
    const id = panel?.dataset.panelId;
    if (id && panelState[id] === false) {
      btn.setAttribute("aria-expanded", "false");
    }

    btn.addEventListener("click", () => {
      const expanded = btn.getAttribute("aria-expanded") !== "false";
      const next = !expanded;
      btn.setAttribute("aria-expanded", String(next));
      if (id) {
        panelState[id] = next;
        savePanelState();
      }
    });
  });
}

function bindEventListeners() {
  bindCollapsibleUi();
  document.getElementById("open-data-modal").addEventListener("click", openDataModal);
  document.getElementById("close-data-modal").addEventListener("click", closeDataModal);
  document.getElementById("modal-backdrop").addEventListener("click", closeDataModal);

  document.getElementById("theme-dark").addEventListener("click", async () => {
    if (settings.theme === "dark") return;
    applyTheme("dark");
    await pushSettings({ theme: "dark" });
    if (session?.row_count > 0) {
      await refreshPlot({ preserveZoom: true, showLoading: true, loadingMessage: "Updating plot…" });
    }
  });

  document.getElementById("theme-light").addEventListener("click", async () => {
    if (settings.theme === "light") return;
    applyTheme("light");
    await pushSettings({ theme: "light" });
    if (session?.row_count > 0) {
      await refreshPlot({ preserveZoom: true, showLoading: true, loadingMessage: "Updating plot…" });
    }
  });

  document.getElementById("browse-up").addEventListener("click", async () => {
    const data = await api("GET", `/api/csv/browse?path=${encodeURIComponent(browsePath)}`);
    loadBrowse(data.parent_path ?? "");
  });

  document.getElementById("csv-upload").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      await withLoading("Uploading CSV…", async () => {
        const res = await fetch("/api/csv/upload", { method: "POST", body: form });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "Upload failed");
        }
        session = await res.json();
        settings = session.settings;
        applySettingsToForm(settings);
        populateStateSelect(session.available_states, settings.state_name);
        populateOtherDataSelect(session.columns, getSelectedOtherData());
        renderOtherDataList(settings.selected_other_data || []);
        updateSliderRange(session.comparison_slider);
        updateFileStatus();
        closeDataModal();
        updateLoadingMessage("Running comparison…");
        await runFullComparisonAndRefresh({
          preserveZoom: false,
          pushFirst: false,
          message: "Running comparison…",
          wrapLoading: false,
        });
      });
    } catch (err) {
      alert(err.message);
    }
    e.target.value = "";
  });

  document.getElementById("update-range").addEventListener("click", async () => {
    try {
      await withLoading("Updating range…", async () => {
        await pushSettings(undefined, false);
        session = await api("GET", "/api/session", undefined, false);
        updateSliderRange(session.comparison_slider);
        updateLoadingMessage("Running comparison…");
        await runFullComparisonAndRefresh({
          preserveZoom: true,
          pushFirst: false,
          message: "Running comparison…",
          wrapLoading: false,
        });
      });
    } catch (e) {
      alert(e.message);
    }
  });

  document.getElementById("reload-csv").addEventListener("click", async () => {
    try {
      await withLoading("Reloading CSV…", async () => {
        session = await api("POST", "/api/csv/reload", undefined, false);
        settings = session.settings;
        updateFileStatus();
        updateLoadingMessage("Running comparison…");
        await runFullComparisonAndRefresh({
          preserveZoom: true,
          pushFirst: false,
          message: "Running comparison…",
          wrapLoading: false,
        });
      });
    } catch (e) {
      alert(e.message);
    }
  });

  document.getElementById("state-select").addEventListener("change", async () => {
    await pushSettings(undefined, false);
    await refreshPlot({ preserveZoom: false, showLoading: true, loadingMessage: "Updating plot…" });
  });

  document.getElementById("other-data-select").addEventListener("change", async (e) => {
    const col = e.target.value;
    if (!col) return;
    const current = getSelectedOtherData();
    if (!current.includes(col)) {
      current.push(col);
      await pushSettings({ selected_other_data: current }, false);
      renderOtherDataList(current);
      populateOtherDataSelect(session?.columns || [], current);
      await refreshPlot({ preserveZoom: true, showLoading: true, loadingMessage: "Updating plot…" });
    }
    e.target.value = "";
  });

  document.getElementById("clear-other-data").addEventListener("click", async () => {
    await pushSettings({ selected_other_data: [] }, false);
    renderOtherDataList([]);
    populateOtherDataSelect(session?.columns || [], []);
    await refreshPlot({ preserveZoom: true, showLoading: true, loadingMessage: "Updating plot…" });
  });

  ["sync-scales", "show-all-comparisons", "show-metrics"].forEach((id) => {
    document.getElementById(id).addEventListener("change", async () => {
      await pushSettings(undefined, false);
      await refreshPlot({ preserveZoom: true, showLoading: true, loadingMessage: "Updating plot…" });
    });
  });

  ["show-controls", "show-delta"].forEach((id) => {
    document.getElementById(id).addEventListener("change", async () => {
      await pushSettings(undefined, false);
      await refreshPlot({ preserveZoom: false, showLoading: true, loadingMessage: "Updating plot…" });
    });
  });

  document.getElementById("enable-comparison").addEventListener("change", async () => {
    await pushSettings(undefined, false);
    if (document.getElementById("enable-comparison").checked && session?.row_count > 0) {
      await runFullComparisonAndRefresh({ preserveZoom: true, pushFirst: false });
    } else {
      await refreshPlot({ preserveZoom: true, showLoading: true, loadingMessage: "Updating plot…" });
    }
  });

  ["horizon-steps", "steering-delay", "accel-delay"].forEach((id) => {
    document.getElementById(id).addEventListener("change", async () => {
      await pushSettings(undefined, false);
      session = await api("GET", "/api/session", undefined, false);
      updateSliderRange(session.comparison_slider);
      if (session?.row_count > 0) {
        await runFullComparisonAndRefresh({ preserveZoom: true, pushFirst: false });
      }
    });
  });

  document.getElementById("model-select").addEventListener("change", () => onModelOrParamsChanged());
  document.getElementById("params-select").addEventListener("change", () => onModelOrParamsChanged());

  document.getElementById("run-full-comparison").addEventListener("click", async () => {
    try {
      await runFullComparisonAndRefresh({ preserveZoom: true });
    } catch (e) {
      if (!isComparisonCancelled(e)) alert(e.message);
    }
  });

  document.getElementById("comparison-slider").addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    document.getElementById("comparison-index-label").textContent = `Index: ${val}`;
    if (sliderTimer) clearTimeout(sliderTimer);
    sliderTimer = setTimeout(async () => {
      await withLoading("Updating comparison…", async () => {
        await pushSettings({ comparison_start_index: val }, false);
        if (settings.enable_comparison) {
          await api("POST", "/api/comparison/single", { start_index: val }, false);
        }
        await refreshPlot({ preserveZoom: true });
      });
    }, 300);
  });

  document.getElementById("save-settings").addEventListener("click", async () => {
    await pushSettings();
    alert("Settings saved.");
  });

  window.addEventListener("resize", () => {
    const container = document.getElementById("chart-container");
    if (container && container.data) {
      Plotly.Plots.resize(container);
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  startBrowserHeartbeat();
  init().catch((err) => {
    console.error("Init failed:", err);
    resetLoading();
  });
});
