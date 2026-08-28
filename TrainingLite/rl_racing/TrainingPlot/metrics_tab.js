const METRICS_API = "/api/metrics";
const TRACE_COLOR = "#8ab4f8";
const X_AXIS_STORAGE_KEY = "trainingplot.metrics.xAxis";
const SERIES_STORAGE_KEY = "trainingplot.metrics.seriesVisible";

const metricsState = {
  pollTimer: null,
  lastMtime: null,
  lastPayload: null,
  seriesSignature: null,
  seriesOverrides: loadSeriesOverrides(),
  active: false,
};

function loadSeriesOverrides() {
  try {
    const raw = window.localStorage.getItem(SERIES_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function saveSeriesOverrides() {
  window.localStorage.setItem(SERIES_STORAGE_KEY, JSON.stringify(metricsState.seriesOverrides));
}

const HIDDEN_BY_DEFAULT = new Set([
  "stream_batch_sizes",
  "difficulty",
  "reward_difficulty",
]);

function isIngestSeries(name) {
  return String(name || "").startsWith("ingest:");
}

function isDefaultVisible(name) {
  const seriesName = String(name || "");
  if (isIngestSeries(seriesName) || HIDDEN_BY_DEFAULT.has(seriesName)) {
    return false;
  }
  return true;
}

function isSeriesVisible(name) {
  if (Object.prototype.hasOwnProperty.call(metricsState.seriesOverrides, name)) {
    return Boolean(metricsState.seriesOverrides[name]);
  }
  return isDefaultVisible(name);
}

function setSeriesVisible(name, visible) {
  metricsState.seriesOverrides[name] = Boolean(visible);
}

function seriesNames(series) {
  return (series || []).map((item) => item.name).filter(Boolean);
}

function seriesSignature(series) {
  return seriesNames(series).join("|");
}

function applySeriesPreset(series, preset) {
  for (const name of seriesNames(series)) {
    if (preset === "all") setSeriesVisible(name, true);
    else if (preset === "none") setSeriesVisible(name, false);
    else setSeriesVisible(name, isDefaultVisible(name));
  }
  saveSeriesOverrides();
}

function filteredMetricsData(data) {
  const series = (data.series || []).filter((item) => isSeriesVisible(item.name));
  return { ...data, series };
}

function seriesToggleLabel(series) {
  const names = seriesNames(series);
  const total = names.length;
  const shown = names.filter((name) => isSeriesVisible(name)).length;
  if (total === 0) return "None";
  if (shown === 0) return "None";
  if (shown === total) return `All (${total})`;
  const matchesTrainingDefault = names.every((name) => isSeriesVisible(name) === isDefaultVisible(name));
  if (matchesTrainingDefault) return `Training (${shown})`;
  return `${shown} of ${total}`;
}

function updateSeriesToggleLabel(series) {
  const toggle = document.getElementById("metrics-series-toggle");
  if (toggle) toggle.textContent = seriesToggleLabel(series);
}

function renderSeriesPicker(series) {
  const list = document.getElementById("metrics-series-list");
  if (!list) return;

  const names = seriesNames(series);
  list.replaceChildren();
  if (!names.length) {
    const empty = document.createElement("div");
    empty.className = "metrics-series-group-label";
    empty.textContent = "No metrics yet";
    list.appendChild(empty);
    updateSeriesToggleLabel(series);
    return;
  }

  const groups = [
    { label: "Training", names: names.filter((name) => !isIngestSeries(name)) },
    { label: "Ingest", names: names.filter((name) => isIngestSeries(name)) },
  ];

  for (const group of groups) {
    if (!group.names.length) continue;
    const heading = document.createElement("div");
    heading.className = "metrics-series-group-label";
    heading.textContent = group.label;
    list.appendChild(heading);
    for (const name of group.names) {
      const item = document.createElement("label");
      item.className = "metrics-series-item";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = isSeriesVisible(name);
      checkbox.dataset.seriesName = name;
      const text = document.createElement("span");
      text.textContent = name;
      item.append(checkbox, text);
      list.appendChild(item);
    }
  }
  updateSeriesToggleLabel(series);
}

function setSeriesMenuOpen(open) {
  const toggle = document.getElementById("metrics-series-toggle");
  const menu = document.getElementById("metrics-series-menu");
  if (!toggle || !menu) return;
  menu.hidden = !open;
  toggle.setAttribute("aria-expanded", open ? "true" : "false");
}

function bindSeriesPicker() {
  const toggle = document.getElementById("metrics-series-toggle");
  const menu = document.getElementById("metrics-series-menu");
  const list = document.getElementById("metrics-series-list");
  if (!toggle || !menu || !list || toggle.dataset.bound === "1") return;

  toggle.addEventListener("click", (event) => {
    event.stopPropagation();
    setSeriesMenuOpen(menu.hidden);
  });

  menu.addEventListener("click", (event) => {
    event.stopPropagation();
    const actionBtn = event.target instanceof Element
      ? event.target.closest("[data-series-action]")
      : null;
    const action = actionBtn?.dataset.seriesAction;
    if (!action || !metricsState.lastPayload) return;
    applySeriesPreset(metricsState.lastPayload.series, action);
    renderSeriesPicker(metricsState.lastPayload.series);
    void renderMetrics(metricsState.lastPayload);
  });

  list.addEventListener("change", (event) => {
    const checkbox = event.target;
    if (!(checkbox instanceof HTMLInputElement) || checkbox.type !== "checkbox") return;
    const name = checkbox.dataset.seriesName;
    if (!name) return;
    setSeriesVisible(name, checkbox.checked);
    saveSeriesOverrides();
    updateSeriesToggleLabel(metricsState.lastPayload?.series);
    if (metricsState.lastPayload) void renderMetrics(metricsState.lastPayload);
  });

  document.addEventListener("click", () => setSeriesMenuOpen(false));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setSeriesMenuOpen(false);
  });
  toggle.dataset.bound = "1";
}

function getSelectedXAxis() {
  const select = document.getElementById("metrics-x-axis");
  const value = select?.value || window.localStorage.getItem(X_AXIS_STORAGE_KEY) || "wallclock";
  if (value === "simulation" || value === "timesteps") return value;
  return "wallclock";
}

function bindXAxisSelect() {
  const select = document.getElementById("metrics-x-axis");
  if (!select || select.dataset.bound === "1") return;
  const stored = window.localStorage.getItem(X_AXIS_STORAGE_KEY);
  if (stored === "simulation" || stored === "wallclock" || stored === "timesteps") {
    select.value = stored;
  }
  select.addEventListener("change", () => {
    window.localStorage.setItem(X_AXIS_STORAGE_KEY, getSelectedXAxis());
    metricsState.lastMtime = null;
    void refreshMetrics();
  });
  select.dataset.bound = "1";
}

function setMetricsStatus(text, isError = false) {
  const el = document.getElementById("metrics-status");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function isTouchScrollMode() {
  return (
    window.matchMedia("(pointer: coarse)").matches
    || window.matchMedia("(hover: none)").matches
    || window.matchMedia("(max-width: 900px)").matches
  );
}

function applyTouchScrollMode() {
  document.body.classList.toggle("touch-scroll", isTouchScrollMode());
}

function plotlyConfig() {
  const touchScroll = isTouchScrollMode();
  return {
    responsive: true,
    displayModeBar: false,
    scrollZoom: false,
    doubleClick: false,
    staticPlot: touchScroll,
  };
}

function niceDtick(span, targetTicks) {
  if (!Number.isFinite(span) || span <= 0) return 1;
  const raw = span / Math.max(1, targetTicks);
  const pow10 = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / pow10;
  let nice = 1;
  if (norm > 5) nice = 10;
  else if (norm > 2) nice = 5;
  else if (norm > 1) nice = 2;
  return nice * pow10;
}

function globalXExtent(series) {
  let min = Infinity;
  let max = -Infinity;
  for (const s of series) {
    for (const x of s.x || []) {
      const v = Number(x);
      if (Number.isFinite(v)) {
        min = Math.min(min, v);
        max = Math.max(max, v);
      }
    }
  }
  if (!Number.isFinite(min)) return [0, 1];
  if (min === max) return [min - 0.5, max + 0.5];
  const pad = (max - min) * 0.02;
  return [min - pad, max + pad];
}

function buildFigure(data) {
  const series = data.series || [];
  if (!series.length) {
    return null;
  }

  const n = series.length;
  const traces = [];
  const xRange = globalXExtent(series);
  const xDtick = niceDtick(xRange[1] - xRange[0], 8);
  const annotations = [];

  const layout = {
    grid: { rows: n, columns: 1, pattern: "independent", roworder: "top to bottom" },
    showlegend: false,
    paper_bgcolor: "#0f1115",
    plot_bgcolor: "#171a21",
    font: { color: "#e8eaed", size: 11 },
    margin: { l: 56, r: 16, t: 12, b: 44 },
    height: Math.max(420, 200 * n),
    annotations,
  };

  series.forEach((s, i) => {
    const row = i + 1;
    const xaxis = row === 1 ? "x" : `x${row}`;
    const yaxis = row === 1 ? "y" : `y${row}`;
    const xaxisLayoutKey = row === 1 ? "xaxis" : `xaxis${row}`;
    const yaxisLayoutKey = row === 1 ? "yaxis" : `yaxis${row}`;
    const isScatter = s.type === "scatter";

    traces.push({
      x: s.x,
      y: s.y,
      type: "scatter",
      mode: isScatter ? "markers" : "lines",
      marker: { size: 4, color: TRACE_COLOR },
      line: { width: 1.5, color: TRACE_COLOR },
      showlegend: false,
      hovertemplate: `${s.name}<br>%{x}<br>%{y}<extra></extra>`,
      xaxis,
      yaxis,
    });

    layout[yaxisLayoutKey] = {
      gridcolor: "#2a2f3a",
      zerolinecolor: "#2a2f3a",
    };

    layout[xaxisLayoutKey] = {
      gridcolor: "#2a2f3a",
      zerolinecolor: "#2a2f3a",
      showticklabels: true,
      range: xRange,
      dtick: xDtick,
      tickmode: "linear",
      ...(i === n - 1
        ? { title: { text: data.x_label || "step", font: { size: 11 } } }
        : {}),
    };

    annotations.push({
      text: `<b>${s.name}</b>`,
      showarrow: false,
      xref: `${xaxis} domain`,
      yref: `${yaxis} domain`,
      x: 0.01,
      y: 1.03,
      xanchor: "left",
      yanchor: "bottom",
      font: { size: 13, color: "#e8eaed", family: "system-ui, sans-serif" },
    });
  });

  return { traces, layout };
}

function metricsUrl(modelName) {
  const params = new URLSearchParams({ model: modelName, x: getSelectedXAxis() });
  return `${METRICS_API}?${params.toString()}`;
}

function stopMetricsPolling() {
  if (metricsState.pollTimer !== null) {
    window.clearTimeout(metricsState.pollTimer);
    metricsState.pollTimer = null;
  }
}

async function renderMetrics(data) {
  const chartEl = document.getElementById("metrics-chart");
  const emptyEl = document.getElementById("metrics-empty");
  const fig = buildFigure(filteredMetricsData(data));
  if (!fig) {
    if (chartEl) chartEl.hidden = true;
    if (emptyEl) {
      emptyEl.hidden = false;
      emptyEl.textContent = (data.series || []).length
        ? "No metrics selected — use the Metrics dropdown to choose series."
        : "No metrics yet — training logs will appear here.";
    }
    return false;
  }
  if (chartEl) chartEl.hidden = false;
  if (emptyEl) emptyEl.hidden = true;
  applyTouchScrollMode();
  await Plotly.react(chartEl, fig.traces, fig.layout, plotlyConfig());
  return true;
}

async function refreshMetrics() {
  if (!metricsState.active) return;

  const modelName = window.TrainingPlot?.getSelectedModel?.();
  const metaEl = document.getElementById("metrics-meta");
  const chartEl = document.getElementById("metrics-chart");
  const emptyEl = document.getElementById("metrics-empty");

  if (!modelName) {
    if (chartEl) chartEl.hidden = true;
    if (emptyEl) {
      emptyEl.hidden = false;
      emptyEl.textContent = "Select a model to view training metrics.";
    }
    setMetricsStatus("no model");
    metricsState.pollTimer = window.setTimeout(refreshMetrics, 3000);
    return;
  }

  try {
    const res = await fetch(metricsUrl(modelName), { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    if (data.error) {
      throw new Error(data.error);
    }

    if (metaEl) {
      const mtime = data.csv_mtime;
      const mtimeStr = mtime ? new Date(mtime * 1000).toLocaleTimeString() : "—";
      let xLabel = data.x_label || data.x_key || "step";
      if (data.x_axis === "simulation" && Number.isFinite(Number(data.sim_dt_s))) {
        xLabel = `${xLabel} · dt=${data.sim_dt_s}s`;
      }
      metaEl.textContent = `${data.model_name || modelName} · ${data.row_count || 0} rows · x: ${xLabel} · updated ${mtimeStr}`;
    }

    metricsState.lastPayload = data;
    const nextSignature = seriesSignature(data.series);
    if (nextSignature !== metricsState.seriesSignature) {
      metricsState.seriesSignature = nextSignature;
      renderSeriesPicker(data.series);
    } else {
      updateSeriesToggleLabel(data.series);
    }

    const plotted = await renderMetrics(data);
    if (!plotted) {
      setMetricsStatus((data.series || []).length ? "no series selected" : "no data yet");
    } else {
      const changed = data.csv_mtime !== metricsState.lastMtime;
      metricsState.lastMtime = data.csv_mtime;
      setMetricsStatus(changed ? "updated" : "live");
    }

    const intervalMs = Math.max(1000, (data.poll_interval_s || 2) * 1000);
    stopMetricsPolling();
    metricsState.pollTimer = window.setTimeout(refreshMetrics, intervalMs);
  } catch (err) {
    setMetricsStatus(`error: ${err.message}`, true);
    stopMetricsPolling();
    metricsState.pollTimer = window.setTimeout(refreshMetrics, 3000);
  }
}

function startMetricsTab() {
  metricsState.active = true;
  metricsState.lastMtime = null;
  bindXAxisSelect();
  bindSeriesPicker();
  applyTouchScrollMode();
  void refreshMetrics();
}

function stopMetricsTab() {
  metricsState.active = false;
  stopMetricsPolling();
}

function onModelChanged() {
  metricsState.lastMtime = null;
  if (metricsState.active) {
    void refreshMetrics();
  }
}

function onTabChanged(tabName) {
  if (tabName === "metrics") {
    startMetricsTab();
  } else {
    stopMetricsTab();
  }
}

applyTouchScrollMode();
bindXAxisSelect();
bindSeriesPicker();
window.addEventListener("resize", applyTouchScrollMode);
window.matchMedia("(pointer: coarse)").addEventListener?.("change", applyTouchScrollMode);

window.TrainingPlot?.onModelChange(onModelChanged);
window.TrainingPlot?.onTabChange(onTabChanged);

if (window.TrainingPlot?.getActiveTab?.() === "metrics") {
  startMetricsTab();
}
