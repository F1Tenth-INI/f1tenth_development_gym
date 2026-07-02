const METRICS_API = "/api/metrics";
const TRACE_COLOR = "#8ab4f8";

const metricsState = {
  pollTimer: null,
  lastMtime: null,
  active: false,
};

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
  const params = new URLSearchParams({ model: modelName });
  return `${METRICS_API}?${params.toString()}`;
}

function stopMetricsPolling() {
  if (metricsState.pollTimer !== null) {
    window.clearTimeout(metricsState.pollTimer);
    metricsState.pollTimer = null;
  }
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
      metaEl.textContent = `${data.model_name || modelName} · ${data.row_count || 0} rows · x: ${data.x_label || data.x_key} · updated ${mtimeStr}`;
    }

    const fig = buildFigure(data);
    if (!fig) {
      if (chartEl) chartEl.hidden = true;
      if (emptyEl) {
        emptyEl.hidden = false;
        emptyEl.textContent = "No metrics yet — training logs will appear here.";
      }
      setMetricsStatus("no data yet");
    } else {
      if (chartEl) chartEl.hidden = false;
      if (emptyEl) emptyEl.hidden = true;
      const changed = data.csv_mtime !== metricsState.lastMtime;
      metricsState.lastMtime = data.csv_mtime;
      applyTouchScrollMode();
      await Plotly.react(chartEl, fig.traces, fig.layout, plotlyConfig());
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
window.addEventListener("resize", applyTouchScrollMode);
window.matchMedia("(pointer: coarse)").addEventListener?.("change", applyTouchScrollMode);

window.TrainingPlot?.onModelChange(onModelChanged);
window.TrainingPlot?.onTabChange(onTabChanged);

if (window.TrainingPlot?.getActiveTab?.() === "metrics") {
  startMetricsTab();
}
