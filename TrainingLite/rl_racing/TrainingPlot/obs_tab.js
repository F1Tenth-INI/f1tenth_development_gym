const OBS_API = "/api/obs-tracking";
const OBS_POLL_INTERVAL_MS = 2000;
const OBS_PLOT_FONT_FAMILY = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
const OBS_HIST_COLOR = "#72B7B2";

const obsState = {
  manifest: null,
  snapshot: null,
  snapshotIndex: 0,
  selectedDim: 0,
  viewMode: "single",
  pollId: null,
  active: false,
  loadGeneration: 0,
};

function lastSnapshotMeta(snapshots) {
  return snapshots.length > 0 ? snapshots[snapshots.length - 1] : null;
}

function setObsMetaLine(message) {
  const el = document.getElementById("obs-meta-line");
  if (el) {
    el.textContent = message;
  }
}

function setObsStatus(message, isError = false) {
  const el = document.getElementById("obs-status");
  if (!el) return;
  el.textContent = message;
  el.classList.toggle("error", isError);
  if (isError) {
    setObsMetaLine(message);
  }
}

function plotTitle(text, { subtitle = null } = {}) {
  return {
    text: subtitle
      ? `${text}<br><span style="font-size:14px;color:#9aa8b8">${subtitle}</span>`
      : text,
    font: { size: 20, color: "#f0f4f8", family: OBS_PLOT_FONT_FAMILY },
    x: 0.02,
    xanchor: "left",
  };
}

function obsApiUrl({ obsSeen = null, stats = false } = {}) {
  const modelName = window.TrainingPlot?.getSelectedModel?.();
  if (!modelName) {
    throw new Error("No model selected");
  }
  const params = new URLSearchParams({ model: modelName });
  if (obsSeen !== null && obsSeen !== undefined) {
    params.set("obs_seen", String(obsSeen));
  }
  if (stats) {
    params.set("stats", "1");
  }
  return `${OBS_API}?${params.toString()}`;
}

function manifestFetchKey(manifest) {
  const snapshots = manifest?.snapshots || [];
  if (snapshots.length === 0) {
    return "empty";
  }
  const last = lastSnapshotMeta(snapshots);
  return `${snapshots.length}:${last?.obs_seen ?? 0}:${last?.hist_sample_count ?? 0}`;
}

function snapshotList(manifest) {
  return Array.isArray(manifest?.snapshots) ? manifest.snapshots : [];
}

function currentSnapshotMeta() {
  const snapshots = snapshotList(obsState.manifest);
  if (snapshots.length === 0) {
    return null;
  }
  const index = Math.max(0, Math.min(snapshots.length - 1, obsState.snapshotIndex));
  return snapshots[index];
}

function formatSnapshotLabel(meta) {
  if (!meta) {
    return "—";
  }
  const seen = Number(meta.obs_seen) || 0;
  const sampleCount = Number(meta.hist_sample_count) || 0;
  const live = meta.is_live ? " · live" : "";
  return `${seen.toLocaleString()} observations · ${sampleCount.toLocaleString()} hist samples${live}`;
}

function histogramTrace(dim) {
  const counts = Array.isArray(dim?.counts) ? dim.counts : [];
  const edges = Array.isArray(dim?.edges) ? dim.edges : [];
  if (counts.length === 0 || edges.length !== counts.length + 1) {
    return null;
  }
  const centers = [];
  for (let i = 0; i < counts.length; i += 1) {
    centers.push(0.5 * (edges[i] + edges[i + 1]));
  }
  return {
    type: "bar",
    x: centers,
    y: counts,
    marker: { color: OBS_HIST_COLOR, line: { color: "#0f1419", width: 0.3 } },
    hovertemplate: "value=%{x:.4f}<br>count=%{y}<extra></extra>",
    showlegend: false,
  };
}

function dimStatsSubtitle(dim) {
  if (!dim) {
    return "";
  }
  return `mean=${Number(dim.mean).toFixed(4)} · std=${Number(dim.std).toFixed(4)} · min=${Number(dim.min).toFixed(4)} · max=${Number(dim.max).toFixed(4)}`;
}

function populateDimSelect() {
  const select = document.getElementById("obs-dim-select");
  if (!select) return;
  const dims = obsState.snapshot?.dims || [];
  const previous = obsState.selectedDim;
  while (select.firstChild) {
    select.removeChild(select.firstChild);
  }
  if (dims.length === 0) {
    const option = document.createElement("option");
    option.value = "0";
    option.textContent = "No dimensions";
    select.appendChild(option);
    select.disabled = true;
    return;
  }
  select.disabled = false;
  for (const dim of dims) {
    const option = document.createElement("option");
    option.value = String(dim.idx);
    option.textContent = `obs[${dim.idx}]`;
    select.appendChild(option);
  }
  const hasPrevious = dims.some((dim) => Number(dim.idx) === Number(previous));
  obsState.selectedDim = hasPrevious ? previous : Number(dims[0].idx);
  select.value = String(obsState.selectedDim);
}

function updateSnapshotSliderUi() {
  const slider = document.getElementById("obs-snapshot-slider");
  const output = document.getElementById("obs-snapshot-value");
  const snapshots = snapshotList(obsState.manifest);
  const count = snapshots.length;
  const summary = obsState.manifest?.summary || {};
  const latest = lastSnapshotMeta(snapshots);
  const obsDim = Number(summary.obs_dim) || Number(latest?.obs_dim) || 0;
  const modelName = window.TrainingPlot?.getSelectedModel?.() || "—";

  setObsMetaLine(
    count > 0
      ? `${modelName} · ${count} snapshot(s) · obs_dim=${obsDim}`
      : `${modelName} · waiting for obs_tracking flush…`,
  );

  if (!slider) return;
  if (count === 0) {
    slider.disabled = true;
    slider.min = "0";
    slider.max = "0";
    slider.value = "0";
    if (output) output.textContent = "No snapshots yet";
    return;
  }

  slider.disabled = false;
  slider.min = "0";
  slider.max = String(count - 1);
  const index = Math.max(0, Math.min(count - 1, obsState.snapshotIndex));
  obsState.snapshotIndex = index;
  slider.value = String(index);
  if (output) {
    output.textContent = formatSnapshotLabel(snapshots[index]);
  }
}

function waitForVisiblePanel() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(resolve);
    });
  });
}

function resizeObsPlots() {
  if (!window.Plotly?.Plots?.resize) {
    return;
  }
  for (const id of ["plot-obs-distributions", "plot-obs-evolution"]) {
    const el = document.getElementById(id);
    if (el) {
      Plotly.Plots.resize(el);
    }
  }
}

async function renderSingleDimPlot() {
  if (!window.Plotly?.react) {
    setObsStatus("Plotly failed to load.", true);
    return;
  }
  await waitForVisiblePanel();
  const plotId = "plot-obs-distributions";
  const dims = obsState.snapshot?.dims || [];
  const dim = dims.find((item) => Number(item.idx) === Number(obsState.selectedDim)) || dims[0];
  const meta = currentSnapshotMeta();

  if (!dim || !Array.isArray(dim.counts) || dim.counts.length === 0) {
    await Plotly.react(
      plotId,
      [],
      {
        title: plotTitle("Observation distribution", {
          subtitle: meta ? formatSnapshotLabel(meta) : "Waiting for obs_tracking data…",
        }),
        paper_bgcolor: "#0f1419",
        plot_bgcolor: "#121a24",
        font: { color: "#e8edf2", family: OBS_PLOT_FONT_FAMILY },
        margin: { l: 55, r: 25, t: 80, b: 55 },
        annotations: [{
          text: "No histogram data yet — snapshots are written on each obs tracker flush.",
          xref: "paper",
          yref: "paper",
          x: 0.5,
          y: 0.5,
          showarrow: false,
          font: { color: "#9aa8b8", size: 14, family: OBS_PLOT_FONT_FAMILY },
        }],
      },
      { responsive: true, displayModeBar: true },
    );
    resizeObsPlots();
    return;
  }

  const trace = histogramTrace(dim);
  await Plotly.react(
    plotId,
    trace ? [trace] : [],
    {
      title: plotTitle(`obs[${dim.idx}] distribution`, {
        subtitle: `${formatSnapshotLabel(meta)} · ${dimStatsSubtitle(dim)}`,
      }),
      paper_bgcolor: "#0f1419",
      plot_bgcolor: "#121a24",
      font: { color: "#e8edf2", family: OBS_PLOT_FONT_FAMILY, size: 13 },
      margin: { l: 55, r: 25, t: 90, b: 55 },
      xaxis: {
        title: { text: "Value", font: { size: 13, color: "#c5d0db" } },
        gridcolor: "#2a3441",
      },
      yaxis: {
        title: { text: "Count", font: { size: 13, color: "#c5d0db" } },
        gridcolor: "#2a3441",
        zerolinecolor: "#6b7280",
      },
    },
    { responsive: true, displayModeBar: true },
  );
  resizeObsPlots();
}

async function renderEvolutionPlot() {
  if (!window.Plotly?.react) {
    return;
  }
  await waitForVisiblePanel();
  const plotId = "plot-obs-evolution";
  const dimIdx = Number(obsState.selectedDim);
  const history = Array.isArray(obsState.manifest?.stats_history)
    ? obsState.manifest.stats_history.filter((row) => Number(row.obs_idx) === dimIdx)
    : [];

  if (history.length === 0) {
    await Plotly.react(plotId, [], {
      paper_bgcolor: "#0f1419",
      plot_bgcolor: "#121a24",
      font: { color: "#e8edf2", family: OBS_PLOT_FONT_FAMILY },
      margin: { l: 55, r: 25, t: 40, b: 55 },
      annotations: [{
        text: "Stats evolution appears after multiple tracker flushes.",
        xref: "paper",
        yref: "paper",
        x: 0.5,
        y: 0.5,
        showarrow: false,
        font: { color: "#9aa8b8", size: 13, family: OBS_PLOT_FONT_FAMILY },
      }],
    }, { responsive: true, displayModeBar: false });
    resizeObsPlots();
    return;
  }

  history.sort((a, b) => Number(a.obs_seen) - Number(b.obs_seen));
  const xs = history.map((row) => Number(row.obs_seen));
  const means = history.map((row) => Number(row.mean));
  const stds = history.map((row) => Number(row.std));
  const mins = history.map((row) => Number(row.min));
  const maxs = history.map((row) => Number(row.max));
  const upper = means.map((m, i) => m + stds[i]);
  const lower = means.map((m, i) => m - stds[i]);

  await Plotly.react(
    plotId,
    [
      {
        type: "scatter",
        mode: "lines+markers",
        name: "mean",
        x: xs,
        y: means,
        line: { color: "#8ab4f8", width: 2 },
        marker: { size: 5 },
      },
      {
        type: "scatter",
        mode: "lines",
        name: "mean ± std",
        x: xs.concat(xs.slice().reverse()),
        y: upper.concat(lower.slice().reverse()),
        fill: "toself",
        fillcolor: "rgba(138, 180, 248, 0.15)",
        line: { color: "rgba(138, 180, 248, 0.0)" },
        hoverinfo: "skip",
        showlegend: true,
      },
      {
        type: "scatter",
        mode: "lines",
        name: "min",
        x: xs,
        y: mins,
        line: { color: "#f28b82", width: 1, dash: "dot" },
      },
      {
        type: "scatter",
        mode: "lines",
        name: "max",
        x: xs,
        y: maxs,
        line: { color: "#81c995", width: 1, dash: "dot" },
      },
    ],
    {
      title: {
        text: `obs[${dimIdx}] stats over training`,
        font: { size: 15, color: "#f0f4f8", family: OBS_PLOT_FONT_FAMILY },
        x: 0.02,
        xanchor: "left",
      },
      paper_bgcolor: "#0f1419",
      plot_bgcolor: "#121a24",
      font: { color: "#e8edf2", family: OBS_PLOT_FONT_FAMILY, size: 12 },
      margin: { l: 55, r: 25, t: 50, b: 55 },
      xaxis: {
        title: { text: "Observations seen", font: { size: 12, color: "#c5d0db" } },
        gridcolor: "#2a3441",
      },
      yaxis: {
        title: { text: "Value", font: { size: 12, color: "#c5d0db" } },
        gridcolor: "#2a3441",
        zerolinecolor: "#6b7280",
      },
      legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11 } },
    },
    { responsive: true, displayModeBar: false },
  );
  resizeObsPlots();
}

async function renderObsPlots() {
  if (obsState.viewMode === "grid") {
    await renderGridPlot();
  } else {
    await renderSingleDimPlot();
  }
  await renderEvolutionPlot();
}

async function renderGridPlot() {
  if (!window.Plotly?.react) {
    setObsStatus("Plotly failed to load.", true);
    return;
  }
  await waitForVisiblePanel();
  const plotId = "plot-obs-distributions";
  const dims = (obsState.snapshot?.dims || []).filter((dim) => Array.isArray(dim.counts) && dim.counts.length > 0);
  const meta = currentSnapshotMeta();

  if (dims.length === 0) {
    await renderSingleDimPlot();
    return;
  }

  const nCols = Math.ceil(Math.sqrt(dims.length));
  const nRows = Math.ceil(dims.length / nCols);
  const traces = [];
  const layout = {
    title: plotTitle("Observation histograms", {
      subtitle: formatSnapshotLabel(meta),
    }),
    paper_bgcolor: "#0f1419",
    plot_bgcolor: "#121a24",
    font: { color: "#e8edf2", family: OBS_PLOT_FONT_FAMILY, size: 11 },
    margin: { l: 40, r: 20, t: 90, b: 40 },
    grid: { rows: nRows, columns: nCols, pattern: "independent", roworder: "top to bottom" },
    showlegend: false,
  };

  for (let i = 0; i < dims.length; i += 1) {
    const dim = dims[i];
    const trace = histogramTrace(dim);
    if (!trace) continue;
    const axisSuffix = i === 0 ? "" : String(i + 1);
    trace.xaxis = `x${axisSuffix}`;
    trace.yaxis = `y${axisSuffix}`;
    traces.push(trace);

    layout[`xaxis${axisSuffix}`] = {
      title: { text: `obs[${dim.idx}]`, font: { size: 10, color: "#c5d0db" } },
      tickfont: { size: 8 },
      gridcolor: "#2a3441",
    };
    layout[`yaxis${axisSuffix}`] = {
      tickfont: { size: 8 },
      gridcolor: "#2a3441",
      zerolinecolor: "#6b7280",
    };
  }

  await Plotly.react(plotId, traces, layout, { responsive: true, displayModeBar: true });
  resizeObsPlots();
}

async function fetchObsStatsHistory(generation) {
  try {
    const response = await fetch(obsApiUrl({ stats: true }), { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    if (generation !== obsState.loadGeneration || !obsState.manifest) {
      return;
    }
    obsState.manifest.stats_history = Array.isArray(payload.stats_history) ? payload.stats_history : [];
  } catch (_error) {
    // Evolution chart is optional; ignore stats-history failures.
  }
}

async function loadSnapshotForIndex(index, { preserveDim = true, generation = obsState.loadGeneration } = {}) {
  const snapshots = snapshotList(obsState.manifest);
  if (snapshots.length === 0) {
    obsState.snapshot = null;
    populateDimSelect();
    updateSnapshotSliderUi();
    await renderObsPlots();
    return;
  }

  const clamped = Math.max(0, Math.min(snapshots.length - 1, index));
  obsState.snapshotIndex = clamped;
  const meta = snapshots[clamped];
  try {
    const response = await fetch(obsApiUrl({ obsSeen: meta.obs_seen }), { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load obs snapshot: HTTP ${response.status}`);
    }
    const payload = await response.json();
    if (generation !== obsState.loadGeneration) {
      return;
    }
    obsState.snapshot = payload.snapshot || null;
    if (!obsState.snapshot) {
      throw new Error(payload.error || `Snapshot missing for obs_seen=${meta.obs_seen}`);
    }
    if (!preserveDim && obsState.snapshot.dims?.length) {
      obsState.selectedDim = Number(obsState.snapshot.dims[0].idx);
    }
    populateDimSelect();
    updateSnapshotSliderUi();
    await renderObsPlots();
    setObsStatus(formatSnapshotLabel(meta));
  } catch (error) {
    setObsStatus(error.message, true);
  }
}

async function loadObsManifest({ preserveIndex = true } = {}) {
  const generation = obsState.loadGeneration + 1;
  obsState.loadGeneration = generation;
  const modelName = window.TrainingPlot?.getSelectedModel?.();

  if (!modelName) {
    obsState.manifest = { snapshots: [] };
    obsState.snapshot = null;
    updateSnapshotSliderUi();
    await renderObsPlots();
    return;
  }

  setObsMetaLine(`Loading ${modelName} obs_tracking data…`);

  try {
    const response = await fetch(obsApiUrl(), { cache: "no-store" });
    if (!response.ok) {
      if (response.status === 404) {
        obsState.manifest = { snapshots: [] };
        obsState.snapshot = null;
        updateSnapshotSliderUi();
        await renderObsPlots();
        setObsStatus("No obs_tracking data for this model yet.");
        return;
      }
      throw new Error(`Failed to load obs tracking manifest: HTTP ${response.status}`);
    }

    const payload = await response.json();
    if (generation !== obsState.loadGeneration) {
      return;
    }

    const prevKey = manifestFetchKey(obsState.manifest);
    const nextKey = manifestFetchKey(payload);
    const prevIndex = obsState.snapshotIndex;
    obsState.manifest = payload;
    void fetchObsStatsHistory(generation).then(() => {
      if (generation === obsState.loadGeneration) {
        void renderEvolutionPlot();
      }
    });

    const snapshots = snapshotList(payload);
    if (snapshots.length === 0) {
      obsState.snapshot = null;
      obsState.snapshotIndex = 0;
      populateDimSelect();
      updateSnapshotSliderUi();
      await renderObsPlots();
      setObsStatus("Waiting for first obs tracker flush (every ~10k observations)…");
      return;
    }

    let nextIndex = snapshots.length - 1;
    if (preserveIndex && prevKey === nextKey) {
      nextIndex = Math.min(prevIndex, snapshots.length - 1);
    } else if (preserveIndex && prevIndex < snapshots.length) {
      nextIndex = prevIndex;
    }

    if (nextKey !== prevKey || !obsState.snapshot) {
      await loadSnapshotForIndex(nextIndex, { preserveDim: preserveIndex, generation });
    } else {
      updateSnapshotSliderUi();
      await renderObsPlots();
      setObsStatus(formatSnapshotLabel(snapshots[nextIndex]));
    }
  } catch (error) {
    if (generation === obsState.loadGeneration) {
      setObsStatus(error.message, true);
    }
  }
}

function stopObsPolling() {
  if (obsState.pollId !== null) {
    window.clearInterval(obsState.pollId);
    obsState.pollId = null;
  }
}

function startObsPolling() {
  stopObsPolling();
  obsState.pollId = window.setInterval(() => {
    void loadObsManifest({ preserveIndex: true }).catch((error) => {
      setObsStatus(error.message, true);
    });
  }, OBS_POLL_INTERVAL_MS);
}

function bindObsUi() {
  document.getElementById("obs-snapshot-slider")?.addEventListener("input", (event) => {
    const index = Number.parseInt(event.target.value, 10) || 0;
    obsState.snapshotIndex = index;
    updateSnapshotSliderUi();
  });
  document.getElementById("obs-snapshot-slider")?.addEventListener("change", async (event) => {
    const index = Number.parseInt(event.target.value, 10) || 0;
    await loadSnapshotForIndex(index, { preserveDim: true });
  });

  document.getElementById("obs-dim-select")?.addEventListener("change", async (event) => {
    obsState.selectedDim = Number.parseInt(event.target.value, 10) || 0;
    await renderObsPlots();
  });

  document.getElementById("obs-view-mode")?.addEventListener("change", async (event) => {
    obsState.viewMode = event.target.value === "grid" ? "grid" : "single";
    await renderObsPlots();
  });
}

function onModelChanged() {
  obsState.snapshotIndex = 0;
  obsState.selectedDim = 0;
  obsState.snapshot = null;
  obsState.manifest = null;
  if (obsState.active) {
    void loadObsManifest({ preserveIndex: false });
  }
}

function startObsTab() {
  obsState.active = true;
  startObsPolling();
  void loadObsManifest({ preserveIndex: false });
}

function stopObsTab() {
  obsState.active = false;
  stopObsPolling();
}

function onTabChanged(tabName) {
  if (tabName === "observations") {
    startObsTab();
  } else {
    stopObsTab();
  }
}

bindObsUi();
setObsMetaLine("Observations tab ready — open this tab to load data.");
window.TrainingPlot?.onModelChange(onModelChanged);
window.TrainingPlot?.onTabChange(onTabChanged);

window.ObsTab = {
  start: startObsTab,
  refresh: () => {
    if (!obsState.active) {
      return Promise.resolve();
    }
    return loadObsManifest({ preserveIndex: true });
  },
  reload: () => loadObsManifest({ preserveIndex: false }),
};

if (window.TrainingPlot?.getActiveTab?.() === "observations") {
  startObsTab();
}
