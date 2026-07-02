const EPISODES_API = "/api/episodes.csv";
const POLL_INTERVAL_MS = 2000;

const REWARD_COMPONENT_KEYS = [
  "progress",
  "crash_reward",
  "wp_distance_penalty",
  "d_action_penality",
  "speed_cap_penalty",
  "proximity_penalty",
  "stuck_reward",
  "spin_reward",
];

const REWARD_COMPONENT_COLORS = {
  progress: "#4C78A8",
  crash_reward: "#E45756",
  wp_distance_penalty: "#72B7B2",
  d_action_penality: "#F58518",
  speed_cap_penalty: "#54A24B",
  proximity_penalty: "#B279A2",
  stuck_reward: "#EECA3B",
  spin_reward: "#9D755D",
};

const PLOT_FONT_FAMILY = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif";

const rewardState = {
  payload: null,
  episodeIndex: 0,
  pollId: null,
  playing: false,
  playRafId: null,
  playLastTs: null,
  playPosition: 0,
  renderedIndex: -1,
  scrubbing: false,
  active: false,
};

function setRewardStatus(message, isError = false) {
  const el = document.getElementById("reward-status");
  if (!el) return;
  el.textContent = message;
  el.classList.toggle("error", isError);
}

function plotTitle(text, { subtitle = null, size = 20, x = 0.5, xanchor = "center" } = {}) {
  return {
    text: subtitle
      ? `${text}<br><span style="font-size:14px;color:#9aa8b8">${subtitle}</span>`
      : text,
    font: { size, color: "#f0f4f8", family: PLOT_FONT_FAMILY },
    x,
    xanchor,
  };
}

function subplotHeading(text, xref, { subtitle = null, y = 1.18 } = {}) {
  const items = [{
    text,
    xref,
    yref: "paper",
    x: 0.5,
    y,
    showarrow: false,
    align: "center",
    font: { size: 17, color: "#f0f4f8", family: PLOT_FONT_FAMILY },
    xanchor: "center",
  }];
  if (subtitle) {
    items.push({
      text: subtitle,
      xref,
      yref: "paper",
      x: 0.5,
      y: y - 0.048,
      showarrow: false,
      align: "center",
      font: { size: 12, color: "#9aa8b8", family: PLOT_FONT_FAMILY },
      xanchor: "center",
    });
  }
  return items;
}

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (ch === "," && !inQuotes) {
      values.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  values.push(current);
  return values;
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    return [];
  }
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });
    return row;
  });
}

function toFloat(value, fallback = 0) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toInt(value, fallback = 0) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function episodesPayloadFromCsv(text, csvMtime = 0) {
  const rows = parseCsv(text);
  const cumulative = Object.fromEntries(REWARD_COMPONENT_KEYS.map((key) => [key, 0]));
  const episodes = rows.map((row, rowIndex) => {
    const episodeAccumulated = {};
    for (const key of REWARD_COMPONENT_KEYS) {
      const value = toFloat(row[`comp_${key}`], 0);
      episodeAccumulated[key] = value;
      cumulative[key] += value;
    }

    return {
      episode_index: toInt(row.episode_index, rowIndex),
      actor_id: toInt(row.actor_id, 0),
      episode_id: toInt(row.episode_id, 0),
      length: toInt(row.length, 0),
      total_reward: toFloat(row.total_reward, 0),
      mean_reward: toFloat(row.mean_reward, 0),
      total_timesteps: toInt(row.total_timesteps, 0),
      time_s: toFloat(row.time_s, 0),
      lap_times: row.lap_times ?? null,
      episode_accumulated: { ...episodeAccumulated },
      total_accumulated: { ...cumulative },
    };
  });

  const modelName = window.TrainingPlot?.getSelectedModel?.();
  return {
    csv_path: modelName ? `${modelName}/episodes.csv` : "episodes.csv",
    csv_mtime: csvMtime,
    episodes,
    episode_count: episodes.length,
    component_keys: [...REWARD_COMPONENT_KEYS],
    component_colors: { ...REWARD_COMPONENT_COLORS },
  };
}

function payloadFetchKey(payload) {
  const episodes = Array.isArray(payload?.episodes) ? payload.episodes : [];
  if (episodes.length === 0) {
    return "empty";
  }
  const last = episodes[episodes.length - 1];
  return [
    episodes.length,
    payload?.csv_mtime ?? 0,
    last?.episode_index ?? 0,
    last?.total_timesteps ?? 0,
  ].join(":");
}

function formatEpisodeLabel(episode) {
  if (!episode) {
    return "—";
  }
  const idx = Number(episode.episode_index);
  const length = Number(episode.length) || 0;
  const timesteps = Number(episode.total_timesteps) || 0;
  return `Episode ${idx} · ${length} steps · ${timesteps.toLocaleString()} actor steps`;
}

function componentColors(keys) {
  return keys.map((key) => REWARD_COMPONENT_COLORS[key] || "#888888");
}

function componentValuesFromBag(bag, keys) {
  return keys.map((key) => Number(bag?.[key]) || 0);
}

function updateSliderUi() {
  const slider = document.getElementById("episode-slider");
  const output = document.getElementById("episode-slider-value");
  const meta = document.getElementById("episode-meta");
  const episodes = rewardState.payload?.episodes || [];
  const count = episodes.length;

  if (meta) {
    meta.textContent = `${count} episode(s)`;
  }

  if (count === 0) {
    slider.disabled = true;
    document.getElementById("episode-play")?.setAttribute("disabled", "disabled");
    slider.min = "0";
    slider.max = "0";
    slider.value = "0";
    if (output) {
      output.textContent = "No episode data yet";
    }
    return;
  }

  document.getElementById("episode-play")?.removeAttribute("disabled");
  slider.disabled = false;
  slider.min = "0";
  slider.max = String(count - 1);
  const index = Math.max(0, Math.min(count - 1, rewardState.episodeIndex));
  rewardState.episodeIndex = index;
  if (!rewardState.playing) {
    rewardState.playPosition = index;
    rewardState.renderedIndex = index;
  }
  if (!rewardState.scrubbing) {
    slider.step = rewardState.playing ? "any" : "1";
    slider.value = rewardState.playing ? String(rewardState.playPosition) : String(index);
  }
  if (output) {
    output.textContent = formatEpisodeLabel(episodes[index]);
  }
}

async function renderPlot() {
  const episodes = rewardState.payload?.episodes || [];
  const episode = episodes[rewardState.episodeIndex];
  const plotId = "plot-reward-components";
  const keys = rewardState.payload?.component_keys || REWARD_COMPONENT_KEYS;

  if (!episode) {
    Plotly.react(
      plotId,
      [],
      {
        title: plotTitle("Reward components", {
          subtitle: "Waiting for episodes.csv from training…",
          size: 20,
          x: 0.02,
          xanchor: "left",
        }),
        paper_bgcolor: "#0f1419",
        plot_bgcolor: "#121a24",
        font: { color: "#e8edf2", family: PLOT_FONT_FAMILY },
        margin: { l: 55, r: 25, t: 80, b: 110 },
        annotations: [{
          text: "Waiting for episodes.csv from training…",
          xref: "paper",
          yref: "paper",
          x: 0.5,
          y: 0.5,
          showarrow: false,
          font: { color: "#9aa8b8", size: 15, family: PLOT_FONT_FAMILY },
        }],
      },
      { responsive: true, displayModeBar: false },
    );
    return;
  }

  const colors = componentColors(keys);
  const totalValues = componentValuesFromBag(episode.total_accumulated, keys);
  const episodeValues = componentValuesFromBag(episode.episode_accumulated, keys);
  const totalSteps = Number(episode.total_timesteps) || 0;
  const episodeSteps = Number(episode.length) || 0;

  Plotly.react(
    plotId,
    [
      {
        type: "bar",
        x: keys,
        y: totalValues,
        marker: { color: colors, line: { color: "#0f1419", width: 0.4 } },
        hovertemplate: "%{x}<br>accumulated=%{y:.4f}<extra>Total</extra>",
        xaxis: "x",
        yaxis: "y",
        showlegend: false,
      },
      {
        type: "bar",
        x: keys,
        y: episodeValues,
        marker: { color: colors, line: { color: "#0f1419", width: 0.4 } },
        hovertemplate: "%{x}<br>accumulated=%{y:.4f}<extra>Episode</extra>",
        xaxis: "x2",
        yaxis: "y2",
        showlegend: false,
      },
    ],
    {
      title: plotTitle("Reward components", {
        subtitle: formatEpisodeLabel(episode),
        size: 22,
        x: 0.02,
        xanchor: "left",
      }),
      paper_bgcolor: "#0f1419",
      plot_bgcolor: "#121a24",
      font: { color: "#e8edf2", family: PLOT_FONT_FAMILY, size: 13 },
      margin: { l: 55, r: 25, t: 105, b: 110 },
      grid: { rows: 1, columns: 2, pattern: "independent" },
      xaxis: {
        title: { text: "Reward component", font: { size: 13, color: "#c5d0db" } },
        tickangle: -35,
        tickfont: { size: 11 },
        gridcolor: "#2a3441",
      },
      yaxis: {
        title: { text: "Accumulated reward", font: { size: 13, color: "#c5d0db" } },
        zerolinecolor: "#6b7280",
        gridcolor: "#2a3441",
      },
      xaxis2: {
        title: { text: "Reward component", font: { size: 13, color: "#c5d0db" } },
        tickangle: -35,
        tickfont: { size: 11 },
        gridcolor: "#2a3441",
      },
      yaxis2: {
        title: { text: "Accumulated reward", font: { size: 13, color: "#c5d0db" } },
        zerolinecolor: "#6b7280",
        gridcolor: "#2a3441",
      },
      annotations: [
        ...subplotHeading(
          "Total accumulated",
          "x domain",
          {
            subtitle: `Through episode ${episode.episode_index} · ${totalSteps.toLocaleString()} actor steps`,
            y: 1.18,
          },
        ),
        ...subplotHeading(
          "This episode",
          "x2 domain",
          {
            subtitle: episodeSteps > 0
              ? `${episodeSteps.toLocaleString()} steps in episode ${episode.episode_index}`
              : "No completed episode data",
            y: 1.18,
          },
        ),
      ],
    },
    { responsive: true, displayModeBar: true },
  );
}

function episodesUrl() {
  const modelName = window.TrainingPlot?.getSelectedModel?.();
  if (!modelName) {
    throw new Error("No model selected");
  }
  const params = new URLSearchParams({ model: modelName });
  return `${EPISODES_API}?${params.toString()}`;
}

async function loadEpisodes({ preserveIndex = true } = {}) {
  const modelName = window.TrainingPlot?.getSelectedModel?.();
  const metaLine = document.getElementById("reward-meta-line");

  if (!modelName) {
    rewardState.payload = episodesPayloadFromCsv("");
    if (metaLine) metaLine.textContent = "Select a model to view reward components.";
    updateSliderUi();
    await renderPlot();
    return;
  }

  try {
    const response = await fetch(episodesUrl(), { cache: "no-store" });
    if (!response.ok) {
      if (response.status === 404) {
        rewardState.payload = episodesPayloadFromCsv("");
        if (metaLine) metaLine.textContent = `${modelName}/episodes.csv · waiting for data`;
        updateSliderUi();
        await renderPlot();
        return;
      }
      throw new Error(`Failed to load episodes.csv: HTTP ${response.status}`);
    }

    const text = await response.text();
    const lastModified = response.headers.get("Last-Modified");
    const csvMtime = lastModified ? Date.parse(lastModified) : Date.now();
    const payload = episodesPayloadFromCsv(text, csvMtime);
    const fetchKey = payloadFetchKey(payload);
    const prevKey = payloadFetchKey(rewardState.payload);
    const prevEpisodeIndex = rewardState.payload?.episodes?.[rewardState.episodeIndex]?.episode_index ?? null;

    rewardState.payload = payload;

    const episodes = payload.episodes || [];
    if (episodes.length === 0) {
      rewardState.episodeIndex = 0;
      rewardState.playPosition = 0;
    } else if (rewardState.playing) {
      const maxIndex = episodes.length - 1;
      rewardState.playPosition = Math.max(0, Math.min(maxIndex, rewardState.playPosition));
      rewardState.episodeIndex = Math.round(rewardState.playPosition);
    } else if (!preserveIndex || fetchKey !== prevKey) {
      let nextIndex = episodes.length - 1;
      if (preserveIndex && prevEpisodeIndex !== null) {
        const matched = episodes.findIndex((item) => item?.episode_index === prevEpisodeIndex);
        if (matched >= 0) {
          nextIndex = matched;
        }
      } else if (preserveIndex && rewardState.episodeIndex < episodes.length) {
        nextIndex = rewardState.episodeIndex;
      }
      rewardState.episodeIndex = nextIndex;
    }

    if (metaLine) {
      metaLine.textContent = `${modelName}/episodes.csv · ${episodes.length} episode(s)`;
    }

    updateSliderUi();
    if (!rewardState.scrubbing) {
      await renderPlot();
    }
  } catch (error) {
    setRewardStatus(error.message, true);
  }
}

function stopRewardPolling() {
  if (rewardState.pollId !== null) {
    window.clearInterval(rewardState.pollId);
    rewardState.pollId = null;
  }
}

function startRewardPolling() {
  stopRewardPolling();
  rewardState.pollId = window.setInterval(() => {
    void loadEpisodes({ preserveIndex: true }).catch((error) => {
      setRewardStatus(error.message, true);
    });
  }, POLL_INTERVAL_MS);
}

function readPlayHz() {
  const input = document.getElementById("episode-play-hz");
  const value = Number.parseFloat(input?.value ?? "2");
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error("Episode playback speed must be a positive number (episodes/s).");
  }
  return value;
}

function readSliderIndex(slider = document.getElementById("episode-slider")) {
  const maxIndex = Number.parseInt(slider?.max, 10) || 0;
  const raw = Number.parseFloat(slider?.value ?? "0");
  const index = Number.isFinite(raw) ? Math.round(raw) : 0;
  return Math.max(0, Math.min(maxIndex, index));
}

function updatePlayButton() {
  const button = document.getElementById("episode-play");
  if (!button) return;
  button.textContent = rewardState.playing ? "Pause" : "Play";
  button.classList.toggle("playing", rewardState.playing);
  button.setAttribute("aria-pressed", rewardState.playing ? "true" : "false");
}

function stopPlay({ preservePosition = false } = {}) {
  rewardState.playing = false;
  if (rewardState.playRafId !== null) {
    window.cancelAnimationFrame(rewardState.playRafId);
    rewardState.playRafId = null;
  }
  rewardState.playLastTs = null;
  const slider = document.getElementById("episode-slider");
  if (slider) {
    slider.step = "1";
    if (preservePosition) {
      const index = readSliderIndex(slider);
      rewardState.episodeIndex = index;
      rewardState.playPosition = index;
      slider.value = String(index);
    } else {
      const snapped = Math.max(
        0,
        Math.min(Number.parseInt(slider.max, 10) || 0, Math.round(rewardState.playPosition)),
      );
      rewardState.episodeIndex = snapped;
      rewardState.playPosition = snapped;
      slider.value = String(snapped);
      updateSliderUi();
    }
  }
  updatePlayButton();
}

async function setEpisodeIndex(index, { render = true } = {}) {
  const slider = document.getElementById("episode-slider");
  const episodes = rewardState.payload?.episodes || [];
  const maxIndex = Math.max(0, episodes.length - 1);
  const clamped = Math.max(0, Math.min(maxIndex, index));
  rewardState.episodeIndex = clamped;
  rewardState.playPosition = clamped;
  if (slider) {
    slider.value = String(clamped);
  }
  updateSliderUi();
  if (render && clamped !== rewardState.renderedIndex) {
    rewardState.renderedIndex = clamped;
    await renderPlot();
  }
}

function playFrame(timestamp) {
  if (!rewardState.playing || rewardState.scrubbing) {
    if (rewardState.playing && rewardState.scrubbing) {
      rewardState.playRafId = window.requestAnimationFrame(playFrame);
    }
    return;
  }

  const episodes = rewardState.payload?.episodes || [];
  const maxIndex = Math.max(0, episodes.length - 1);
  if (maxIndex <= 0) {
    stopPlay();
    return;
  }

  if (rewardState.playLastTs === null) {
    rewardState.playLastTs = timestamp;
  }
  const dt = Math.max(0, (timestamp - rewardState.playLastTs) / 1000);
  rewardState.playLastTs = timestamp;

  let nextPos = rewardState.playPosition + dt * readPlayHz();
  const crossedEnd = rewardState.playPosition < maxIndex && nextPos >= maxIndex;
  if (nextPos >= maxIndex) {
    nextPos = maxIndex;
  }
  rewardState.playPosition = nextPos;

  const slider = document.getElementById("episode-slider");
  if (slider) {
    slider.step = "any";
    slider.value = String(nextPos);
  }

  const output = document.getElementById("episode-slider-value");
  const previewIndex = Math.min(maxIndex, Math.max(0, Math.round(nextPos)));
  rewardState.episodeIndex = previewIndex;
  if (output) {
    output.textContent = formatEpisodeLabel(episodes[previewIndex]);
  }

  const roundedIndex = Math.round(nextPos);
  if (roundedIndex !== rewardState.renderedIndex && roundedIndex >= 0 && roundedIndex <= maxIndex) {
    rewardState.renderedIndex = roundedIndex;
    void renderPlot();
  }

  if (crossedEnd) {
    stopPlay();
    setRewardStatus(`${formatEpisodeLabel(episodes[maxIndex])} (end)`);
    return;
  }

  rewardState.playRafId = window.requestAnimationFrame(playFrame);
}

function startPlay() {
  const episodes = rewardState.payload?.episodes || [];
  if (episodes.length === 0) {
    return;
  }
  readPlayHz();
  const maxIndex = Math.max(0, episodes.length - 1);
  if (rewardState.episodeIndex >= maxIndex) {
    setRewardStatus(`${formatEpisodeLabel(episodes[maxIndex])} (end)`);
    return;
  }
  rewardState.playing = true;
  rewardState.playLastTs = null;
  rewardState.playPosition = rewardState.episodeIndex;
  rewardState.renderedIndex = rewardState.episodeIndex;
  updatePlayButton();
  if (rewardState.playRafId !== null) {
    window.cancelAnimationFrame(rewardState.playRafId);
  }
  rewardState.playRafId = window.requestAnimationFrame(playFrame);
}

function togglePlay() {
  if (rewardState.playing) {
    stopPlay();
    setRewardStatus(formatEpisodeLabel(rewardState.payload?.episodes?.[rewardState.episodeIndex]));
    return;
  }
  try {
    startPlay();
    setRewardStatus(`Playing episodes at ${readPlayHz()} ep/s`);
  } catch (error) {
    stopPlay();
    setRewardStatus(error.message, true);
  }
}

function bindRewardUi() {
  const slider = document.getElementById("episode-slider");
  slider?.addEventListener("pointerdown", () => {
    rewardState.scrubbing = true;
    stopPlay({ preservePosition: true });
  });
  slider?.addEventListener("pointerup", () => {
    rewardState.scrubbing = false;
  });
  slider?.addEventListener("pointercancel", () => {
    rewardState.scrubbing = false;
  });
  slider?.addEventListener("input", () => {
    const index = readSliderIndex(slider);
    stopPlay({ preservePosition: true });
    rewardState.episodeIndex = index;
    rewardState.playPosition = index;
    rewardState.renderedIndex = index;
    slider.step = "1";
    slider.value = String(index);
    const output = document.getElementById("episode-slider-value");
    const episode = rewardState.payload?.episodes?.[index];
    if (output) {
      output.textContent = formatEpisodeLabel(episode);
    }
    void renderPlot();
  });
  slider?.addEventListener("change", async () => {
    try {
      rewardState.scrubbing = false;
      const index = readSliderIndex(slider);
      stopPlay({ preservePosition: true });
      await setEpisodeIndex(index);
      setRewardStatus(formatEpisodeLabel(rewardState.payload?.episodes?.[rewardState.episodeIndex]));
    } catch (error) {
      setRewardStatus(error.message, true);
    }
  });

  document.getElementById("episode-play")?.addEventListener("click", () => {
    togglePlay();
  });
}

function onModelChanged() {
  stopPlay();
  rewardState.episodeIndex = 0;
  rewardState.playPosition = 0;
  rewardState.renderedIndex = -1;
  if (rewardState.active) {
    void loadEpisodes({ preserveIndex: false }).then(() => {
      setRewardStatus(`Ready · ${formatEpisodeLabel(rewardState.payload?.episodes?.[rewardState.episodeIndex])}`);
    });
  }
}

function startRewardTab() {
  rewardState.active = true;
  void loadEpisodes({ preserveIndex: false }).then(() => {
    setRewardStatus(`Ready · ${formatEpisodeLabel(rewardState.payload?.episodes?.[rewardState.episodeIndex])}`);
  });
  startRewardPolling();
}

function stopRewardTab() {
  rewardState.active = false;
  stopPlay();
  stopRewardPolling();
}

function onTabChanged(tabName) {
  if (tabName === "reward") {
    startRewardTab();
  } else {
    stopRewardTab();
  }
}

bindRewardUi();
updatePlayButton();
window.TrainingPlot?.onModelChange(onModelChanged);
window.TrainingPlot?.onTabChange(onTabChanged);

if (window.TrainingPlot?.getActiveTab?.() === "reward") {
  startRewardTab();
}
