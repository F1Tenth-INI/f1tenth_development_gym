const MODELS_API = "/api/models";

const appState = {
  models: [],
  selectedModel: null,
  activeTab: "metrics",
  modelListeners: new Set(),
  tabListeners: new Set(),
};

function modelsFetchKey(models) {
  return (models || [])
    .map((model) => `${model.name}:${model.mtime}:${model.has_episodes}:${model.has_metrics}`)
    .join("|");
}

function setGlobalStatus(message, isError = false) {
  const el = document.getElementById("global-status");
  if (!el) return;
  el.textContent = message;
  el.classList.toggle("error", isError);
}

function updateModelSelectUi() {
  const select = document.getElementById("model-select");
  if (!select) return;

  const previous = appState.selectedModel;
  select.replaceChildren();
  for (const model of appState.models) {
    const option = document.createElement("option");
    option.value = model.name;
    const suffix = [];
    if (!model.has_metrics) suffix.push("no metrics");
    if (!model.has_episodes) suffix.push("no episodes");
    option.textContent = suffix.length
      ? `${model.name} (${suffix.join(", ")})`
      : model.name;
    select.appendChild(option);
  }

  if (appState.models.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No models found";
    select.appendChild(option);
    select.disabled = true;
    appState.selectedModel = null;
    return;
  }

  select.disabled = false;
  const hasPrevious = appState.models.some((model) => model.name === previous);
  appState.selectedModel = hasPrevious ? previous : appState.models[0].name;
  select.value = appState.selectedModel;
}

async function loadModels({ preserveSelection = false } = {}) {
  const response = await fetch(MODELS_API, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load models: HTTP ${response.status}`);
  }
  const payload = await response.json();
  const nextModels = Array.isArray(payload?.models) ? payload.models : [];
  const prevKey = modelsFetchKey(appState.models);
  const nextKey = modelsFetchKey(nextModels);
  const previous = appState.selectedModel;

  appState.models = nextModels;
  if (!appState.selectedModel && payload?.default) {
    appState.selectedModel = payload.default;
  } else if (
    preserveSelection
    && previous
    && nextModels.some((model) => model.name === previous)
  ) {
    appState.selectedModel = previous;
  } else if (!nextModels.some((model) => model.name === stateSelectedModel())) {
    appState.selectedModel = payload?.default || nextModels[0]?.name || null;
  }

  updateModelSelectUi();
  return {
    changed: prevKey !== nextKey,
    selectionChanged: previous !== appState.selectedModel,
  };
}

function stateSelectedModel() {
  return appState.selectedModel;
}

function notifyModelChange() {
  for (const listener of appState.modelListeners) {
    listener(appState.selectedModel);
  }
}

function notifyTabChange() {
  for (const listener of appState.tabListeners) {
    listener(appState.activeTab);
  }
}

function setActiveTab(tabName) {
  if (appState.activeTab === tabName) return;
  appState.activeTab = tabName;

  document.querySelectorAll(".tab").forEach((button) => {
    const isActive = button.dataset.tab === tabName;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });

  document.querySelectorAll(".tab-panel").forEach((panel) => {
    const isActive = panel.id === `tab-${tabName}`;
    panel.classList.toggle("active", isActive);
    panel.hidden = !isActive;
  });

  notifyTabChange();
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      setActiveTab(button.dataset.tab);
    });
  });

  const params = new URLSearchParams(window.location.search);
  const requestedTab = params.get("tab");
  if (requestedTab === "reward" || requestedTab === "metrics") {
    setActiveTab(requestedTab);
  }
}

function bindModelSelect() {
  document.getElementById("model-select")?.addEventListener("change", async (event) => {
    const nextModel = event.target.value;
    if (!nextModel || nextModel === appState.selectedModel) return;
    appState.selectedModel = nextModel;
    setGlobalStatus(`Switched to ${nextModel}`);
    notifyModelChange();
  });
}

window.TrainingPlot = {
  getSelectedModel: () => appState.selectedModel,
  getActiveTab: () => appState.activeTab,
  onModelChange(listener) {
    appState.modelListeners.add(listener);
    return () => appState.modelListeners.delete(listener);
  },
  onTabChange(listener) {
    appState.tabListeners.add(listener);
    return () => appState.tabListeners.delete(listener);
  },
  refreshModels: () => loadModels({ preserveSelection: true }),
};

async function initMain() {
  bindTabs();
  bindModelSelect();
  setGlobalStatus("Loading models…");
  try {
    await loadModels();
    setGlobalStatus(appState.selectedModel ? `Model: ${appState.selectedModel}` : "No models");
    notifyModelChange();
    notifyTabChange();
    window.setInterval(() => {
      void loadModels({ preserveSelection: true }).then(({ changed, selectionChanged }) => {
        if (selectionChanged) {
          setGlobalStatus(`Switched to ${appState.selectedModel}`);
          notifyModelChange();
        } else if (changed) {
          setGlobalStatus(`Model list updated · ${appState.selectedModel || "—"}`);
        }
      }).catch((error) => {
        setGlobalStatus(error.message, true);
      });
    }, 10000);
  } catch (error) {
    if (window.location.protocol === "file:") {
      setGlobalStatus(
        "Cannot load data via file://. Run: python TrainingLite/rl_racing/TrainingPlot/run_training_plot.py",
        true,
      );
    } else {
      setGlobalStatus(error.message, true);
    }
  }
}

initMain();
