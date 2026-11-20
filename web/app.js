"use strict";

const SUBTYPES = ["A1", "A2A", "A2B", "A3"];
const SUBTYPE_PAIRS = [
  ["A1", "A2A"],
  ["A1", "A2B"],
  ["A1", "A3"],
  ["A2A", "A2B"],
  ["A2A", "A3"],
  ["A2B", "A3"],
];

const PLOTLY_DARK_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "#162032",
  font: { color: "#cbd5e1", family: "Inter, Segoe UI, system-ui, sans-serif", size: 12 },
  margin: { l: 55, r: 24, t: 30, b: 48 },
};

const state = {
  presets: {},
  receptors: {},
  catalog: [],
  currentPrediction: null,
  conformerSdf: null,
  conformerSmiles: null,
  mol3dViewer: null,
  mol3dStyle: "ballstick",
  gpcrViewer: null,
  gpcrState: "active",
  gpcrLoadedKey: null,
  provenanceLoaded: false,
  benchmarksLoaded: false,
  benchmarksData: null,
  batchData: null,
  batchSort: { key: "max_pki", dir: "desc" },
};


function escapeHtml(val) {
  if (val === null || val === undefined) return "";
  return String(val)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtNum(val, digits = 2, fallback = "N/A") {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return fallback;
  return Number(val).toFixed(digits);
}

function fmtPct(val, digits = 1, fallback = "N/A") {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return fallback;
  return `${(Number(val) * 100).toFixed(digits)}%`;
}

function metricPick(obj, primary, legacy) {
  if (!obj) return null;
  if (obj[primary] !== undefined && obj[primary] !== null) return obj[primary];
  if (legacy && obj[legacy] !== undefined && obj[legacy] !== null) return obj[legacy];
  return null;
}

function setBanner(el, msg, type = "info") {
  if (!el) return;
  if (!msg) {
    el.className = "status-banner hidden";
    el.textContent = "";
    return;
  }
  el.className = `status-banner ${type}`;
  el.textContent = msg;
}

function badgeClassFromText(text) {
  const s = String(text || "").toUpperCase();
  if (s.includes("HIGH") || s.includes("OUTSIDE") || s.includes("UNFAVORABLE") || s.includes("FAIL")) {
    return "badge-red";
  }
  if (s.includes("MODERATE") || s.includes("BORDERLINE") || s.includes("MONITORED") || s.includes("UNCLASSIFIED")) {
    return "badge-amber";
  }
  if (s.includes("LOW") || s.includes("INSIDE") || s.includes("FAVORABLE") || s.includes("PASS") || s.includes("AGONIST") || s.includes("CLEAN")) {
    return "badge-green";
  }
  return "badge-cyan";
}

function downloadBlob(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function renderPlotlyOrFallback(containerId, traces, layout, config = {}) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (typeof Plotly === "undefined") {
    el.innerHTML = `<div class="chart-fallback">Plotly.js CDN unavailable offline. Tabular metrics remain available below.</div>`;
    return;
  }
  el.innerHTML = "";
  const mergedLayout = Object.assign({}, PLOTLY_DARK_LAYOUT, layout);
  const mergedConfig = Object.assign({ responsive: true, displayModeBar: true, displaylogo: false }, config);
  Plotly.newPlot(el, traces, mergedLayout, mergedConfig);
}

async function apiJson(url, options = {}) {
  const resp = await fetch(url, options);
  const text = await resp.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!resp.ok) {
    const detail = (data && (data.detail || data.message)) || `HTTP ${resp.status}`;
    throw new Error(detail);
  }
  return data;
}


document.addEventListener("DOMContentLoaded", () => {
  bindWorkspaceNavigation();
  bindSingleWorkspaceEvents();
  bindBatchWorkspaceEvents();
  bindBenchmarkFilters();
  initPlatform();
});

async function initPlatform() {
  const statusPill = document.getElementById("api-status-pill");
  const versionTag = document.getElementById("app-version-tag");

  try {
    const [health, presetPayload] = await Promise.all([
      apiJson("/api/health"),
      apiJson("/api/presets"),
    ]);

    statusPill.textContent = "API Online";
    statusPill.classList.add("ok");
    if (health && health.version) {
      versionTag.textContent = `v${health.version}`;
    }

    state.presets = presetPayload.presets || {};
    state.receptors = presetPayload.receptors || {};
    state.catalog = presetPayload.catalog || [];

    populatePresetsDropdown();
    renderGpcrCatalogTable();

    const presetKeys = Object.keys(state.presets);
    if (presetKeys.length > 0) {
      const firstKey = presetKeys[0];
      const select = document.getElementById("preset-select");
      const input = document.getElementById("smiles-input");
      select.value = firstKey;
      input.value = state.presets[firstKey].smiles;
      runSinglePrediction();
    }
  } catch (err) {
    statusPill.textContent = "API Offline";
    statusPill.classList.add("err");
    setBanner(document.getElementById("single-status-bar"), `Backend connection failed: ${err.message}`, "error");
  }
}

function populatePresetsDropdown() {
  const select = document.getElementById("preset-select");
  select.innerHTML = `<option value="__custom__">Custom SMILES Input</option>`;
  for (const [key, entry] of Object.entries(state.presets)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = entry.name || key;
    select.appendChild(opt);
  }
}


function bindWorkspaceNavigation() {
  const wsButtons = document.querySelectorAll(".ws-tab");
  wsButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.workspace;
      switchWorkspace(target);
    });
  });

  const subButtons = document.querySelectorAll(".subtab-btn");
  subButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.subtab;
      switchSubtab(target);
    });
  });
}

function switchWorkspace(name) {
  document.querySelectorAll(".ws-tab").forEach((b) => {
    const active = b.dataset.workspace === name;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".workspace-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `ws-${name}`);
  });

  if (name === "benchmarks" && !state.benchmarksLoaded) {
    loadBenchmarks();
  }
}

function switchSubtab(name) {
  document.querySelectorAll(".subtab-btn").forEach((b) => {
    const active = b.dataset.subtab === name;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".subtab-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `subtab-${name}`);
  });

  if (name === "structure" && state.currentPrediction) {
    load3DConformerOnDemand();
  } else if (name === "selectivity" && state.currentPrediction) {
    renderSelectivityRadar(state.currentPrediction);
  } else if (name === "shap" && state.currentPrediction) {
    const st = document.getElementById("shap-subtype-select").value || "A2A";
    loadChemicalSpaceAndShap(state.currentPrediction.smiles, st);
  } else if (name === "gpcrdb") {
    loadGpcrStructure();
  } else if (name === "provenance" && !state.provenanceLoaded) {
    loadProvenance();
  }
}


function bindSingleWorkspaceEvents() {
  const presetSelect = document.getElementById("preset-select");
  const smilesInput = document.getElementById("smiles-input");
  const thresholdSlider = document.getElementById("threshold-slider");
  const thresholdValue = document.getElementById("threshold-value");
  const form = document.getElementById("single-predict-form");

  presetSelect.addEventListener("change", () => {
    const val = presetSelect.value;
    if (val !== "__custom__" && state.presets[val]) {
      smilesInput.value = state.presets[val].smiles;
    }
  });

  smilesInput.addEventListener("input", () => {
    const cur = smilesInput.value.trim();
    const matchedKey = Object.keys(state.presets).find((k) => state.presets[k].smiles === cur);
    presetSelect.value = matchedKey || "__custom__";
  });

  thresholdSlider.addEventListener("input", () => {
    thresholdValue.textContent = Number(thresholdSlider.value).toFixed(1);
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    runSinglePrediction();
  });

  document.getElementById("btn-export-json").addEventListener("click", () => {
    if (!state.currentPrediction) return;
    downloadBlob(
      JSON.stringify(state.currentPrediction, null, 2),
      "adenosine_profile_report.json",
      "application/json"
    );
  });

  document.getElementById("btn-load-3d").addEventListener("click", () => load3DConformerOnDemand(true));
  document.getElementById("btn-dl-sdf").addEventListener("click", () => export3DFile("sdf"));
  document.getElementById("btn-dl-pdb").addEventListener("click", () => export3DFile("pdb"));

  document.querySelectorAll("[data-3d-style]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mol3dStyle = btn.dataset["3dStyle"];
      applyMol3dStyle();
    });
  });

  document.getElementById("shap-subtype-select").addEventListener("change", (e) => {
    if (state.currentPrediction) {
      loadChemicalSpaceAndShap(state.currentPrediction.smiles, e.target.value);
    }
  });

  document.getElementById("gpcr-subtype-select").addEventListener("change", () => {
    state.gpcrLoadedKey = null;
    loadGpcrStructure();
  });

  document.querySelectorAll("[data-gpcr-state]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.gpcrState = btn.dataset.gpcrState;
      document.querySelectorAll("[data-gpcr-state]").forEach((b) => {
        b.classList.toggle("active", b === btn);
      });
      state.gpcrLoadedKey = null;
      loadGpcrStructure();
    });
  });
}

async function runSinglePrediction() {
  const smiles = document.getElementById("smiles-input").value.trim();
  const threshold = Number(document.getElementById("threshold-slider").value);
  const statusBar = document.getElementById("single-status-bar");
  const btn = document.getElementById("btn-profile-molecule");

  if (!smiles) {
    setBanner(statusBar, "Please enter a valid SMILES string or choose a reference ligand preset.", "error");
    return;
  }

  btn.disabled = true;
  btn.querySelector(".btn-label").textContent = "Profiling...";
  setBanner(statusBar, "Running conformal multi-target pKi ensemble (XGBoost, LightGBM, Random Forest)...", "loading");

  try {
    const data = await apiJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles, threshold }),
    });

    state.currentPrediction = data;
    state.conformerSdf = null;
    state.conformerSmiles = null;

    setBanner(statusBar, null);
    document.getElementById("single-results-shell").classList.remove("hidden");

    renderDatabaseLookupBanner(data);
    renderOverviewTab(data);
    renderStructureTab(data);
    renderSelectivityTab(data);
    renderEfficacySafetyAdmetTab(data);

    const bestTarget = SUBTYPES.includes(data.best_target) ? data.best_target : "A2A";
    document.getElementById("shap-subtype-select").value = bestTarget;
    renderNeighborsCards(data.neighbors || []);

    const activeSub = document.querySelector(".subtab-btn.active")?.dataset.subtab;
    if (activeSub === "structure") {
      load3DConformerOnDemand();
    } else if (activeSub === "selectivity") {
      renderSelectivityRadar(data);
    } else if (activeSub === "shap") {
      loadChemicalSpaceAndShap(data.smiles, bestTarget);
    }
  } catch (err) {
    setBanner(statusBar, `Prediction failed: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.querySelector(".btn-label").textContent = "Profile Molecule";
  }
}

function renderDatabaseLookupBanner(data) {
  const banner = document.getElementById("db-lookup-banner");
  const preds = data.predictions || {};
  const cons = preds.Consensus || {};
  const xgb = preds.XGBoost || {};
  const lgb = preds.LightGBM || {};
  const rf = preds.RandomForest || {};
  const dbVal = data.db_value || {};
  const inDb = Boolean(data.in_database);
  const splitLabel = data.db_status === "training" ? "Training Set" : data.db_status === "test" ? "Holdout Test Set" : "De Novo";

  banner.classList.remove("hidden", "known", "de-novo");
  banner.classList.add(inDb ? "known" : "de-novo");

  const headerHtml = inDb
    ? `<div class="db-banner-header">
         <div class="db-banner-title">Known Experimental pKi (Curated Ki/Kd Dataset &middot; ${escapeHtml(splitLabel)})</div>
         <span class="badge badge-green">In Curated Dataset (${escapeHtml(splitLabel)})</span>
       </div>`
    : `<div class="db-banner-header">
         <div class="db-banner-title">De Novo Candidate (Not in Curated Ki/Kd Dataset)</div>
         <span class="badge badge-cyan">De Novo Ensemble Inference</span>
       </div>`;

  const chipsHtml = SUBTYPES.map((st) => {
    const expVal = inDb && dbVal[st] !== null && dbVal[st] !== undefined ? fmtNum(dbVal[st], 2) : null;
    const expLine = inDb
      ? `<div class="db-chip-exp">${expVal !== null ? `Exp. pKi: ${expVal}` : "Exp. pKi: Not assayed"}</div>`
      : `<div class="db-chip-exp" style="color: var(--accent);">Consensus: ${fmtNum(cons[st], 2)}</div>`;

    const rfPart = rf[st] !== undefined && rf[st] !== null ? ` | RF: ${fmtNum(rf[st], 2)}` : "";
    const predsLine = inDb
      ? `Ens: ${fmtNum(cons[st], 2)} | XGB: ${fmtNum(xgb[st], 2)} | LGB: ${fmtNum(lgb[st], 2)}${rfPart}`
      : `XGB: ${fmtNum(xgb[st], 2)} | LGB: ${fmtNum(lgb[st], 2)}${rfPart}`;

    return `<div class="db-chip">
      <div class="db-chip-label">Human ${st}</div>
      ${expLine}
      <div class="db-chip-preds">${predsLine}</div>
    </div>`;
  }).join("");

  banner.innerHTML = `${headerHtml}<div class="db-banner-grid">${chipsHtml}</div>`;
}


function renderOverviewTab(data) {
  const grid = document.getElementById("subtype-cards-grid");
  const cons = (data.predictions && data.predictions.Consensus) || {};
  const ivs = (data.intervals && data.intervals.Consensus) || {};
  const ki = data.ki_values || {};
  const hits = new Set(data.target_hits || []);
  const dbVal = data.db_value || {};
  const inDb = Boolean(data.in_database);

  grid.innerHTML = SUBTYPES.map((st) => {
    const isBest = data.best_target === st;
    const isHit = hits.has(st);
    const iv = ivs[st] || {};
    const kiInfo = ki[st] || {};
    const badge = isHit
      ? `<span class="badge badge-green">${isBest ? "Primary Hit" : "Hit"}</span>`
      : `<span class="badge">Below Cutoff</span>`;

    const expHtml =
      inDb && dbVal[st] !== null && dbVal[st] !== undefined
        ? `<div class="subtype-exp-line">Experimental pKi: ${fmtNum(dbVal[st], 2)}</div>`
        : "";

    return `<div class="subtype-card ${isBest ? "primary-hit" : ""}">
      <div class="subtype-card-top">
        <span class="subtype-name">Human ${st}</span>
        ${badge}
      </div>
      <div class="subtype-pki">${fmtNum(cons[st], 2)}</div>
      <div class="subtype-ki">Ki: ${escapeHtml(kiInfo.display || "N/A")}</div>
      <div class="subtype-ci">90% MAPIE CI: [${fmtNum(iv.lower, 2)}, ${fmtNum(iv.upper, 2)}] (w=${fmtNum(iv.width, 2)})</div>
      ${expHtml}
    </div>`;
  }).join("");

  const ad = data.applicability_domain || {};
  const adBadge = document.getElementById("ad-status-badge");
  adBadge.textContent = ad.status || ad.domain_status || "Unknown";
  adBadge.className = `badge ${badgeClassFromText(ad.status)}`;

  const violations = ad.property_violations || ad.violations || [];
  document.getElementById("ad-card-body").innerHTML = `
    <div class="kv-list">
      <div class="kv-row">
        <span class="kv-key">Max Training Tanimoto</span>
        <span class="kv-val mono">${fmtNum(ad.tanimoto_max, 3)}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Reference Fingerprint Set</span>
        <span class="kv-val">${escapeHtml(ad.reference_source || "train_fps.pkl")}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">In Applicability Domain</span>
        <span class="kv-val">${escapeHtml(ad.in_domain_label || (ad.in_domain ? "Yes" : "No (Extrapolation Warning)"))}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Property Gate Violations</span>
        <span class="kv-val">${violations.length ? escapeHtml(violations.join("; ")) : "None"}</span>
      </div>
    </div>`;

  const desc = data.descriptors || {};
  document.getElementById("identity-card-body").innerHTML = `
    <div class="kv-list">
      <div class="kv-row">
        <span class="kv-key">Canonical SMILES</span>
        <span class="kv-val mono">${escapeHtml(data.smiles)}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Primary Predicted Target</span>
        <span class="kv-val">${data.best_target && SUBTYPES.includes(data.best_target) ? `Human ${escapeHtml(data.best_target)}` : escapeHtml(data.best_target || "None (Inactive across all subtypes)")}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Hits (&ge; ${fmtNum(data.threshold, 1)} pKi)</span>
        <span class="kv-val">${(data.target_hits && data.target_hits.length) ? escapeHtml(data.target_hits.join(", ")) : "None"}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">MW / LogP / TPSA</span>
        <span class="kv-val mono">${fmtNum(desc.MW, 1)} Da / ${fmtNum(desc.LogP, 2)} / ${fmtNum(desc.TPSA, 1)} A^2</span>
      </div>
    </div>`;

  const tbody = document.querySelector("#model-comparison-table tbody");
  const modelRows = [
    { label: "Ensemble Mean (Consensus)", key: "Consensus", highlight: true },
    { label: "XGBoost (Conformal)", key: "XGBoost" },
    { label: "LightGBM (Conformal)", key: "LightGBM" },
    { label: "Random Forest (Conformal)", key: "RandomForest" },
  ];

  let rowsHtml = modelRows
    .map((m) => {
      const mPreds = (data.predictions && data.predictions[m.key]) || null;
      const mIvs = (data.intervals && data.intervals[m.key]) || {};
      const cells = SUBTYPES.map((st) => {
        if (!mPreds || mPreds[st] === undefined || mPreds[st] === null) {
          return `<td class="mono muted-note">N/A</td>`;
        }
        const iv = mIvs[st];
        const ciStr = iv ? ` <span class="muted-note">[${fmtNum(iv.lower, 2)}, ${fmtNum(iv.upper, 2)}]</span>` : "";
        return `<td class="mono">${m.highlight ? `<strong>${fmtNum(mPreds[st], 3)}</strong>` : fmtNum(mPreds[st], 3)}${ciStr}</td>`;
      }).join("");
      return `<tr><td>${m.highlight ? `<strong>${m.label}</strong>` : m.label}</td>${cells}</tr>`;
    })
    .join("");

  if (inDb) {
    const expCells = SUBTYPES.map((st) => {
      const v = dbVal[st];
      return `<td class="mono" style="color: var(--green); font-weight: 700;">${v !== null && v !== undefined ? fmtNum(v, 2) : "--"}</td>`;
    }).join("");
    rowsHtml = `<tr><td><strong style="color: var(--green);">Known Experimental pKi (${escapeHtml(data.db_status)})</strong></td>${expCells}</tr>` + rowsHtml;
  }

  tbody.innerHTML = rowsHtml;

  const funcTbody = document.querySelector("#functional-comparison-table tbody");
  if (funcTbody) {
    const fa = data.functional_activity || {};
    const fPredsMap = fa.predictions || {};
    const fIvsMap = fa.intervals || {};
    const fDbVal = fa.db_value || {};
    const fInDb = Boolean(fa.in_database);

    let funcRowsHtml = modelRows
      .map((m) => {
        const mPreds = fPredsMap[m.key] || null;
        const mIvs = fIvsMap[m.key] || {};
        const cells = SUBTYPES.map((st) => {
          if (!mPreds || mPreds[st] === undefined || mPreds[st] === null) {
            return `<td class="mono muted-note">N/A</td>`;
          }
          const iv = mIvs[st];
          const ciStr = iv ? ` <span class="muted-note">[${fmtNum(iv.lower, 2)}, ${fmtNum(iv.upper, 2)}]</span>` : "";
          return `<td class="mono">${m.highlight ? `<strong>${fmtNum(mPreds[st], 3)}</strong>` : fmtNum(mPreds[st], 3)}${ciStr}</td>`;
        }).join("");
        return `<tr><td>${m.highlight ? `<strong>${m.label}</strong>` : m.label}</td>${cells}</tr>`;
      })
      .join("");

    if (fInDb) {
      const fExpCells = SUBTYPES.map((st) => {
        const v = fDbVal[st];
        return `<td class="mono" style="color: var(--accent); font-weight: 700;">${v !== null && v !== undefined ? fmtNum(v, 2) : "--"}</td>`;
      }).join("");
      funcRowsHtml = `<tr><td><strong style="color: var(--accent);">Known Experimental pIC50/pEC50 (${escapeHtml(fa.db_status)})</strong></td>${fExpCells}</tr>` + funcRowsHtml;
    }

    funcTbody.innerHTML = funcRowsHtml;
  }
}


function renderStructureTab(data) {
  const svgBox = document.getElementById("mol-2d-container");
  if (data.svg_2d && data.svg_2d.includes("<svg")) {
    svgBox.innerHTML = data.svg_2d;
  } else {
    svgBox.innerHTML = `<div class="muted-note">2D SVG rendering unavailable for this structure.</div>`;
  }

  const desc = data.descriptors || {};
  const strip = document.getElementById("struct-desc-strip");
  const pills = [
    ["MW", `${fmtNum(desc.MW, 1)} Da`],
    ["LogP", fmtNum(desc.LogP, 2)],
    ["TPSA", `${fmtNum(desc.TPSA, 1)} A^2`],
    ["HBD", desc.HBD],
    ["HBA", desc.HBA],
    ["RotBonds", desc.RotBonds],
    ["AromRings", desc.AromRings],
  ];
  strip.innerHTML = pills
    .map(([k, v]) => `<span class="desc-pill">${k}: <strong>${escapeHtml(v ?? "N/A")}</strong></span>`)
    .join("");

  const overlay = document.getElementById("mol-3d-status");
  overlay.classList.remove("hidden");
  overlay.textContent = "Click 'Generate 3D' or wait for ETKDGv3/MMFF94 conformer...";
}

async function load3DConformerOnDemand(force = false) {
  if (!state.currentPrediction) return;
  const smiles = state.currentPrediction.smiles;
  const overlay = document.getElementById("mol-3d-status");

  if (typeof $3Dmol === "undefined") {
    overlay.classList.remove("hidden");
    overlay.textContent = "3Dmol.js CDN unavailable offline. Use Download SDF / PDB buttons to inspect locally.";
    return;
  }

  if (!force && state.conformerSdf && state.conformerSmiles === smiles) {
    renderMol3d(state.conformerSdf);
    return;
  }

  overlay.classList.remove("hidden");
  overlay.textContent = "Generating ETKDGv3 / MMFF94 3D conformer...";

  try {
    const res = await apiJson("/api/export/3d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles, fmt: "sdf" }),
    });
    state.conformerSdf = res.content;
    state.conformerSmiles = smiles;
    renderMol3d(res.content);
  } catch (err) {
    overlay.classList.remove("hidden");
    overlay.textContent = `3D conformer failed: ${err.message}`;
  }
}

function renderMol3d(sdfBlock) {
  const el = document.getElementById("mol-3d-viewer");
  const overlay = document.getElementById("mol-3d-status");
  if (typeof $3Dmol === "undefined") return;

  el.innerHTML = "";
  const viewer = $3Dmol.createViewer(el, { backgroundColor: "#0b1120" });
  viewer.addModel(sdfBlock, "sdf");
  state.mol3dViewer = viewer;
  applyMol3dStyle();
  viewer.zoomTo();
  viewer.render();
  overlay.classList.add("hidden");
}

function applyMol3dStyle() {
  if (!state.mol3dViewer) return;
  const v = state.mol3dViewer;
  if (state.mol3dStyle === "stick") {
    v.setStyle({}, { stick: { radius: 0.16, colorscheme: "Jmol" } });
  } else if (state.mol3dStyle === "sphere") {
    v.setStyle({}, { sphere: { scale: 0.85, colorscheme: "Jmol" } });
  } else {
    v.setStyle({}, { stick: { radius: 0.14, colorscheme: "Jmol" }, sphere: { scale: 0.28, colorscheme: "Jmol" } });
  }
  v.render();
}

async function export3DFile(fmt) {
  if (!state.currentPrediction) return;
  try {
    const res = await apiJson("/api/export/3d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles: state.currentPrediction.smiles, fmt }),
    });
    downloadBlob(res.content, `ligand_conformer.${fmt}`, "chemical/x-mdl-sdfile");
  } catch (err) {
    alert(`Failed to export ${fmt.toUpperCase()}: ${err.message}`);
  }
}


function renderSelectivityTab(data) {
  const spec = data.selectivity_spectrum || {};
  const cons = (data.predictions && data.predictions.Consensus) || {};
  const ivs = (data.intervals && data.intervals.Consensus) || {};
  const ki = data.ki_values || {};

  document.getElementById("selectivity-class-badge").textContent = spec.classification || "N/A";

  const rankBody = document.querySelector("#selectivity-rank-table tbody");
  const ranked = spec.rank_hierarchy || SUBTYPES;
  rankBody.innerHTML = ranked
    .map((st, i) => {
      const iv = ivs[st] || {};
      const kInfo = ki[st] || {};
      return `<tr>
        <td class="mono">#${i + 1}</td>
        <td><strong>Human ${st}</strong></td>
        <td class="mono">${fmtNum(cons[st], 3)}</td>
        <td class="mono">${escapeHtml(kInfo.display || "N/A")}</td>
        <td class="mono">[${fmtNum(iv.lower, 2)}, ${fmtNum(iv.upper, 2)}]</td>
      </tr>`;
    })
    .join("");

  const pairBody = document.querySelector("#pairwise-delta-table tbody");
  const deltas = spec.pairwise_deltas || data.selectivity_profile || {};
  pairBody.innerHTML = SUBTYPE_PAIRS.map(([a, b]) => {
    const key = `${a}_vs_${b}`;
    const d = deltas[key];
    const absD = d !== undefined && d !== null ? Math.abs(Number(d)) : null;
    const fold = absD !== null ? Math.pow(10, absD) : null;
    const favored = d > 0.05 ? a : d < -0.05 ? b : "Equipotent";
    const foldStr = fold !== null ? (fold >= 1000 ? `${fold.toFixed(0)}x` : `${fold.toFixed(1)}x`) : "N/A";
    return `<tr>
      <td><strong>${a} vs ${b}</strong></td>
      <td class="mono">${d !== undefined && d !== null ? (d >= 0 ? `+${fmtNum(d, 2)}` : fmtNum(d, 2)) : "N/A"}</td>
      <td class="mono">${foldStr}</td>
      <td><span class="badge ${favored === "Equipotent" ? "" : "badge-cyan"}">${favored}</span></td>
    </tr>`;
  }).join("");

  renderSelectivityRadar(data);
}

function renderSelectivityRadar(data) {
  const cons = (data.predictions && data.predictions.Consensus) || {};
  const dbVal = data.db_value || {};
  const rVals = SUBTYPES.map((st) => Number(cons[st] || 0));
  const theta = [...SUBTYPES, SUBTYPES[0]];
  const rClosed = [...rVals, rVals[0]];

  const traces = [
    {
      type: "scatterpolar",
      r: rClosed,
      theta,
      fill: "toself",
      fillcolor: "rgba(56, 189, 248, 0.22)",
      line: { color: "#38bdf8", width: 2.5 },
      marker: { color: "#38bdf8", size: 7 },
      name: "Consensus pKi",
    },
  ];

  if (data.in_database && SUBTYPES.some((st) => dbVal[st] !== null && dbVal[st] !== undefined)) {
    const expR = SUBTYPES.map((st) => (dbVal[st] !== null && dbVal[st] !== undefined ? Number(dbVal[st]) : null));
    traces.push({
      type: "scatterpolar",
      r: [...expR, expR[0]],
      theta,
      mode: "lines+markers",
      line: { color: "#4ade80", width: 2, dash: "dot" },
      marker: { color: "#4ade80", size: 8 },
      name: "Experimental pKi",
    });
  }

  renderPlotlyOrFallback("selectivity-radar-chart", traces, {
    margin: { l: 40, r: 40, t: 30, b: 30 },
    polar: {
      bgcolor: "#162032",
      radialaxis: { visible: true, range: [3.5, 10.5], gridcolor: "#334155", tickfont: { color: "#94a3b8" } },
      angularaxis: { gridcolor: "#334155", tickfont: { color: "#f8fafc", size: 13 } },
    },
    legend: { orientation: "h", y: -0.1, x: 0.5, xanchor: "center" },
  });
}


function renderEfficacySafetyAdmetTab(data) {
  const eff = data.functional_efficacy || {};
  const pocket = data.pocket_interactions || {};
  const safety = data.safety_profile || {};
  const cns = data.cns_admet || {};
  const qed = data.qed_profile || {};
  const pains = data.pains_alerts || [];

  const effBadge = document.getElementById("efficacy-class-badge");
  effBadge.textContent = eff.efficacy_class || "Unclassified";
  effBadge.className = `badge ${badgeClassFromText(eff.efficacy_class)}`;

  document.getElementById("efficacy-body").innerHTML = `
    <div class="kv-list">
      <div class="kv-row">
        <span class="kv-key">2D SMARTS MoA Hypothesis</span>
        <span class="kv-val"><strong>${escapeHtml(eff.mode_of_action || "N/A")}</strong></span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Signaling Pathway (Gs / Gi)</span>
        <span class="kv-val">${escapeHtml(eff.signaling_pathway || "N/A")}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Translational Indication</span>
        <span class="kv-val">${escapeHtml(eff.therapeutic_indication || "N/A")}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">${escapeHtml(pocket.asn_residue || "Asn6.55")} H-Bond Motif (2D)</span>
        <span class="kv-val">${pocket.asn_anchor_hbond ? `<span class="badge badge-green">Present</span>` : `<span class="badge">Not Detected</span>`}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">${escapeHtml(pocket.trp_residue || "Trp6.48")} Toggle Motif (2D)</span>
        <span class="kv-val">${pocket.trp_toggle_switch ? `<span class="badge badge-green">Present</span>` : `<span class="badge">Not Detected</span>`}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">ECL2 Phe168/171 Pi-Stacking (2D)</span>
        <span class="kv-val">${pocket.phe_pi_stacking ? `<span class="badge badge-green">Present (&ge;2 Arom Rings)</span>` : `<span class="badge">Low</span>`}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Stereocenters (${escapeHtml(pocket.stereocenter_count ?? 0)})</span>
        <span class="kv-val">${escapeHtml(pocket.chiral_alert || "N/A")}</span>
      </div>
    </div>`;

  const safeBadge = document.getElementById("safety-overall-badge");
  safeBadge.textContent = (safety.overall_safety_index || "Unknown").split("(")[0].trim();
  safeBadge.className = `badge ${badgeClassFromText(safety.overall_safety_index)}`;

  document.getElementById("safety-body").innerHTML = `
    <div class="kv-list">
      <div class="kv-row">
        <span class="kv-key">Overall Safety Index</span>
        <span class="kv-val">${escapeHtml(safety.overall_safety_index || "N/A")}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">A1 Cardiac / AV-Block Risk</span>
        <span class="kv-val"><span class="badge ${badgeClassFromText(safety.a1_bradycardia_risk)}">${escapeHtml(safety.a1_bradycardia_risk || "N/A")}</span></span>
      </div>
      <div class="kv-row">
        <span class="kv-key">A3 Mast-Cell Degranulation</span>
        <span class="kv-val"><span class="badge ${badgeClassFromText(safety.a3_mast_cell_risk)}">${escapeHtml(safety.a3_mast_cell_risk || "N/A")}</span></span>
      </div>
      <div class="kv-row">
        <span class="kv-key">PDE1-10 Off-Target Alert</span>
        <span class="kv-val"><span class="badge ${badgeClassFromText(safety.pde_cross_reactivity)}">${escapeHtml(safety.pde_cross_reactivity || "N/A")}</span></span>
      </div>
    </div>
    <p class="muted-note mt-sm">Thresholds: High Risk &ge; 7.5 pKi, Moderate Risk &ge; 6.0 pKi; PDE alerts screened via xanthine/purine SMARTS.</p>`;

  const cnsBadge = document.getElementById("cns-mpo-badge");
  const mpoScore = Number(cns.cns_mpo_score || 0);
  cnsBadge.textContent = `CNS-MPO: ${fmtNum(mpoScore, 2)} / 6.0`;
  cnsBadge.className = `badge ${mpoScore >= 4.0 ? "badge-green" : mpoScore >= 3.0 ? "badge-amber" : "badge-cyan"}`;

  const mpoComps = cns.mpo_components || {};
  const barsHtml = Object.entries(mpoComps)
    .map(([k, v]) => {
      const pct = Math.max(0, Math.min(100, Number(v) * 100));
      return `<div class="mpo-bar-item">
        <div class="mpo-bar-head"><span>${escapeHtml(k)}</span><span class="mono">${fmtNum(v, 2)}</span></div>
        <div class="mpo-bar-track"><div class="mpo-bar-fill" style="width:${pct}%"></div></div>
      </div>`;
    })
    .join("");

  document.getElementById("cns-admet-body").innerHTML = `
    <div class="kv-list">
      <div class="kv-row">
        <span class="kv-key">CNS Distribution Class</span>
        <span class="kv-val">${escapeHtml(cns.cns_class || "N/A")}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Clark Empirical LogBB</span>
        <span class="kv-val mono">${fmtNum(cns.log_bb, 2)} (${escapeHtml(cns.bbb_status || "N/A")})</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Est. Basic pKa / cLogD7.4</span>
        <span class="kv-val mono">${escapeHtml(cns.est_pka_display || (cns.est_pka !== null && cns.est_pka !== undefined ? fmtNum(cns.est_pka, 1) : "None (Acidic/Neutral)"))} / ${fmtNum(cns.c_logd74, 2)}</span>
      </div>
    </div>
    <div class="mpo-bars">${barsHtml}</div>`;

  const painsBadge = document.getElementById("pains-badge");
  painsBadge.textContent = pains.length === 0 ? "PAINS Clean (0)" : `PAINS Alerts (${pains.length})`;
  painsBadge.className = `badge ${pains.length === 0 ? "badge-green" : "badge-red"}`;

  document.getElementById("druglikeness-body").innerHTML = `
    <div class="kv-list">
      <div class="kv-row">
        <span class="kv-key">QED (Drug-Likeness 0-1)</span>
        <span class="kv-val mono"><strong>${fmtNum(qed.qed, 4)}</strong></span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Lipinski Rule of 5</span>
        <span class="kv-val">
          <span class="badge ${qed.lipinski_pass ? "badge-green" : "badge-red"}">
            ${qed.lipinski_pass ? "PASS" : "FAIL"} (${qed.lipinski_violations ?? 0} violations)
          </span>
        </span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Veber's Rules (TPSA &le; 140, RotB &le; 10)</span>
        <span class="kv-val">
          <span class="badge ${qed.veber_pass ? "badge-green" : "badge-red"}">
            ${qed.veber_pass ? "PASS" : "FAIL"} (TPSA: ${fmtNum(qed.tpsa, 1)}, RotB: ${qed.rot_bonds ?? "N/A"})
          </span>
        </span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Ro5 Descriptors (MW / LogP / HBD / HBA)</span>
        <span class="kv-val mono">${fmtNum(qed.mw, 1)} / ${fmtNum(qed.logp, 2)} / ${qed.hbd ?? 0} / ${qed.hba ?? 0}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">PAINS A/B/C Substructure Filter</span>
        <span class="kv-val">${pains.length ? escapeHtml(pains.join("; ")) : "No PAINS structural alerts detected"}</span>
      </div>
    </div>`;
}


async function loadChemicalSpaceAndShap(smiles, subtype) {
  const statusNote = document.getElementById("shap-status-note");
  statusNote.textContent = `Fetching nearest training analogs and local TreeSHAP attributions for Human ${subtype}...`;

  const [nbRes, shapRes] = await Promise.allSettled([
    apiJson(`/api/neighbors?smiles=${encodeURIComponent(smiles)}&subtype=${encodeURIComponent(subtype)}&top_k=5`),
    apiJson(`/api/shap?smiles=${encodeURIComponent(smiles)}&subtype=${encodeURIComponent(subtype)}&top_k=10`),
  ]);

  if (nbRes.status === "fulfilled") {
    renderNeighborsCards(nbRes.value.neighbors || []);
  } else {
    document.getElementById("neighbors-grid").innerHTML = `<div class="muted-note">Failed to load neighbors: ${escapeHtml(nbRes.reason.message)}</div>`;
  }

  if (shapRes.status === "fulfilled") {
    renderLocalShap(shapRes.value);
    statusNote.textContent = `Loaded local TreeSHAP and top-5 training analogs for Human ${subtype}.`;
  } else {
    document.getElementById("local-shap-chart").innerHTML = `<div class="chart-fallback">Local TreeSHAP unavailable: ${escapeHtml(shapRes.reason.message)}</div>`;
    document.querySelector("#local-shap-table tbody").innerHTML = "";
    statusNote.textContent = `TreeSHAP unavailable for Human ${subtype}.`;
  }
}

function renderLocalShap(shapData) {
  document.getElementById("shap-base-value").textContent = `Base expected value: ${fmtNum(shapData.base_value, 3)} pKi`;
  const feats = shapData.features || [];
  const reversed = [...feats].reverse();

  renderPlotlyOrFallback(
    "local-shap-chart",
    [
      {
        type: "bar",
        orientation: "h",
        y: reversed.map((f) => f.feature),
        x: reversed.map((f) => f.value),
        marker: { color: reversed.map((f) => (f.positive ? "#4ade80" : "#f87171")) },
        hovertemplate: "%{y}<br>SHAP: %{x:+.4f} pKi<extra></extra>",
      },
    ],
    {
      margin: { l: 150, r: 24, t: 20, b: 45 },
      xaxis: { title: "SHAP Contribution (delta pKi)", zeroline: true, zerolinecolor: "#94a3b8", gridcolor: "#334155" },
      yaxis: { automargin: true },
    }
  );

  const tbody = document.querySelector("#local-shap-table tbody");
  tbody.innerHTML = feats
    .map(
      (f) => `<tr>
        <td class="mono">${escapeHtml(f.feature)}</td>
        <td class="mono" style="color:${f.positive ? "var(--green)" : "var(--red)"}; font-weight:700;">${f.value >= 0 ? "+" : ""}${fmtNum(f.value, 4)}</td>
        <td>${escapeHtml(f.direction)}</td>
        <td>${escapeHtml(f.meaning)}</td>
      </tr>`
    )
    .join("");
}

function renderNeighborsCards(neighbors) {
  const grid = document.getElementById("neighbors-grid");
  if (!neighbors || !neighbors.length) {
    grid.innerHTML = `<div class="muted-note">No training-set analogs with measured experimental activity found for this subtype.</div>`;
    return;
  }

  grid.innerHTML = neighbors
    .map(
      (nb) => `<div class="neighbor-card">
        <div class="neighbor-svg">${nb.svg && nb.svg.includes("<svg") ? nb.svg : `<span class="muted-note">No 2D SVG</span>`}</div>
        <div class="neighbor-meta">
          <span>Sim: <strong>${fmtNum(nb.tanimoto, 3)}</strong></span>
          <span class="badge ${nb.pchembl >= 6.0 ? "badge-green" : "badge-amber"}">pKi: ${fmtNum(nb.pchembl, 2)}</span>
        </div>
        <div class="neighbor-smiles" title="${escapeHtml(nb.smiles)}">${escapeHtml(nb.smiles)}</div>
      </div>`
    )
    .join("");
}


function renderGpcrCatalogTable() {
  const tbody = document.querySelector("#gpcr-catalog-table tbody");
  if (!tbody) return;
  tbody.innerHTML = (state.catalog || [])
    .map(
      (r) => `<tr>
        <td><strong>${escapeHtml(r.Subtype)}</strong></td>
        <td><span class="badge ${r.State === "Active" ? "badge-green" : "badge-cyan"}">${escapeHtml(r.State)}</span></td>
        <td class="mono">${escapeHtml(r["PDB ID"])}</td>
        <td>${escapeHtml(r.Method)}</td>
        <td class="mono">${escapeHtml(r.Resolution)}</td>
        <td>${escapeHtml(r["Ligand / Complex"])}</td>
        <td>${escapeHtml(r.Reference)}</td>
      </tr>`
    )
    .join("");
}

function loadGpcrStructure() {
  const subtype = document.getElementById("gpcr-subtype-select").value || "A2A";
  const confState = state.gpcrState || "active";
  const key = `${subtype}_${confState}`;

  const info = state.receptors[subtype] && state.receptors[subtype][confState];
  if (!info) return;

  document.getElementById("gpcr-viewer-title").textContent = info.title || `Human ${subtype} (${confState})`;
  document.getElementById("gpcr-pdb-badge").textContent = info.pdb_id === "NONE" ? "AlphaFold AF-P29275" : `PDB: ${info.pdb_id}`;

  document.getElementById("gpcr-meta-body").innerHTML = `
    <div class="kv-list">
      <div class="kv-row"><span class="kv-key">Subtype &amp; State</span><span class="kv-val">Human ${escapeHtml(subtype)} (${escapeHtml(confState)})</span></div>
      <div class="kv-row"><span class="kv-key">PDB Identifier</span><span class="kv-val mono">${escapeHtml(info.pdb_id)}</span></div>
      <div class="kv-row"><span class="kv-key">Experimental Method</span><span class="kv-val">${escapeHtml(info.method)}</span></div>
      <div class="kv-row"><span class="kv-key">Resolution</span><span class="kv-val mono">${escapeHtml(info.resolution)}</span></div>
      <div class="kv-row"><span class="kv-key">Co-Complex Ligand</span><span class="kv-val">${escapeHtml(info.ligand_name)} (${escapeHtml(info.ccd)})</span></div>
      <div class="kv-row"><span class="kv-key">Activation Mechanism</span><span class="kv-val">${escapeHtml(info.mechanism)}</span></div>
      <div class="kv-row"><span class="kv-key">CADD Structural Note</span><span class="kv-val">${escapeHtml(info.cadd_note)}</span></div>
    </div>`;

  if (state.gpcrLoadedKey === key) return;

  const container = document.getElementById("gpcr-3d-viewer");
  const overlay = document.getElementById("gpcr-3d-status");

  if (info.pdb_id === "NONE") {
    container.innerHTML = "";
    overlay.classList.remove("hidden");
    overlay.textContent = "No experimental inactive PDB structure exists for Human A2B. Structural modeling relies on AlphaFold DB (AF-P29275).";
    state.gpcrLoadedKey = key;
    return;
  }

  if (typeof $3Dmol === "undefined") {
    overlay.classList.remove("hidden");
    overlay.textContent = `3Dmol.js CDN unavailable offline. Inspect PDB ${info.pdb_id} directly on RCSB PDB.`;
    return;
  }

  overlay.classList.remove("hidden");
  overlay.textContent = `Downloading PDB:${info.pdb_id} from RCSB...`;
  container.innerHTML = "";

  try {
    const viewer = $3Dmol.createViewer(container, { backgroundColor: "#0b1120" });
    $3Dmol.download(`pdb:${info.pdb_id}`, viewer, {}, () => {
      viewer.setStyle({}, { cartoon: { color: "spectrum" } });
      if (info.ccd && info.ccd !== "N/A") {
        viewer.setStyle({ resn: info.ccd }, { stick: { radius: 0.25, colorscheme: "greenCarbon" }, sphere: { scale: 0.35 } });
      }
      viewer.zoomTo();
      viewer.render();
      overlay.classList.add("hidden");
      state.gpcrViewer = viewer;
      state.gpcrLoadedKey = key;
    });
  } catch (err) {
    overlay.classList.remove("hidden");
    overlay.textContent = `Failed loading PDB:${info.pdb_id}: ${err.message}`;
  }
}


async function loadProvenance() {
  try {
    const prov = await apiJson("/api/provenance");
    state.provenanceLoaded = true;
    document.getElementById("prov-schema-badge").textContent = prov.schema || "adeno-manifest/v2";

    const kpis = [
      ["Training Run ID", prov.training_run_id || prov.run_id || "N/A"],
      ["Platform Version", `v${prov.version || prov.app_version || "2.0.0"}`],
      ["Trained At", prov.trained_at || prov.timestamp || "N/A"],
      ["Training Molecules", prov.train_records ?? "N/A"],
      ["Data Fingerprint", prov.data_fingerprint || "N/A"],
      ["Model Fingerprint", prov.model_fingerprint || "N/A"],
    ];

    document.getElementById("provenance-summary-grid").innerHTML = kpis
      .map(
        ([label, val]) => `<div class="kpi-card">
          <div class="kpi-label">${escapeHtml(label)}</div>
          <div class="kpi-value" style="font-size:15px; word-break:break-all;">${escapeHtml(val)}</div>
        </div>`
      )
      .join("");

    const rows = [];
    for (const [k, v] of Object.entries(prov.data_hashes || {})) {
      rows.push(["Processed Data", k, v || "Missing"]);
    }
    for (const [k, v] of Object.entries(prov.model_hashes || {})) {
      rows.push(["Production Model", k, v || "Missing"]);
    }

    document.querySelector("#provenance-hashes-table tbody").innerHTML = rows
      .map(
        ([cat, path, hash]) => `<tr>
          <td><span class="badge ${cat === "Processed Data" ? "badge-cyan" : "badge-green"}">${cat}</span></td>
          <td class="mono">${escapeHtml(path)}</td>
          <td class="mono">${escapeHtml(hash)}</td>
        </tr>`
      )
      .join("");
  } catch (err) {
    document.getElementById("provenance-summary-grid").innerHTML = `<div class="muted-note">Failed loading provenance: ${escapeHtml(err.message)}</div>`;
  }
}


function bindBatchWorkspaceEvents() {
  const slider = document.getElementById("batch-threshold-slider");
  const valEl = document.getElementById("batch-threshold-value");
  slider.addEventListener("input", () => {
    valEl.textContent = Number(slider.value).toFixed(1);
  });

  document.getElementById("btn-load-sample-batch").addEventListener("click", () => {
    const lines = Object.values(state.presets).map((p) => p.smiles);
    document.getElementById("batch-smiles-textarea").value = lines.join("\n");
  });

  document.getElementById("batch-file-input").addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || "");
      const parsed = parseSmilesFromUpload(text);
      document.getElementById("batch-smiles-textarea").value = parsed.join("\n");
    };
    reader.readAsText(file);
    e.target.value = "";
  });

  document.getElementById("btn-run-batch").addEventListener("click", runBatchScreening);
  document.getElementById("btn-export-batch-csv").addEventListener("click", exportBatchCsv);

  document.querySelectorAll("#batch-results-table th[data-sort-key]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sortKey;
      if (state.batchSort.key === key) {
        state.batchSort.dir = state.batchSort.dir === "desc" ? "asc" : "desc";
      } else {
        state.batchSort.key = key;
        state.batchSort.dir = "desc";
      }
      renderBatchTable();
    });
  });
}

function parseSmilesFromUpload(rawText) {
  const lines = rawText.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (!lines.length) return [];
  const firstCols = lines[0].split(/[,\t]/).map((c) => c.trim().toLowerCase());
  const smiColIdx = firstCols.findIndex((c) => c === "smiles" || c === "canonical_smiles");
  const startIdx = smiColIdx >= 0 ? 1 : 0;
  const colIdx = smiColIdx >= 0 ? smiColIdx : 0;

  const out = [];
  for (let i = startIdx; i < lines.length; i++) {
    const parts = lines[i].split(/[,\t\s]+/);
    const cand = parts[colIdx] || parts[0];
    if (cand) out.push(cand.trim());
  }
  return out;
}

async function runBatchScreening() {
  const raw = document.getElementById("batch-smiles-textarea").value;
  const lines = raw
    .split(/\r?\n/)
    .map((l) => l.trim().split(/[\s,]+/)[0])
    .filter(Boolean);

  const threshold = Number(document.getElementById("batch-threshold-slider").value);
  const statusBar = document.getElementById("batch-status-bar");
  const btn = document.getElementById("btn-run-batch");

  if (!lines.length) {
    setBanner(statusBar, "Paste or upload at least one SMILES string.", "error");
    return;
  }
  if (lines.length > 250) {
    setBanner(statusBar, `Submitted ${lines.length} SMILES. Maximum batch size is 250.`, "error");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Screening...";
  setBanner(statusBar, `Screening ${lines.length} compounds across A1, A2A, A2B, and A3...`, "loading");

  try {
    const res = await apiJson("/api/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles_list: lines, threshold }),
    });
    state.batchData = res;
    setBanner(statusBar, null);
    document.getElementById("batch-results-shell").classList.remove("hidden");

    const kpis = [
      ["Submitted", res.total_submitted],
      ["Valid Screened", res.valid_count],
      [`Hits (>= ${fmtNum(threshold, 1)} pKi)`, res.hit_count],
      ["Invalid SMILES", res.invalid_count],
    ];
    document.getElementById("batch-kpi-grid").innerHTML = kpis
      .map(([l, v]) => `<div class="kpi-card"><div class="kpi-label">${l}</div><div class="kpi-value">${v}</div></div>`)
      .join("");

    renderBatchTable();
  } catch (err) {
    setBanner(statusBar, `Batch screening failed: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Batch Screening";
  }
}

function renderBatchTable() {
  if (!state.batchData) return;
  const rows = [...(state.batchData.results || [])];
  const { key, dir } = state.batchSort;

  rows.sort((a, b) => {
    const va = a[key];
    const vb = b[key];
    if (va === vb) return 0;
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    if (typeof va === "number" && typeof vb === "number") {
      return dir === "asc" ? va - vb : vb - va;
    }
    return dir === "asc" ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
  });

  const tbody = document.querySelector("#batch-results-table tbody");
  tbody.innerHTML = rows
    .map(
      (r) => `<tr>
        <td class="mono">${r.index}</td>
        <td class="smiles-cell" title="${escapeHtml(r.smiles)}">${escapeHtml(r.smiles)}</td>
        <td><span class="badge ${r.is_hit ? "badge-green" : ""}">${escapeHtml(r.best_target)}</span></td>
        <td class="mono"><strong>${fmtNum(r.max_pki, 2)}</strong></td>
        <td class="mono">${fmtNum(r.A1, 2)}</td>
        <td class="mono">${fmtNum(r.A2A, 2)}</td>
        <td class="mono">${fmtNum(r.A2B, 2)}</td>
        <td class="mono">${fmtNum(r.A3, 2)}</td>
        <td class="mono">${fmtNum(r.selectivity_margin, 2)}</td>
        <td class="mono">${fmtNum(r.tanimoto_max, 3)}</td>
        <td class="mono">${fmtNum(r.cns_mpo, 2)}</td>
        <td><span class="badge ${r.db_status === "de_novo" ? "badge-cyan" : "badge-green"}">${escapeHtml(r.db_status)}</span></td>
        <td><button type="button" class="btn btn-secondary btn-sm" data-profile-smiles="${escapeHtml(r.smiles)}">Profile</button></td>
      </tr>`
    )
    .join("");

  tbody.querySelectorAll("[data-profile-smiles]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const smi = btn.dataset.profileSmiles;
      document.getElementById("preset-select").value = "__custom__";
      document.getElementById("smiles-input").value = smi;
      switchWorkspace("single");
      runSinglePrediction();
    });
  });
}

function exportBatchCsv() {
  if (!state.batchData || !state.batchData.results) return;
  const cols = ["index", "smiles", "best_target", "max_pki", "A1", "A2A", "A2B", "A3", "selectivity_margin", "selectivity_fold", "tanimoto_max", "in_domain", "cns_mpo", "db_status"];
  const header = cols.join(",");
  const body = state.batchData.results
    .map((r) => cols.map((c) => `"${String(r[c] ?? "").replace(/"/g, '""')}"`).join(","))
    .join("\n");
  downloadBlob(`${header}\n${body}\n`, "adenosine_batch_screening.csv", "text/csv;charset=utf-8");
}


function bindBenchmarkFilters() {
  document.getElementById("parity-subtype-filter").addEventListener("change", renderParityAndResiduals);
  document.getElementById("parity-model-filter").addEventListener("change", renderParityAndResiduals);
  document.getElementById("global-shap-subtype").addEventListener("change", renderGlobalShapChart);
  document.getElementById("yrand-subtype").addEventListener("change", renderYRandChart);
}

async function loadBenchmarks() {
  const statusBar = document.getElementById("bench-status-bar");
  setBanner(statusBar, "Loading empirical holdout evaluation metrics, diagnostics, SHAP, and Y-randomization controls...", "loading");

  try {
    const data = await apiJson("/api/benchmarks");
    state.benchmarksData = data;
    state.benchmarksLoaded = true;
    setBanner(statusBar, null);
    document.getElementById("bench-content").classList.remove("hidden");

    renderBenchmarkKpisAndTables(data);
    renderParityAndResiduals();
    renderCalibrationQuartiles(data);
    renderDatasetDiagnostics(data);
    renderGlobalShapChart();
    renderYRandChart();
  } catch (err) {
    setBanner(statusBar, `Failed to load benchmarks: ${err.message}`, "error");
  }
}

function renderBenchmarkKpisAndTables(data) {
  const ev = data.evaluation_report || {};
  const ov = ev.overall || {};
  const per = ev.per_subtype || {};

  const kpis = [
    ["Consensus R2", fmtNum(metricPick(ov, "r2", "model_r2"), 3), `Test N = ${ev.n_test ?? "N/A"}`],
    ["Spearman rho", fmtNum(ov.spearman_rho, 3), "Rank-order correlation"],
    ["Consensus MAE", fmtNum(metricPick(ov, "mae", "model_mae"), 3), "pKi units"],
    ["Consensus RMSE", fmtNum(metricPick(ov, "rmse", "model_rmse"), 3), "pKi units"],
    ["90% MAPIE Coverage", fmtPct(ov.conformal_coverage_90, 1), "Target nominal: 90.0%"],
    ["Mean CI Width", fmtNum(ov.mean_interval_width, 2), "pKi interval span"],
  ];

  document.getElementById("bench-kpi-grid").innerHTML = kpis
    .map(
      ([label, val, sub]) => `<div class="kpi-card">
        <div class="kpi-label">${label}</div>
        <div class="kpi-value">${val}</div>
        <div class="kpi-sub">${sub}</div>
      </div>`
    )
    .join("");

  const tbody = document.querySelector("#bench-subtype-table tbody");
  tbody.innerHTML = SUBTYPES.map((st) => {
    const m = per[st] || {};
    return `<tr>
      <td><strong>Human ${st}</strong></td>
      <td class="mono">${m.n_test ?? "N/A"}</td>
      <td class="mono"><strong>${fmtNum(metricPick(m, "r2", "model_r2"), 3)}</strong></td>
      <td class="mono">${fmtNum(m.spearman_rho, 3)}</td>
      <td class="mono">${fmtNum(metricPick(m, "mae", "model_mae"), 3)}</td>
      <td class="mono">${fmtNum(metricPick(m, "rmse", "model_rmse"), 3)}</td>
      <td class="mono">${fmtNum(m.xgb_r2, 3)}</td>
      <td class="mono">${fmtNum(m.lgb_r2, 3)}</td>
      <td class="mono">${fmtNum(m.rf_r2, 3)}</td>
      <td class="mono">${fmtNum(m.baseline_r2, 3)}</td>
      <td class="mono">${fmtPct(m.conformal_coverage_90, 1)}</td>
    </tr>`;
  }).join("");

  const traces = [
    { name: "Consensus", x: SUBTYPES, y: SUBTYPES.map((st) => metricPick(per[st], "r2", "model_r2")), type: "bar", marker: { color: "#38bdf8" } },
    { name: "XGBoost", x: SUBTYPES, y: SUBTYPES.map((st) => (per[st] || {}).xgb_r2), type: "bar", marker: { color: "#4ade80" } },
    { name: "LightGBM", x: SUBTYPES, y: SUBTYPES.map((st) => (per[st] || {}).lgb_r2), type: "bar", marker: { color: "#fbbf24" } },
  ];
  if (SUBTYPES.some((st) => per[st] && per[st].rf_r2 !== undefined && per[st].rf_r2 !== null)) {
    traces.push({ name: "Random Forest", x: SUBTYPES, y: SUBTYPES.map((st) => (per[st] || {}).rf_r2), type: "bar", marker: { color: "#a78bfa" } });
  }
  traces.push({ name: "Dummy Baseline", x: SUBTYPES, y: SUBTYPES.map((st) => (per[st] || {}).baseline_r2), type: "bar", marker: { color: "#64748b" } });

  renderPlotlyOrFallback("bench-r2-bar-chart", traces, {
    barmode: "group",
    yaxis: { title: "Holdout R2", gridcolor: "#334155", zerolinecolor: "#94a3b8" },
    legend: { orientation: "h", y: 1.12, x: 0.5, xanchor: "center" },
  });
}

function renderParityAndResiduals() {
  const rows = (state.benchmarksData && state.benchmarksData.holdout_predictions) || [];
  const subFilter = document.getElementById("parity-subtype-filter").value;
  const modelKey = document.getElementById("parity-model-filter").value;

  const filtered = rows.filter((r) => {
    const matchSub = subFilter === "ALL" || r.subtype === subFilter;
    return matchSub && r.y_true !== null && r[modelKey] !== null && r[modelKey] !== undefined;
  });

  if (!filtered.length) {
    document.getElementById("bench-parity-chart").innerHTML = `<div class="chart-fallback">No holdout predictions available for the selected model/subtype.</div>`;
    document.getElementById("bench-residual-chart").innerHTML = `<div class="chart-fallback">No residuals available.</div>`;
    return;
  }

  const yTrue = filtered.map((r) => r.y_true);
  const yPred = filtered.map((r) => r[modelKey]);
  const residuals = filtered.map((r) => r[modelKey] - r.y_true);
  const minVal = Math.min(...yTrue, ...yPred);
  const maxVal = Math.max(...yTrue, ...yPred);

  renderPlotlyOrFallback(
    "bench-parity-chart",
    [
      {
        type: "scattergl",
        mode: "markers",
        x: yTrue,
        y: yPred,
        text: filtered.map((r) => `${r.subtype} | ${r.smiles}`),
        marker: { color: "#38bdf8", size: 5, opacity: 0.55 },
        name: "Holdout Compounds",
        hovertemplate: "%{text}<br>True: %{x:.2f}<br>Pred: %{y:.2f}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        x: [minVal, maxVal],
        y: [minVal, maxVal],
        line: { color: "#4ade80", dash: "dash", width: 2 },
        name: "Ideal Parity (y = x)",
      },
    ],
    {
      title: { text: `Holdout Parity (n=${filtered.length})`, font: { size: 13 } },
      xaxis: { title: "Experimental pKi (y_true)", gridcolor: "#334155" },
      yaxis: { title: "Predicted pKi", gridcolor: "#334155" },
      legend: { orientation: "h", y: -0.2, x: 0.5, xanchor: "center" },
    }
  );

  renderPlotlyOrFallback(
    "bench-residual-chart",
    [
      {
        type: "histogram",
        x: residuals,
        nbinsx: 40,
        marker: { color: "#38bdf8", line: { color: "#0f172a", width: 1 } },
        name: "Residuals",
      },
    ],
    {
      title: { text: "Holdout Residual Distribution (y_pred - y_true)", font: { size: 13 } },
      xaxis: { title: "Prediction Residual (pKi)", gridcolor: "#334155", zeroline: true, zerolinecolor: "#f87171" },
      yaxis: { title: "Compound Count", gridcolor: "#334155" },
    }
  );
}

function renderCalibrationQuartiles(data) {
  const ov = (data.evaluation_report && data.evaluation_report.overall) || {};
  const quartiles = ov.calibration_quartiles || [];
  const tbody = document.querySelector("#bench-calibration-table tbody");

  if (!quartiles.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="muted-note">No calibration quartile data available.</td></tr>`;
    return;
  }

  tbody.innerHTML = quartiles
    .map(
      (q) => `<tr>
        <td><strong>Q${q.bin}</strong></td>
        <td class="mono">${q.n}</td>
        <td class="mono">${fmtNum(q.std_mean, 3)}</td>
        <td class="mono">${fmtNum(q.mae_mean, 3)}</td>
        <td class="mono">${fmtPct(q.coverage_90, 1)}</td>
      </tr>`
    )
    .join("");

  renderPlotlyOrFallback(
    "bench-calibration-chart",
    [
      {
        type: "bar",
        name: "Empirical MAE (pKi)",
        x: quartiles.map((q) => `Q${q.bin}`),
        y: quartiles.map((q) => q.mae_mean),
        marker: { color: "#fbbf24" },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "90% Coverage",
        yaxis: "y2",
        x: quartiles.map((q) => `Q${q.bin}`),
        y: quartiles.map((q) => q.coverage_90),
        line: { color: "#4ade80", width: 2.5 },
        marker: { size: 8 },
      },
    ],
    {
      margin: { l: 50, r: 55, t: 24, b: 40 },
      yaxis: { title: "MAE (pKi)", gridcolor: "#334155" },
      yaxis2: { title: "Coverage", overlaying: "y", side: "right", range: [0.5, 1.05], tickformat: ".0%" },
      legend: { orientation: "h", y: 1.15, x: 0.5, xanchor: "center" },
    }
  );
}

function renderDatasetDiagnostics(data) {
  const diag = data.dataset_diagnostics;
  if (!diag) return;

  const stBreakdown = diag.standard_type_breakdown || {};
  const breakdownStr = Object.entries(stBreakdown)
    .map(([k, v]) => `${k}: ${Number(v).toLocaleString()}`)
    .join(" | ");
  document.getElementById("diag-assay-breakdown").textContent = breakdownStr || "Assay breakdown";

  const scaff = diag.scaffold_diversity || {};
  const pstats = diag.pchembl_stats || {};
  const kpis = [
    ["Total Curated Records", Number(diag.n_records_total || 0).toLocaleString()],
    ["Unique Molecules", Number(diag.n_unique_molecules || 0).toLocaleString()],
    ["Multi-Subtype Molecules", Number(diag.n_multi_subtype_molecules || 0).toLocaleString()],
    ["Murcko Scaffolds", Number(scaff.n_unique_scaffolds || 0).toLocaleString()],
    ["Scaffold Diversity", fmtNum(scaff.diversity_ratio, 3)],
    ["Mean / Median pKi", `${fmtNum(pstats.mean, 2)} / ${fmtNum(pstats.median, 2)}`],
  ];

  document.getElementById("diag-kpi-grid").innerHTML = kpis
    .map(([l, v]) => `<div class="kpi-card"><div class="kpi-label">${l}</div><div class="kpi-value" style="font-size:19px;">${v}</div></div>`)
    .join("");

  const per = diag.per_subtype || {};
  document.querySelector("#diag-subtype-table tbody").innerHTML = SUBTYPES.map((st) => {
    const d = per[st] || {};
    return `<tr>
      <td><strong>Human ${st}</strong></td>
      <td class="mono">${Number(d.n_records || 0).toLocaleString()}</td>
      <td class="mono">${fmtNum(d.pchembl_min, 2)}</td>
      <td class="mono">${fmtNum(d.pchembl_median, 2)}</td>
      <td class="mono">${fmtNum(d.pchembl_mean, 2)}</td>
      <td class="mono">${fmtNum(d.pchembl_max, 2)}</td>
      <td class="mono">${Number(d.n_weak_or_inactive_lt6 || 0).toLocaleString()}</td>
    </tr>`;
  }).join("");
}

function renderGlobalShapChart() {
  const shap = state.benchmarksData && state.benchmarksData.shap_summary;
  const st = document.getElementById("global-shap-subtype").value;
  const sub = shap && shap.per_subtype && shap.per_subtype[st];

  if (!sub || !sub.top_features || !sub.top_features.length) {
    document.getElementById("bench-global-shap-chart").innerHTML = `<div class="chart-fallback">Global TreeSHAP summary not available for ${st}.</div>`;
    return;
  }

  const feats = [...sub.top_features].slice(0, 12).reverse();
  renderPlotlyOrFallback(
    "bench-global-shap-chart",
    [
      {
        type: "bar",
        orientation: "h",
        y: feats.map((f) => f.feature),
        x: feats.map((f) => f.mean_abs_shap),
        marker: { color: "#38bdf8" },
        hovertemplate: "%{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>",
      },
    ],
    {
      margin: { l: 145, r: 20, t: 20, b: 45 },
      xaxis: { title: "Mean |SHAP| (pKi impact)", gridcolor: "#334155" },
      yaxis: { automargin: true },
    }
  );
}

function renderYRandChart() {
  const yr = state.benchmarksData && state.benchmarksData.y_randomization_summary;
  const st = document.getElementById("yrand-subtype").value;
  const sub = yr && yr.per_subtype && yr.per_subtype[st];

  if (!sub || !sub.shuffled_r2_scores) {
    document.getElementById("bench-yrand-chart").innerHTML = `<div class="chart-fallback">Y-Randomization control data unavailable for ${st}.</div>`;
    return;
  }

  renderPlotlyOrFallback(
    "bench-yrand-chart",
    [
      {
        type: "histogram",
        x: sub.shuffled_r2_scores,
        nbinsx: 12,
        marker: { color: "#64748b", line: { color: "#0f172a", width: 1 } },
        name: "Shuffled Null R2",
      },
    ],
    {
      margin: { l: 50, r: 24, t: 28, b: 45 },
      xaxis: { title: "Holdout R2", range: [Math.min(-0.25, ...sub.shuffled_r2_scores) - 0.05, Math.max(0.75, sub.real_r2 + 0.08)], gridcolor: "#334155" },
      yaxis: { title: "Null Folds", gridcolor: "#334155" },
      shapes: [
        {
          type: "line",
          x0: sub.real_r2,
          x1: sub.real_r2,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: "#4ade80", width: 3, dash: "dash" },
        },
      ],
      annotations: [
        {
          x: sub.real_r2,
          y: 0.95,
          yref: "paper",
          text: `Real R2 = ${fmtNum(sub.real_r2, 3)}`,
          showarrow: false,
          xanchor: "right",
          font: { color: "#4ade80", size: 12 },
        },
      ],
    }
  );

  document.getElementById("bench-yrand-meta").textContent =
    `Real R2: ${fmtNum(sub.real_r2, 3)} | Shuffled Mean R2: ${fmtNum(sub.shuffled_mean_r2, 3)} (SD: ${fmtNum(sub.shuffled_std_r2, 3)}) | Permutation p-value: ${fmtNum(sub.p_value, 4)} | Leakage warning: ${sub.leakage_warning ? "Yes" : "No"}`;
}
