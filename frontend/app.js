const API_BASE = "http://localhost:8000";

const modeButtons = document.querySelectorAll(".mode-btn");
const originalSampleImage = document.getElementById("originalSampleImage");
const fieldSampleImage = document.getElementById("fieldSampleImage");
const formationImage = document.getElementById("formationImage");
const originalSampleHint = document.getElementById("originalSampleHint");
const fieldSampleHint = document.getElementById("fieldSampleHint");
const formationHint = document.getElementById("formationHint");
const insightButtons = document.querySelectorAll(".insight-btn");
const insightNumbers = document.getElementById("insightNumbers");
const insightGraph = document.getElementById("insightGraph");
const viewerControlButtons = document.querySelectorAll(".ctrl-btn");
const processFpsInput = document.getElementById("processFpsInput");
const runAnalysisBtn = document.getElementById("runAnalysisBtn");
const runStatusChip = document.getElementById("runStatusChip");
const videoPathInput = document.getElementById("videoPathInput");
const filmInput = document.getElementById("filmInput");
const framesStat = document.getElementById("framesStat");
const originalFpsStat = document.getElementById("originalFpsStat");
const avgSelectedStat = document.getElementById("avgSelectedStat");
const timelineClock = document.getElementById("timelineClock");
const globalProgress = document.getElementById("globalProgress");
const timelineBar = document.querySelector(".timeline-bar");
const playPauseBtn = document.getElementById("playPauseBtn");
const playbackSpeedSelect = document.getElementById("playbackSpeedSelect");
const saveNoteBtn = document.getElementById("saveNoteBtn");
const noteInput = document.getElementById("noteInput");
const noteMarkers = document.getElementById("noteMarkers");
const overallCommentBubble = document.getElementById("overallCommentBubble");
const currentCommentBubble = document.getElementById("currentCommentBubble");
const togglePlayerBox = document.getElementById("togglePlayerBox");
const togglePocketBox = document.getElementById("togglePocketBox");
const toggleYardline = document.getElementById("toggleYardline");
const toggleHashmark = document.getElementById("toggleHashmark");
const originalOverlayCanvas = document.getElementById("originalOverlayCanvas");
const originalViewport = document.getElementById("originalViewport");
const countOffWr = document.getElementById("countOffWr");
const countOffLineman = document.getElementById("countOffLineman");
const countOffBacks = document.getElementById("countOffBacks");
const countDefDl = document.getElementById("countDefDl");
const countDefSecond = document.getElementById("countDefSecond");
const countDefDeep = document.getElementById("countDefDeep");

const viewerState = {
  original: { scale: 1, tx: 0, ty: 0, minScale: 1, maxScale: 4 },
  field: { scale: 1, tx: 0, ty: 0, minScale: 1, maxScale: 4 },
};

const appState = {
  currentRunId: null,
  currentMode: "overall",
  frames: [],
  currentFrameIndex: 0,
  notes: [],
  runStatus: null,
  pollTimer: null,
  playbackTimer: null,
  isPlaying: false,
  playbackSpeed: 1,
  overlayByFrame: {},
  isUploadingFilm: false,
};

function apiUrl(path) {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE}${path}`;
}

function formatTime(sec) {
  const safe = Number.isFinite(sec) && sec >= 0 ? sec : 0;
  const m = Math.floor(safe / 60);
  const s = Math.floor(safe % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function currentFrame() {
  if (!appState.frames.length) return null;
  return appState.frames[Math.max(0, Math.min(appState.currentFrameIndex, appState.frames.length - 1))];
}

function setStatus(statusText, tone = "idle") {
  if (!runStatusChip) return;
  runStatusChip.textContent = statusText;
  runStatusChip.classList.remove("status-idle", "status-running", "status-complete");
  // Inline style fallback avoids stale CSS cache issues.
  if (tone === "running") {
    runStatusChip.style.background = "#cf2b2b";
    runStatusChip.style.borderColor = "#e15a5a";
    runStatusChip.style.color = "#ffffff";
  } else if (tone === "complete") {
    runStatusChip.style.background = "#2c9a55";
    runStatusChip.style.borderColor = "#49ba73";
    runStatusChip.style.color = "#ffffff";
  } else {
    runStatusChip.style.background = "#ffffff";
    runStatusChip.style.borderColor = "#d8dfef";
    runStatusChip.style.color = "#18213b";
  }
  if (tone === "running") runStatusChip.classList.add("status-running");
  else if (tone === "complete") runStatusChip.classList.add("status-complete");
  else runStatusChip.classList.add("status-idle");
}

function setCountText(el, value) {
  if (!el) return;
  const n = Number.parseInt(String(value), 10);
  el.textContent = Number.isFinite(n) ? String(n) : "0";
}

function setPlayButtonLabel() {
  if (!playPauseBtn) return;
  playPauseBtn.textContent = appState.isPlaying ? "Pause" : "Play";
}

function applyViewerTransform(viewerKey, imageEl) {
  const state = viewerState[viewerKey];
  if (!state || !imageEl) return;
  const transform = `translate(${state.tx}px, ${state.ty}px) scale(${state.scale})`;
  imageEl.style.transform = transform;
  if (viewerKey === "original" && originalOverlayCanvas) {
    originalOverlayCanvas.style.transform = transform;
    originalOverlayCanvas.style.transformOrigin = "center center";
  }
}

function zoomViewer(viewerKey, delta) {
  const state = viewerState[viewerKey];
  const imgEl = document.querySelector(`[data-viewer="${viewerKey}"]`);
  if (!state || !imgEl) return;
  state.scale = Math.min(state.maxScale, Math.max(state.minScale, state.scale + delta));
  applyViewerTransform(viewerKey, imgEl);
}

function resetViewer(viewerKey) {
  const state = viewerState[viewerKey];
  const imgEl = document.querySelector(`[data-viewer="${viewerKey}"]`);
  if (!state || !imgEl) return;
  state.scale = 1;
  state.tx = 0;
  state.ty = 0;
  applyViewerTransform(viewerKey, imgEl);
}

function setupDragPan(viewerKey) {
  const imageEl = document.querySelector(`[data-viewer="${viewerKey}"]`);
  const state = viewerState[viewerKey];
  if (!imageEl || !state) return;

  let dragging = false;
  let startX = 0;
  let startY = 0;
  let startTx = 0;
  let startTy = 0;

  imageEl.addEventListener("mousedown", (event) => {
    event.preventDefault();
    dragging = true;
    startX = event.clientX;
    startY = event.clientY;
    startTx = state.tx;
    startTy = state.ty;
    imageEl.classList.add("dragging");
  });

  window.addEventListener("mousemove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - startX;
    const dy = event.clientY - startY;
    state.tx = startTx + dx;
    state.ty = startTy + dy;
    applyViewerTransform(viewerKey, imageEl);
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    imageEl.classList.remove("dragging");
  });

  imageEl.addEventListener("wheel", (event) => {
    event.preventDefault();
    const zoomStep = event.deltaY < 0 ? 0.12 : -0.12;
    zoomViewer(viewerKey, zoomStep);
  });
}

function parseYoloBoxes(txt) {
  if (!txt) return [];
  return txt
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(/\s+/).map((v) => Number.parseFloat(v));
      if (parts.length < 5 || parts.slice(1, 5).some((v) => !Number.isFinite(v))) return null;
      return { cx: parts[1], cy: parts[2], w: parts[3], h: parts[4] };
    })
    .filter(Boolean);
}

function parseLineSegments(txt) {
  if (!txt) return [];
  return txt
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(/\s+/).map((v) => Number.parseFloat(v));
      if (parts.length < 4 || parts.slice(0, 4).some((v) => !Number.isFinite(v))) return null;
      return { x1: parts[0], y1: parts[1], x2: parts[2], y2: parts[3] };
    })
    .filter(Boolean);
}

function fitContainRect(imgW, imgH, viewW, viewH) {
  if (!(imgW > 0 && imgH > 0 && viewW > 0 && viewH > 0)) return null;
  const scale = Math.min(viewW / imgW, viewH / imgH);
  const drawW = imgW * scale;
  const drawH = imgH * scale;
  const x = (viewW - drawW) / 2;
  const y = (viewH - drawH) / 2;
  return { x, y, drawW, drawH };
}

async function fetchOverlayData(frame) {
  if (!frame?.frame_key) return null;
  if (appState.overlayByFrame[frame.frame_key]) return appState.overlayByFrame[frame.frame_key];
  const urls = frame.overlay_txt_urls || {};
  const [players, pocket, yardline, harshmark] = await Promise.all([
    urls.players ? fetch(apiUrl(urls.players)).then((r) => (r.ok ? r.text() : "")) : Promise.resolve(""),
    urls.pocket ? fetch(apiUrl(urls.pocket)).then((r) => (r.ok ? r.text() : "")) : Promise.resolve(""),
    urls.yardline_line ? fetch(apiUrl(urls.yardline_line)).then((r) => (r.ok ? r.text() : "")) : Promise.resolve(""),
    urls.harshmark_line ? fetch(apiUrl(urls.harshmark_line)).then((r) => (r.ok ? r.text() : "")) : Promise.resolve(""),
  ]);
  const parsed = {
    players: parseYoloBoxes(players),
    pocket: parseYoloBoxes(pocket),
    yardline: parseLineSegments(yardline),
    harshmark: parseLineSegments(harshmark),
  };
  appState.overlayByFrame[frame.frame_key] = parsed;
  return parsed;
}

async function renderOriginalOverlay() {
  if (!originalOverlayCanvas || !originalViewport || !originalSampleImage) return;
  const frame = currentFrame();
  if (!frame) return;
  const overlay = await fetchOverlayData(frame);
  if (!overlay) return;

  const dpr = window.devicePixelRatio || 1;
  const viewW = Math.max(1, originalViewport.clientWidth);
  const viewH = Math.max(1, originalViewport.clientHeight);
  originalOverlayCanvas.width = Math.round(viewW * dpr);
  originalOverlayCanvas.height = Math.round(viewH * dpr);
  originalOverlayCanvas.style.width = `${viewW}px`;
  originalOverlayCanvas.style.height = `${viewH}px`;

  const ctx = originalOverlayCanvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, viewW, viewH);

  const fit = fitContainRect(
    originalSampleImage.naturalWidth || 1,
    originalSampleImage.naturalHeight || 1,
    viewW,
    viewH
  );
  if (!fit) return;

  const toX = (nx) => fit.x + nx * fit.drawW;
  const toY = (ny) => fit.y + ny * fit.drawH;

  if (togglePlayerBox?.checked) {
    ctx.strokeStyle = "#67a8ff";
    ctx.lineWidth = 1.5;
    overlay.players.forEach((b) => {
      const x = toX(b.cx - b.w / 2);
      const y = toY(b.cy - b.h / 2);
      const w = b.w * fit.drawW;
      const h = b.h * fit.drawH;
      ctx.strokeRect(x, y, w, h);
    });
  }

  if (togglePocketBox?.checked) {
    ctx.strokeStyle = "#ffd34f";
    ctx.lineWidth = 2;
    overlay.pocket.forEach((b) => {
      const x = toX(b.cx - b.w / 2);
      const y = toY(b.cy - b.h / 2);
      const w = b.w * fit.drawW;
      const h = b.h * fit.drawH;
      ctx.strokeRect(x, y, w, h);
    });
  }

  if (toggleYardline?.checked) {
    ctx.strokeStyle = "#ff7f6a";
    ctx.lineWidth = 2;
    overlay.yardline.forEach((l) => {
      ctx.beginPath();
      ctx.moveTo(toX(l.x1), toY(l.y1));
      ctx.lineTo(toX(l.x2), toY(l.y2));
      ctx.stroke();
    });
  }

  if (toggleHashmark?.checked) {
    ctx.strokeStyle = "#47e6b0";
    ctx.lineWidth = 2;
    overlay.harshmark.forEach((l) => {
      ctx.beginPath();
      ctx.moveTo(toX(l.x1), toY(l.y1));
      ctx.lineTo(toX(l.x2), toY(l.y2));
      ctx.stroke();
    });
  }
}

function renderCommentBubbles() {
  if (!overallCommentBubble || !currentCommentBubble) return;
  const overall = appState.notes.find((n) => n.scope === "overall");
  const hasOverall = Boolean(overall?.text);
  overallCommentBubble.textContent = hasOverall ? `Overall: ${overall.text}` : "Overall: no comment yet.";
  overallCommentBubble.classList.toggle("comment-empty", !hasOverall);

  const frame = currentFrame();
  const frameKey = frame?.frame_key || null;
  if (!frameKey) {
    currentCommentBubble.textContent = "Current frame: no frame selected.";
    currentCommentBubble.classList.add("comment-empty");
    return;
  }
  const notes = appState.notes.filter((n) => n.scope === "frame" && n.frame_key === frameKey).map((n) => n.text);
  const hasFrameComment = notes.length > 0;
  currentCommentBubble.textContent = hasFrameComment
    ? `Current (${frameKey}): ${notes.join(" | ")}`
    : `Current (${frameKey}): no comment yet.`;
  currentCommentBubble.classList.toggle("comment-empty", !hasFrameComment);
}

function renderMarkers() {
  if (!noteMarkers || !appState.frames.length) return;
  noteMarkers.innerHTML = "";

  appState.notes
    .filter((n) => n.scope === "frame" && n.frame_key)
    .forEach((note) => {
      const idx = appState.frames.findIndex((f) => f.frame_key === note.frame_key);
      if (idx < 0) return;
      const leftPercent = appState.frames.length > 1 ? (idx / (appState.frames.length - 1)) * 100 : 0;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "marker marker-frame";
      btn.style.left = `${leftPercent}%`;
      btn.title = note.text;
      btn.addEventListener("click", () => {
        appState.currentFrameIndex = idx;
        updateCurrentFrame();
      });
      noteMarkers.appendChild(btn);
    });
}

function updateCurrentFrame() {
  if (!appState.frames.length) return;
  const frame = currentFrame();
  if (!frame) return;

  if (originalSampleImage && frame.original_image_url) {
    originalSampleImage.src = apiUrl(frame.original_image_url);
    applyViewerTransform("original", originalSampleImage);
  }
  if (fieldSampleImage && frame.field_image_url) {
    fieldSampleImage.src = apiUrl(frame.field_image_url);
    applyViewerTransform("field", fieldSampleImage);
  }
  if (originalSampleHint) originalSampleHint.textContent = frame.frame_key || "original frame";
  if (fieldSampleHint) fieldSampleHint.textContent = frame.frame_key || "field frame";

  if (globalProgress && appState.frames.length > 1) {
    const pct = (appState.currentFrameIndex / (appState.frames.length - 1)) * 100;
    globalProgress.style.width = `${pct}%`;
  } else if (globalProgress && appState.frames.length <= 1) {
    globalProgress.style.width = "0%";
  }
  const now = frame.timestamp_sec ?? 0;
  const last = appState.frames[appState.frames.length - 1]?.timestamp_sec ?? now;
  if (timelineClock) timelineClock.textContent = `${formatTime(now)} / ${formatTime(last)}`;
  renderOriginalOverlay();
  renderCommentBubbles();
}

function frameIntervalMs() {
  const fps = Number(appState.runStatus?.process_fps) > 0 ? Number(appState.runStatus.process_fps) : 12;
  return 1000 / (fps * appState.playbackSpeed);
}

function stopPlayback() {
  if (appState.playbackTimer) {
    clearInterval(appState.playbackTimer);
    appState.playbackTimer = null;
  }
  appState.isPlaying = false;
  setPlayButtonLabel();
}

function tickPlayback() {
  if (!appState.frames.length) return;
  if (appState.currentFrameIndex >= appState.frames.length - 1) {
    stopPlayback();
    return;
  }
  appState.currentFrameIndex += 1;
  updateCurrentFrame();
}

function startPlayback() {
  if (!appState.frames.length) return;
  if (appState.isPlaying) return;
  appState.isPlaying = true;
  setPlayButtonLabel();
  appState.playbackTimer = setInterval(tickPlayback, frameIntervalMs());
}

function restartPlaybackIfNeeded() {
  if (!appState.isPlaying) return;
  if (appState.playbackTimer) clearInterval(appState.playbackTimer);
  appState.playbackTimer = setInterval(tickPlayback, frameIntervalMs());
}

async function loadRunData(runId) {
  const [summaryRes, framesRes, assetsRes, notesRes] = await Promise.all([
    fetch(apiUrl(`/api/runs/${runId}/summary`)),
    fetch(apiUrl(`/api/runs/${runId}/frames`)),
    fetch(apiUrl(`/api/runs/${runId}/assets`)),
    fetch(apiUrl(`/api/runs/${runId}/notes`)),
  ]);
  if (!summaryRes.ok || !framesRes.ok || !assetsRes.ok || !notesRes.ok) return;

  const summary = await summaryRes.json();
  const framePayload = await framesRes.json();
  const assets = await assetsRes.json();
  const notePayload = await notesRes.json();

  appState.frames = framePayload.frames || [];
  appState.notes = notePayload.notes || [];
  appState.currentFrameIndex = Math.max(0, Math.min(appState.currentFrameIndex, appState.frames.length - 1));

  if (framesStat) framesStat.textContent = String(appState.frames.length);
  if (originalFpsStat) originalFpsStat.textContent = Number(summary?.run?.original_fps || 0).toFixed(2);
  if (avgSelectedStat) avgSelectedStat.textContent = Number(summary?.stats?.avg_selected || 0).toFixed(1);
  const counts = summary?.formation?.counts || {};
  setCountText(countOffWr, counts.off_wr);
  setCountText(countOffLineman, counts.off_lineman);
  setCountText(countOffBacks, counts.off_backs);
  setCountText(countDefDl, counts.def_dl);
  setCountText(countDefSecond, counts.def_second);
  setCountText(countDefDeep, counts.def_deep);
  if (formationImage && assets?.formation_image_url) {
    formationImage.src = apiUrl(assets.formation_image_url);
    if (formationHint) formationHint.textContent = "Formation image loaded from current run.";
  }

  updateCurrentFrame();
  renderMarkers();
  renderCommentBubbles();
}

async function refreshRunStatus(runId) {
  const res = await fetch(apiUrl(`/api/runs/${runId}/status`));
  if (!res.ok) return;
  const status = await res.json();
  appState.runStatus = status;
  const progressPct = Math.round((status.progress || 0) * 100);
  const statusText = String(status.status || "").toLowerCase();
  const tone = statusText === "completed" ? "complete" : (statusText === "running" || statusText === "queued" ? "running" : "idle");
  setStatus(`${status.status} ${progressPct}%`, tone);
  await loadRunData(runId);
  if (!["queued", "running"].includes(status.status) && appState.pollTimer) {
    clearInterval(appState.pollTimer);
    appState.pollTimer = null;
  }
}

async function startRun() {
  if (!videoPathInput || !processFpsInput) return;
  if (appState.isUploadingFilm) {
    setStatus("Please wait for film upload to finish", "running");
    return;
  }
  const videoPath = videoPathInput.value.trim();
  const processFps = Number.parseFloat(processFpsInput.value);
  if (!videoPath) {
    setStatus("Video path required", "idle");
    return;
  }
  if (!Number.isFinite(processFps) || processFps <= 0) {
    setStatus("Invalid FPS", "idle");
    return;
  }
  setStatus("Submitting...", "running");
  const res = await fetch(apiUrl("/api/runs"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_path: videoPath, process_fps: processFps }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    setStatus(err.detail || "Run start failed", "idle");
    return;
  }
  const run = await res.json();
  appState.currentRunId = run.run_id;
  setStatus(`Queued @ ${processFps} FPS`, "running");
  if (appState.pollTimer) clearInterval(appState.pollTimer);
  appState.pollTimer = setInterval(() => {
    if (appState.currentRunId) refreshRunStatus(appState.currentRunId);
  }, 2500);
  await refreshRunStatus(appState.currentRunId);
}

async function importFilmToBackend() {
  if (!filmInput || !videoPathInput) return;
  const file = filmInput.files?.[0];
  if (!file) return;
  appState.isUploadingFilm = true;
  if (runAnalysisBtn) runAnalysisBtn.disabled = true;
  setStatus(`Uploading ${file.name}...`, "running");

  const body = new FormData();
  body.append("file", file, file.name);

  try {
    const res = await fetch(apiUrl("/api/uploads/video"), {
      method: "POST",
      body,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Upload failed");
    }
    const payload = await res.json();
    const resolvedPath = String(payload.video_path || "").trim();
    if (!resolvedPath) throw new Error("Upload succeeded but no path returned");
    videoPathInput.value = resolvedPath;
    setStatus(`Imported: ${file.name}`, "idle");
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Upload failed";
    setStatus(msg, "idle");
  } finally {
    appState.isUploadingFilm = false;
    if (runAnalysisBtn) runAnalysisBtn.disabled = false;
  }
}

async function saveNote() {
  if (!appState.currentRunId || !noteInput) return;
  const text = noteInput.value.trim();
  if (!text) return;
  const frame = currentFrame();
  const payload = {
    text,
    scope: appState.currentMode,
    frame_key: appState.currentMode === "frame" ? frame?.frame_key || null : null,
    timestamp_sec: appState.currentMode === "frame" ? frame?.timestamp_sec ?? null : null,
  };
  const res = await fetch(apiUrl(`/api/runs/${appState.currentRunId}/notes`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) return;
  noteInput.value = "";
  const refreshed = await fetch(apiUrl(`/api/runs/${appState.currentRunId}/notes`));
  if (!refreshed.ok) return;
  const noteData = await refreshed.json();
  appState.notes = noteData.notes || [];
  renderMarkers();
  renderCommentBubbles();
}

function bindImageWithFallback(imgEl, hintEl, sourceCandidates, label) {
  if (!imgEl || !hintEl) return;
  let idx = 0;
  const cacheBust = `v=${Date.now()}`;
  const tryNext = () => {
    if (idx >= sourceCandidates.length) {
      hintEl.textContent = `${label} not found.`;
      return;
    }
    const candidate = sourceCandidates[idx++];
    const srcWithBust = candidate.includes("?") ? `${candidate}&${cacheBust}` : `${candidate}?${cacheBust}`;
    imgEl.onerror = tryNext;
    imgEl.onload = () => {
      hintEl.textContent = `Loaded from: ${candidate}`;
      const viewerKey = imgEl.dataset.viewer;
      if (viewerKey && viewerState[viewerKey]) applyViewerTransform(viewerKey, imgEl);
      if (viewerKey === "original") renderOriginalOverlay();
    };
    imgEl.src = srcWithBust;
  };
  tryNext();
}

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const mode = button.dataset.noteMode;
    appState.currentMode = mode === "frame" ? "frame" : "overall";
    modeButtons.forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
  });
});

insightButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const mode = button.dataset.insightMode;
    insightButtons.forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    const showGraph = mode === "graph";
    if (insightNumbers) insightNumbers.classList.toggle("active", !showGraph);
    if (insightGraph) insightGraph.classList.toggle("active", showGraph);
  });
});

viewerControlButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const viewerKey = btn.dataset.target;
    const action = btn.dataset.action;
    if (!viewerKey || !action) return;
    if (action === "zoom-in") zoomViewer(viewerKey, 0.2);
    if (action === "zoom-out") zoomViewer(viewerKey, -0.2);
    if (action === "reset") resetViewer(viewerKey);
  });
});

if (timelineBar) {
  let scrubDragging = false;

  const seekFromClientX = (clientX) => {
    if (!appState.frames.length || appState.frames.length < 2) return;
    const rect = timelineBar.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
    const pct = rect.width > 0 ? x / rect.width : 0;
    appState.currentFrameIndex = Math.round(pct * (appState.frames.length - 1));
    updateCurrentFrame();
  };

  timelineBar.addEventListener("mousedown", (event) => {
    scrubDragging = true;
    seekFromClientX(event.clientX);
  });

  window.addEventListener("mousemove", (event) => {
    if (!scrubDragging) return;
    seekFromClientX(event.clientX);
  });

  window.addEventListener("mouseup", () => {
    scrubDragging = false;
  });
}

[togglePlayerBox, togglePocketBox, toggleYardline, toggleHashmark].forEach((el) => {
  if (!el) return;
  el.addEventListener("change", () => {
    renderOriginalOverlay();
  });
});

if (runAnalysisBtn) runAnalysisBtn.addEventListener("click", startRun);
if (saveNoteBtn) saveNoteBtn.addEventListener("click", saveNote);
if (playPauseBtn) {
  playPauseBtn.addEventListener("click", () => {
    if (appState.isPlaying) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });
}
if (playbackSpeedSelect) {
  playbackSpeedSelect.addEventListener("change", () => {
    const v = Number.parseFloat(playbackSpeedSelect.value);
    appState.playbackSpeed = Number.isFinite(v) && v > 0 ? v : 1;
    restartPlaybackIfNeeded();
  });
}
if (filmInput && videoPathInput) {
  filmInput.addEventListener("change", () => {
    importFilmToBackend();
  });
}

setupDragPan("original");
setupDragPan("field");
if (originalSampleImage) {
  originalSampleImage.addEventListener("load", () => {
    renderOriginalOverlay();
  });
}
window.addEventListener("resize", () => {
  renderOriginalOverlay();
});

bindImageWithFallback(
  originalSampleImage,
  originalSampleHint,
  [
    "./Test/Video_Test_1/Output_globaltrack/frames/frame_000000/frame_000000.jpg",
    "/Test/Video_Test_1/Output_globaltrack/frames/frame_000000/frame_000000.jpg",
  ],
  "Original sample"
);

setPlayButtonLabel();

bindImageWithFallback(
  fieldSampleImage,
  fieldSampleHint,
  [
    "./Test/Video_Test_1/Output_globaltrack/frames/frame_000000/frame_000000_field.jpg",
    "/Test/Video_Test_1/Output_globaltrack/frames/frame_000000/frame_000000_field.jpg",
  ],
  "2D field sample"
);
