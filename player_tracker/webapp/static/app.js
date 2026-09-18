/* player_tracker — фронтенд без сборки. Состояние: выбранный ролик, выбранная задача, таймер опроса. */
const $ = (id) => document.getElementById(id);
const api = async (url, opts = {}) => {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
};
const fmtT = (s) => (s == null ? "" : `${Math.floor(s / 60)}:${String((s % 60).toFixed(1)).padStart(4, "0")}`);
const STATE_COLOR = { active: "#22a06b", contested: "#e8a317", lost: "#d7263d", idle: "#b8bcc6", ambiguous: "#8b5cf6" };

let currentVideo = null;
let frameScale = 1, frameW = 0, frameH = 0;
let point = null;
let currentJob = null;
let pollTimer = null;
let timelineFrames = null;
let wasBallBusy = false;
let wasRegBusy = false;

// ---------- ролики ----------
async function loadVideos() {
  const videos = await api("/api/videos");
  const ul = $("video-list");
  ul.innerHTML = "";
  for (const v of videos) {
    const li = document.createElement("li");
    const i = v.info;
    li.innerHTML = `<span class="name">${v.name}</span><span class="meta">${i.width}×${i.height} · ${i.fps.toFixed(0)} fps · ${fmtT(i.duration)} · ${v.is_hdr ? "HDR" : "SDR"}${i.rotation ? " · поворот " + i.rotation + "°" : ""}</span>`;
    li.onclick = () => selectVideo(v);
    if (currentVideo && currentVideo.id === v.id) li.classList.add("selected");
    ul.appendChild(li);
  }
}

$("upload-form").onsubmit = async (e) => {
  e.preventDefault();
  const file = $("upload-file").files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  const prog = $("upload-progress"), bar = prog.querySelector(".bar");
  prog.classList.remove("hidden");
  $("upload-btn").disabled = true;
  try {
    const rec = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/videos");
      xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) bar.style.width = (100 * ev.loaded / ev.total) + "%"; };
      xhr.onload = () => xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(JSON.parse(xhr.responseText).detail || xhr.statusText));
      xhr.onerror = () => reject(new Error("сеть"));
      xhr.send(fd);
    });
    await loadVideos();
    selectVideo(rec);
    if (rec.duplicate) $("select-hint").textContent = `Этот ролик уже загружен («${rec.name}») — открыт прежний, со всеми его задачами и разборами.`;
  } catch (err) {
    alert("Загрузка не удалась: " + err.message);
  } finally {
    prog.classList.add("hidden"); bar.style.width = "0"; $("upload-btn").disabled = false; $("upload-file").value = "";
  }
};

function selectVideo(v) {
  currentVideo = v;
  point = null;
  $("sel-point").value = ""; $("marker").classList.add("hidden"); $("start-btn").disabled = true;
  $("select-hint").textContent = `Ролик: ${v.name}`;
  $("select-body").classList.remove("hidden");
  const dur = Math.max(v.info.duration || 0, 0.5);
  $("sel-slider").max = dur.toFixed(1); $("sel-time").max = dur.toFixed(1);
  $("sel-slider").value = 0; $("sel-time").value = 0;
  const i = v.info;
  $("video-meta").textContent = `${i.codec} ${i.pix_fmt} ${v.is_hdr ? "HDR " + i.color_transfer : "SDR"} · ${i.is_vfr ? "VFR" : "CFR"} · ~${i.nb_frames} кадров`;
  loadVideos();
  loadFrame();
  loadExisting(v.id);
}

let existingJobs = [];
async function loadExisting(videoId) {
  existingJobs = [];
  try { existingJobs = await api(`/api/videos/${videoId}/jobs`); } catch (_) {}
  const done = existingJobs.filter((x) => x.status === "done" || x.status === "running" || x.status === "queued");
  $("existing").classList.toggle("hidden", !done.length);
  const ul = $("existing-list"); ul.innerHTML = "";
  for (const x of done) {
    const li = document.createElement("li");
    const who = [x.player.age ? x.player.age + " лет" : null, profileOptions.positions[x.player.position],
      (profileOptions.game_formats[x.player.game_format] || "").split(" (")[0] || null].filter(Boolean).join(", ");
    const parts = [
      `игрок в точке ${x.init.point ? x.init.point.map((v) => v.toFixed(0)).join(", ") : "—"} на ${fmtT(x.init.t_sec)}`,
      who || null,
      x.tracked_share != null ? `отслежено ${Math.round(x.tracked_share * 100)}%` : x.status,
      x.corrections ? `поправок ${x.corrections}` : null,
      x.analyses ? `разборов ${x.analyses}${x.analysis_current ? "" : " (по прежней версии)"}` : "разбора нет",
    ].filter(Boolean);
    li.innerHTML = `<span>${new Date(x.created * 1000).toLocaleString("ru-RU")} · ${parts.join(" · ")}</span><button>Открыть</button>`;
    li.querySelector("button").onclick = () => openJob(x.id);
    ul.appendChild(li);
  }
}

async function loadFrame() {
  if (!currentVideo) return;
  const t = parseFloat($("sel-time").value) || 0;
  const img = $("frame-img");
  const r = await fetch(`/api/videos/${currentVideo.id}/frame?t=${t}&width=1280`);
  if (!r.ok) { $("select-hint").textContent = "Кадр не прочитан: " + (await r.text()); return; }
  frameScale = parseFloat(r.headers.get("X-Scale") || "1");
  frameW = parseInt(r.headers.get("X-Width") || "0"); frameH = parseInt(r.headers.get("X-Height") || "0");
  img.src = URL.createObjectURL(await r.blob());
  point = null; $("sel-point").value = ""; $("marker").classList.add("hidden"); updateStart();
}
$("sel-slider").oninput = () => { $("sel-time").value = $("sel-slider").value; };
$("sel-slider").onchange = loadFrame;
$("sel-time").onchange = () => { $("sel-slider").value = $("sel-time").value; loadFrame(); };

$("frame-img").onclick = (e) => {
  const img = e.target, rect = img.getBoundingClientRect();
  // кадр ещё не загрузился: naturalWidth = 0 даёт точку (0, 0) и задачу, в которой цель не найдётся
  if (!img.naturalWidth || !rect.width) return;
  const kx = img.naturalWidth / rect.width, ky = img.naturalHeight / rect.height;
  const px = (e.clientX - rect.left) * kx, py = (e.clientY - rect.top) * ky;   // пиксели уменьшенного кадра
  point = [px / frameScale, py / frameScale];                                   // -> исходный кадр
  $("sel-point").value = `${point[0].toFixed(0)}, ${point[1].toFixed(0)}`;
  const m = $("marker");
  m.style.left = (e.clientX - rect.left) + "px"; m.style.top = (e.clientY - rect.top) + "px";
  m.classList.remove("hidden");
  updateStart();
};

// ---------- опрос об игроке ----------
let profileOptions = { positions: {}, game_formats: {} };
const typicalHeight = (age) => {
  const t = { 5: 1.10, 6: 1.16, 7: 1.22, 8: 1.28, 9: 1.33, 10: 1.38, 11: 1.44, 12: 1.50, 13: 1.56, 14: 1.63, 15: 1.69, 16: 1.73, 17: 1.75, 18: 1.77 };
  return age ? t[Math.round(Math.min(Math.max(age, 5), 18))] : null;
};

async function loadProfileOptions() {
  profileOptions = await api("/api/profile/options");
  for (const [sel, dict] of [["q-position", profileOptions.positions], ["ai-position", profileOptions.positions],
                             ["q-format", profileOptions.game_formats], ["ai-format", profileOptions.game_formats]]) {
    for (const [k, label] of Object.entries(dict)) {
      const o = document.createElement("option"); o.value = k; o.textContent = label; $(sel).appendChild(o);
    }
  }
}

function surveyMissing() {
  const miss = [];
  if (!(parseFloat($("q-age").value) >= 4)) miss.push("возраст");
  if (!$("q-position").value) miss.push("позицию");
  if (!$("q-format").value) miss.push("формат игры");
  return miss;
}

function updateStart() {
  const miss = surveyMissing();
  for (const [id, key] of [["q-age", "возраст"], ["q-position", "позицию"], ["q-format", "формат игры"]])
    $(id).parentElement.classList.toggle("missing", miss.includes(key));
  $("q-position-note-wrap").classList.toggle("hidden", $("q-position").value !== "other");
  $("q-format-note-wrap").classList.toggle("hidden", $("q-format").value !== "other");
  const age = parseFloat($("q-age").value);
  $("q-height").placeholder = typicalHeight(age) ? `по возрасту ≈ ${typicalHeight(age).toFixed(2)}` : "если не знаете — по возрасту";
  $("start-btn").disabled = !point || miss.length > 0;
  $("start-msg").textContent = !point ? "кликните по игроку на кадре" : miss.length ? "укажите " + miss.join(", ") : "";
}
for (const id of ["q-age", "q-position", "q-format", "q-height"]) { $(id).oninput = updateStart; $(id).onchange = updateStart; }

function surveyValues() {
  return {
    age: parseFloat($("q-age").value) || null,
    position: $("q-position").value || null, position_note: $("q-position-note").value.trim() || null,
    game_format: $("q-format").value || null, format_note: $("q-format-note").value.trim() || null,
    height_m: parseFloat($("q-height").value) || null,
  };
}

$("start-btn").onclick = async () => {
  if (!currentVideo || !point || surveyMissing().length) { updateStart(); return; }
  const tm = $("opt-tonemap").value;
  const body = {
    video_id: currentVideo.id,
    init: { t_sec: parseFloat($("sel-time").value) - (parseFloat($("opt-start").value) || 0), point, number: $("sel-number").value.trim() || null },
    options: {
      width: parseInt($("opt-width").value) || 0, weights: $("opt-weights").value.trim() || "yolov8n.pt",
      imgsz: parseInt($("opt-imgsz").value) || 1280, device: $("opt-device").value.trim() || null,
      encoder: $("opt-encoder").value, ocr: $("opt-ocr").value, team: $("opt-team").checked,
      tonemap: tm === "" ? null : tm === "true", start_sec: parseFloat($("opt-start").value) || 0,
      max_frames: parseInt($("opt-maxframes").value) || null, render: $("opt-render").checked,
      refine: $("opt-refine").checked,
      ball_tiles: $("opt-ball").checked,
    },
    player: surveyValues(),
  };
  if (body.init.t_sec < 0) { $("start-msg").textContent = "Момент выбора раньше начала фрагмента"; return; }
  const w = (currentVideo.info && currentVideo.info.width) || 1280;
  const same = existingJobs.find((x) => x.status === "done" && x.init.point && Math.abs((x.init.t_sec || 0) - body.init.t_sec) < 0.5
    && Math.hypot(x.init.point[0] - point[0], x.init.point[1] - point[1]) < 0.04 * w);
  if (same && confirm("Этого игрока в этом ролике уже отслеживали — открыть готовый результат вместо нового прогона?")) {
    openJob(same.id); return;
  }
  $("start-btn").disabled = true; $("start-msg").textContent = "запуск…";
  try {
    const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $("start-msg").textContent = "задача " + job.id;
    await loadJobs();
    openJob(job.id);
  } catch (err) {
    $("start-msg").textContent = "ошибка: " + err.message;
  } finally { updateStart(); }
};

// ---------- задачи ----------
async function loadJobs() {
  const jobs = await api("/api/jobs");
  const ul = $("job-list");
  ul.innerHTML = "";
  for (const j of jobs) {
    const li = document.createElement("li");
    const pct = j.total ? Math.round(100 * j.progress / j.total) : 0;
    li.innerHTML = `<span class="name">${j.id} <span class="badge ${j.status}">${j.status}</span></span><span class="meta">${j.progress}${j.total ? "/" + j.total + " · " + pct + "%" : ""} · ${j.state_now}${j.summary ? " · отслежено " + Math.round(100 * j.summary.tracked_share) + "%" : ""}</span>`;
    li.onclick = () => openJob(j.id);
    if (currentJob && currentJob.id === j.id) li.classList.add("selected");
    ul.appendChild(li);
  }
  if (jobs.some((j) => j.status === "running" || j.status === "queued") && !pollTimer) startPolling();
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(async () => {
    await loadJobs();
    if (currentJob) await refreshJob(false);
  }, 1500);
}
function stopPolling() { if (pollTimer) clearInterval(pollTimer); pollTimer = null; }

async function openJob(id) {
  currentJob = { id };
  wasBallBusy = false; wasRegBusy = false;
  timelineFrames = null;
  $("job-card").classList.remove("hidden");
  $("result-body").classList.add("hidden");
  await refreshJob(true);
  loadJobs();
}

async function refreshJob(force) {
  const j = await api(`/api/jobs/${currentJob.id}`);
  const wasDone = currentJob.status === "done";
  currentJob = j;
  $("job-title").textContent = `${j.id} · ролик ${j.video_id}`;
  const st = $("job-status"); st.textContent = j.status; st.className = "badge " + j.status;
  const pct = j.total ? Math.min(100, 100 * j.progress / j.total) : (j.status === "done" ? 100 : 0);
  $("job-bar").style.width = pct + "%";
  const stage = j.stage === "rendering" ? " · пишется видео" : "";
  const ballBusy = j.status === "done" && ["queued", "running", "waiting"].includes(j.ball_stage);
  const ballInfo = j.ball_stage === "queued" ? " · мяч: в очереди"
    : j.ball_stage === "waiting" ? ` · мяч: пауза ${j.ball_progress}/${j.ball_total || "?"} — идёт слежение другой задачи`
    : j.ball_stage === "running" ? ` · мяч ищется в фоне: ${j.ball_progress}/${j.ball_total || "?"} кадров`
    : j.ball_stage === "failed" ? " · поиск мяча не удался" : "";
  const rev = j.revision ? ` · перестроено ${j.revision}×` : "";
  $("job-progress").textContent = `${j.progress}${j.total ? "/" + j.total : ""} кадров · ${j.fps_proc} кадр/с · ${j.state_now}${stage}${rev}${ballInfo}`;
  $("cancel-btn").classList.toggle("hidden", !(j.status === "running" || j.status === "queued"));
  $("delete-btn").classList.toggle("hidden", j.status === "running" || j.status === "queued");
  $("job-error").classList.toggle("hidden", !j.error);
  $("job-error").textContent = j.error || "";
  renderEvents(j.events);
  // результат этапа 1 показывается сразу, даже если мяч ещё в очереди или ищется в фоне
  if (j.status === "done" && (force || !wasDone)) await renderResult(j);
  // этап 2 (мяч) закончился — доска и метрики пересчитаны на сервере, перечитываем
  else if (wasBallBusy && !ballBusy && j.status === "done") { loadBoard(j); loadPlayerMetrics(j); }
  wasBallBusy = ballBusy;
  // точная привязка кадров досчиталась — линии поля и положения по разметке уточнились
  const regBusy = j.status === "done" && ["queued", "running"].includes(j.reg_stage);
  if (wasRegBusy && !regBusy && j.status === "done") {
    try { vboxes = await api(`/api/jobs/${j.id}/tracks`); } catch (_) {}
    fieldLinesCache = null; fieldEd.occ = null; fieldEd.occFrame = null;
    if (savedMarks().length) { loadBoard(j); loadPlayerMetrics(j); renderFieldCheck(); }
    if (!$("field-editor").classList.contains("hidden")) loadOccupancy();
    redrawAll();
  }
  if (regBusy || wasRegBusy) renderFieldMarks();
  wasRegBusy = regBusy;
  if (j.status === "running" || j.status === "queued" || ballBusy || regBusy) { if (!pollTimer) startPolling(); }
  else {
    const anyRunning = (await api("/api/jobs")).some((x) => x.status === "running" || x.status === "queued"
      || ["queued", "running", "waiting"].includes(x.ball_stage) || ["queued", "running"].includes(x.reg_stage));
    if (!anyRunning) stopPolling();
  }
}

function renderEvents(events) {
  const ul = $("events"); ul.innerHTML = "";
  for (const e of events) {
    const li = document.createElement("li");
    li.textContent = `кадр ${e.frame} · ${fmtT(e.time)} · ${e.event}${e.track_id != null ? " · трек " + e.track_id : ""}`;
    li.onclick = () => seek(e.time);
    li.style.cursor = "pointer";
    ul.appendChild(li);
  }
}

function seek(t) { const v = $("player"); if (v.src) { v.currentTime = t; v.play().catch(() => {}); } }

// переход на кадр n и пауза: время — середина кадра, чтобы кадр определялся однозначно (floor(t·fps))
function seekFrame(n) {
  const v = $("player");
  if (!v.src) return;
  v.pause();
  v.currentTime = (Math.max(n, 0) + 0.3) / jobFps();
}

async function renderResult(j) {
  $("result-body").classList.remove("hidden");
  const base = `/api/jobs/${j.id}/files/`;
  const v = $("player");
  selectedId = null; renderPick();
  vboxes = null;
  try { vboxes = await api(`/api/jobs/${j.id}/tracks`); } catch (_) { vboxes = null; }
  // чистая копия кадров + рамки слоем поверх (цвета меняются от пометок); без неё — видео с впечатанной разметкой
  const clean = vboxes && vboxes.clean_video;
  const videoUrl = base + (clean ? "frames.mp4" : "annotated.mp4") + "?r=" + (j.revision || 0);
  $("video-wrap").classList.toggle("hidden", !(clean || j.options.render));
  if (clean || j.options.render) { if (!v.src.endsWith(videoUrl)) v.src = videoUrl; }
  $("overlay").classList.toggle("hidden", !clean);
  $("overlay-click").classList.toggle("hidden", !clean);
  document.querySelector(".vlegend").classList.toggle("hidden", !clean);
  kv($("summary"), j.summary || {});
  renderMetrics(j.metrics || {});
  kv($("evaluation"), j.evaluation || {});
  $("downloads").innerHTML = ["player_metrics.json", "track.json", "track.csv", "events.log", "metrics.json", "frame_stats.csv", "errors.md", "errors.jsonl", "evaluation.json", "annotated.mp4"]
    .map((n) => `<a href="${base}${n}" download>${n}</a>`).join("");
  await Promise.all([renderTimeline(j), renderErrors(j)]);
  renderCorrections(j);
  loadPitch(j);
  renderFieldCheck();
  loadBoard(j);
  loadPlayerMetrics(j);
  loadAnalysis(j);
}

function kv(table, obj) {
  table.innerHTML = "";
  for (const [k, val] of Object.entries(obj)) {
    if (val && typeof val === "object") continue;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${k}</td><td>${val == null ? "—" : val}</td>`;
    table.appendChild(tr);
  }
}

function renderMetrics(m) {
  const root = $("metrics"); root.innerHTML = "";
  for (const group of ["detection", "target", "tracks", "ocr", "speed"]) {
    if (!m[group]) continue;
    const div = document.createElement("div");
    div.innerHTML = `<h3>${group}</h3>`;
    const t = document.createElement("table"); t.className = "kv";
    kv(t, m[group]);
    if (m[group].events) { const tr = document.createElement("tr"); tr.innerHTML = `<td>events</td><td>${Object.entries(m[group].events).map(([k, v]) => k + ": " + v).join(", ")}</td>`; t.appendChild(tr); }
    div.appendChild(t); root.appendChild(div);
  }
}

async function renderErrors(j) {
  const root = $("errors"); root.innerHTML = "";
  const counts = (j.metrics && j.metrics.errors && j.metrics.errors.by_kind) || {};
  $("error-counts").textContent = Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join(" · ");
  try {
    const r = await fetch(`/api/jobs/${j.id}/files/errors.jsonl`);
    if (!r.ok) { root.textContent = "журнал ещё не создан"; return; }
    const lines = (await r.text()).split("\n").filter((l) => l.trim());
    if (!lines.length) { root.textContent = "ошибок не зафиксировано"; return; }
    for (const l of lines.slice(0, 500)) {
      const e = JSON.parse(l);
      const div = document.createElement("div");
      div.className = "e " + e.severity;
      const run = e.details && e.details.run_length ? ` (серия ${e.details.run_length} кадров, до ${e.details.last_frame})` : "";
      const cands = e.details && e.details.candidates ? " · кандидаты " + JSON.stringify(e.details.candidates) : "";
      div.innerHTML = `<span class="meta">кадр ${e.frame} · ${fmtT(e.time)}</span> <span class="k">${e.kind}</span> ${e.message}${run}${cands}`;
      div.onclick = () => seek(e.time); div.style.cursor = "pointer";
      root.appendChild(div);
    }
    if (lines.length > 500) root.insertAdjacentHTML("beforeend", `<div class="meta">… ещё ${lines.length - 500} в errors.jsonl</div>`);
  } catch (err) { root.textContent = "не удалось прочитать журнал: " + err.message; }
}

async function renderTimeline(j) {
  const canvas = $("timeline");
  const data = await api(`/api/jobs/${j.id}/frames`);
  timelineFrames = data;
  drawTimeline();
  canvas.onclick = (e) => {
    const rect = canvas.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width * data.total;
    // на ленте несколько кадров на пиксель: если рядом с точкой клика есть потеря цели или неоднозначность,
    // переходим точно на первый такой кадр, а не на соседний «зелёный»
    const span = Math.max(2, Math.ceil(data.total / rect.width * 2));
    let idx = Math.min(Math.floor(pos), data.total - 1);
    const bad = (f) => f && (f.state === "lost" || f.ambiguous);
    if (!bad(data.frames[idx])) {
      let best = null;
      for (let i = Math.max(0, idx - span); i <= Math.min(data.total - 1, idx + span); i++) {
        if (bad(data.frames[i]) && (best === null || Math.abs(i - pos) < Math.abs(best - pos))) best = i;
      }
      if (best !== null) {
        idx = best;
        while (idx > 0 && bad(data.frames[idx - 1]) && best - idx < span) idx--;     // к началу красного участка
      }
    }
    seekFrame(idx);            // на паузе: по ленте ищут кадр, чтобы выбрать или поправить игрока
  };
}
function drawTimeline() {
  if (!timelineFrames) return;
  const canvas = $("timeline"), ctx = canvas.getContext("2d");
  canvas.width = canvas.clientWidth || 800;
  const n = timelineFrames.total, W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#e5e7eb"; ctx.fillRect(0, 0, W, H);
  for (const f of timelineFrames.frames) {
    const x0 = Math.floor(f.frame / n * W), x1 = Math.max(x0 + 1, Math.ceil((f.frame + 1) / n * W));
    ctx.fillStyle = STATE_COLOR[f.ambiguous ? "ambiguous" : f.state] || "#999";
    ctx.fillRect(x0, 0, x1 - x0, H - 8);
    if (f.event) { ctx.fillStyle = "#111"; ctx.fillRect(x0, H - 8, Math.max(2, x1 - x0), 8); }
  }
  for (const c of (currentJob && currentJob.corrections) || []) {
    const x = Math.floor(c.frame / n * W);
    ctx.fillStyle = "#2563eb"; ctx.fillRect(x - 1, 0, 3, H);
  }
}
window.addEventListener("resize", drawTimeline);
$("player").addEventListener("timeupdate", () => {
  if (!timelineFrames) return;
  drawTimeline();
  const canvas = $("timeline"), ctx = canvas.getContext("2d");
  const x = $("player").currentTime * timelineFrames.meta.fps / timelineFrames.total * canvas.width;
  ctx.fillStyle = "#fff"; ctx.fillRect(x - 1, 0, 2, canvas.height);
});

$("cancel-btn").onclick = async () => { if (currentJob) { await api(`/api/jobs/${currentJob.id}/cancel`, { method: "POST" }); refreshJob(true); } };
$("delete-btn").onclick = async () => {
  if (!currentJob || !confirm("Удалить задачу и её результаты?")) return;
  await api(`/api/jobs/${currentJob.id}`, { method: "DELETE" });
  currentJob = null; $("job-card").classList.add("hidden"); loadJobs();
};
$("truth-form").onsubmit = async (e) => {
  e.preventDefault();
  if (!currentJob) return;
  const fd = new FormData(); fd.append("file", $("truth-file").files[0]);
  try {
    const ev = await api(`/api/jobs/${currentJob.id}/truth`, { method: "POST", body: fd });
    kv($("evaluation"), ev);
    await refreshJob(true);
  } catch (err) { alert("Оценка не удалась: " + err.message); }
};

loadVideos(); loadJobs(); loadProfileOptions();

// ---------- поправки цели ----------
const fix = { frame: 0, scale: 1, img: null, boxes: [], target: null, point: null };

function jobFps() {
  return (timelineFrames && timelineFrames.meta && timelineFrames.meta.fps) || (currentJob && currentJob.fps) || 30;
}

function renderCorrections(j) {
  const list = j.corrections || [];
  $("fix-list-wrap").classList.toggle("hidden", !list.length);
  const ul = $("corrections"); ul.innerHTML = "";
  for (const c of list) {
    const li = document.createElement("li");
    const what = c.absent ? "цели нет в кадре" : `цель в точке ${c.point.map((v) => v.toFixed(0)).join(", ")}`;
    li.innerHTML = `<a>кадр ${c.frame} · ${fmtT(c.frame / jobFps())}</a><span>${what}</span><button class="secondary" title="Удалить поправку и перестроить">×</button>`;
    li.querySelector("a").onclick = () => seek(c.frame / jobFps());
    li.querySelector("button").onclick = async () => {
      if (!confirm(`Удалить поправку на кадре ${c.frame} и перестроить слежение?`)) return;
      try {
        await api(`/api/jobs/${j.id}/corrections/${c.frame}`, { method: "DELETE" });
        $("result-body").classList.add("hidden");
        await refreshJob(true); startPolling();
      } catch (err) { alert("Не удалось: " + err.message); }
    };
    ul.appendChild(li);
  }
}

async function openFix(frame) {
  if (!currentJob) return;
  const total = (timelineFrames && timelineFrames.total) || currentJob.total || 1;
  fix.frame = Math.max(0, Math.min(frame, total - 1));
  fix.point = null;
  $("fix-panel").classList.remove("hidden");
  $("fix-here").disabled = true;
  $("fix-msg").textContent = "загрузка кадра…";
  $("fix-frame-label").textContent = `кадр ${fix.frame} · ${fmtT(fix.frame / jobFps())}`;
  const [imgResp, det] = await Promise.all([
    fetch(`/api/jobs/${currentJob.id}/frame/${fix.frame}`),
    api(`/api/jobs/${currentJob.id}/detections/${fix.frame}`),
  ]);
  if (!imgResp.ok) { $("fix-msg").textContent = "кадр не прочитан: " + (await imgResp.text()); return; }
  fix.scale = parseFloat(imgResp.headers.get("X-Scale") || "1");
  fix.boxes = det.boxes || [];
  fix.target = det.target;
  const img = new Image();
  img.onload = () => { fix.img = img; drawFix(); applyZoom(true); $("fix-msg").textContent = ""; };
  img.src = URL.createObjectURL(await imgResp.blob());
  const t = det.target;
  $("fix-state").textContent = t ? `сейчас: ${t.state}${t.track_id != null ? " · трек " + t.track_id : ""}` : "";
  if (det.correction) $("fix-state").textContent += " · на этом кадре уже есть поправка (новая заменит её)";
}

function drawFix() {
  const c = $("fix-canvas"), ctx = c.getContext("2d"), img = fix.img;
  if (!img) return;
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  const k = fix.scale, lw = Math.max(1, Math.round(img.naturalWidth / 640));
  const chosen = chosenBox();
  for (const b of fix.boxes) {
    ctx.strokeStyle = b === chosen ? "#22d3ee" : "rgba(220,220,220,.85)";
    ctx.lineWidth = b === chosen ? 3 * lw : lw;
    ctx.strokeRect(b[0] * k, b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k);
  }
  const t = fix.target;
  if (t && t.box) {
    ctx.strokeStyle = "#22a06b"; ctx.lineWidth = 2 * lw; ctx.setLineDash([6 * lw, 4 * lw]);
    ctx.strokeRect(t.box[0] * k, t.box[1] * k, (t.box[2] - t.box[0]) * k, (t.box[3] - t.box[1]) * k);
    ctx.setLineDash([]);
  }
  if (fix.point) {
    ctx.strokeStyle = "#22d3ee"; ctx.lineWidth = 3 * lw;
    ctx.beginPath(); ctx.arc(fix.point[0] * k, fix.point[1] * k, 9 * lw, 0, 2 * Math.PI); ctx.stroke();
  }
}

function applyZoom(centerOnTarget) {
  const c = $("fix-canvas"), wrap = $("fix-wrap"), z = parseFloat($("fix-zoom").value) || 1;
  if (!fix.img) return;
  c.style.width = (wrap.clientWidth * z) + "px";
  if (!centerOnTarget) return;
  // прокрутка к цели (или к последней рамке цели) — мелких игроков так проще выбирать
  let t = fix.target && fix.target.box;
  if (!t && timelineFrames) {           // цель потеряна — к последнему месту, где она была
    for (let i = Math.min(fix.frame, timelineFrames.frames.length - 1); i >= 0 && !t; i--) t = timelineFrames.frames[i].box;
  }
  if (!t) return;
  const k = c.clientWidth / c.width * fix.scale;
  wrap.scrollLeft = ((t[0] + t[2]) / 2) * k - wrap.clientWidth / 2;
  wrap.scrollTop = ((t[1] + t[3]) / 2) * k - wrap.clientHeight / 2;
}
$("fix-zoom").onchange = () => applyZoom(true);

function chosenBox() {
  // как на сервере: из рамок, куда попала точка, — та, чей центр ближе к клику
  if (!fix.point) return null;
  const [x, y] = fix.point;
  let best = null, bestR = Infinity;
  for (const b of fix.boxes) {
    if (x < b[0] || x > b[2] || y < b[1] || y > b[3]) continue;
    const dx = (x - (b[0] + b[2]) / 2) / Math.max(b[2] - b[0], 1), dy = (y - (b[1] + b[3]) / 2) / Math.max(b[3] - b[1], 1);
    const r = Math.sqrt(dx * dx + 0.25 * dy * dy);
    if (r < bestR) { bestR = r; best = b; }
  }
  return best;
}

$("fix-canvas").onclick = (e) => {
  const c = e.target, rect = c.getBoundingClientRect();
  if (!fix.img || !rect.width) return;
  const px = (e.clientX - rect.left) * c.width / rect.width, py = (e.clientY - rect.top) * c.height / rect.height;
  fix.point = [px / fix.scale, py / fix.scale];   // -> координаты исходного кадра
  drawFix();
  const hit = chosenBox();
  $("fix-here").disabled = !hit;
  $("fix-msg").textContent = hit ? "" : "в этой точке игрок не найден — кликните по рамке";
};

$("fix-open").onclick = () => {
  const v = $("player");
  if (!v.paused) v.pause();
  openFix(Math.floor((v.currentTime || 0) * jobFps() + 0.01));
};
$("fix-close").onclick = () => { $("fix-panel").classList.add("hidden"); };
for (const b of document.querySelectorAll("#fix-panel .step")) {
  b.onclick = () => { const f = fix.frame + parseInt(b.dataset.step); seek(Math.max(f, 0) / jobFps()); $("player").pause(); openFix(f); };
}

async function submitFix(body) {
  $("fix-here").disabled = true; $("fix-absent").disabled = true;
  $("fix-msg").textContent = "перестроение…";
  try {
    await api(`/api/jobs/${currentJob.id}/corrections`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $("fix-panel").classList.add("hidden");
    $("result-body").classList.add("hidden");
    await refreshJob(true); startPolling();
  } catch (err) {
    $("fix-msg").textContent = "ошибка: " + err.message;
  } finally { $("fix-absent").disabled = false; }
}
$("fix-here").onclick = () => { if (fix.point) submitFix({ frame: fix.frame, point: fix.point }); };
$("fix-absent").onclick = () => submitFix({ frame: fix.frame, absent: true });

// ---------- метрики игрока ----------
let playerMetrics = null;
const fmtMs = (v) => (v == null ? "—" : `${v.toFixed(1)} м/с · ${(v * 3.6).toFixed(0)} км/ч`);

async function loadPlayerMetrics(j) {
  $("pm-msg").textContent = "";
  $("rerun-btn").classList.add("hidden");
  if (j.player) {
    $("pm-height").value = j.player.height_m || "";
    $("pm-height").placeholder = typicalHeight(j.player.age) ? `по возрасту ≈ ${typicalHeight(j.player.age).toFixed(2)}` : "1.50";
    $("pm-fov").value = String(Math.round(j.player.hfov_deg || 70));
  }
  try {
    playerMetrics = await api(`/api/jobs/${j.id}/player-metrics`);
    renderPlayerMetrics(playerMetrics);
  } catch (err) {
    $("pm-body").classList.add("hidden");
    $("pm-msg").textContent = err.message;
    if (/перезапустите|Пересчитать/.test(err.message)) $("rerun-btn").classList.remove("hidden");
  }
}

function tile(value, label) { return `<div class="tile"><b>${value}</b><span>${label}</span></div>`; }

function episodeList(ul, eps, fmt) {
  ul.innerHTML = eps.length ? "" : "<li>нет</li>";
  for (const e of eps) {
    const li = document.createElement("li");
    li.innerHTML = `<a class="ts">${fmtT(e.start_s)}–${fmtT(e.end_s)}</a> ${fmt(e)}`;
    li.querySelector("a").onclick = () => seek(e.start_s);
    ul.appendChild(li);
  }
}

function renderPlayerMetrics(m) {
  $("pm-body").classList.remove("hidden");
  const p = m.presence, mv = m.movement, inv = m.involvement, cf = m.config || {};
  const who = [cf.age ? `${cf.age} лет` : null, cf.position, cf.game_format,
    `рост ${cf.player_height_m} м${cf.height_source && cf.height_source !== "указан" ? " (" + cf.height_source + ")" : ""}`].filter(Boolean);
  $("pm-msg").textContent = who.join(" · ");
  $("pm-tiles").innerHTML = [
    tile(`${p.tracked_s} с`, `в кадре (${Math.round(p.tracked_share * 100)}% ролика, появлений ${p.appearances})`),
    tile(`${mv.distance_m} м`, `дистанция за ${mv.measured_s} с измерений`),
    tile(mv.distance_per_min_m == null ? "—" : `${mv.distance_per_min_m} м/мин`, "темп"),
    tile(fmtMs(mv.avg_speed_ms), "средняя скорость"),
    tile(fmtMs(mv.max_speed_ms), "максимальная"),
    tile(mv.sprints.length, "спринтов"),
    tile(`${mv.accelerations} / ${mv.decelerations}`, "ускорений / торможений"),
    tile(`${inv.close_contact_s} с`, `единоборства (${inv.close_contacts.length})`),
    tile(m.ball && m.ball.pending ? "…" : `${inv.ball_near_s} с`,
      m.ball && m.ball.pending ? "мяч рядом — мяч ещё ищется" : `мяч рядом (мяч виден на ${Math.round(inv.ball_detected_share * 100)}% кадров)`),
  ].join("");
  const zones = $("pm-zones"); zones.innerHTML = "";
  for (const z of mv.zones) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${z.zone} (${z.from_ms}${z.to_ms == null ? "+" : "–" + z.to_ms} м/с)</td><td>${z.seconds} с · ${z.distance_m} м</td>`;
    zones.appendChild(tr);
  }
  kv($("pm-thirds"), Object.fromEntries(Object.entries((m.position && m.position.thirds) || {}).map(([k, v]) => [k, Math.round(v * 100) + "%"])));
  episodeList($("pm-sprints"), mv.sprints, (e) => `пик ${e.peak.toFixed(1)} м/с · ${e.distance_m} м${e.direction ? " · " + e.direction : ""}`);
  episodeList($("pm-contacts"), inv.close_contacts, (e) => `до ${e.peak.toFixed(1)} м`);
  episodeList($("pm-ball"), inv.ball_episodes, (e) => `пробежал ${e.distance_m} м`);
  $("pm-notes").innerHTML = (m.quality.notes || []).map((n) => `<li>${n}</li>`).join("");
  drawSpeed(m); drawHeat(m); renderPitchMetrics(m); renderGame(m);
}

function renderGame(m) {
  const g = m.game, b = m.ball;
  $("pm-game").classList.toggle("hidden", !g && !b);
  if (!g && !b) return;
  const rows = {};
  if (b && b.pending) rows["игровой мяч"] = "ищется в фоне…";
  else if (b) {
    rows["игровой мяч найден"] = Math.round(b.detected_share * 100) + "% кадров";
    rows["с оценкой в коротких разрывах"] = Math.round(b.known_share * 100) + "% кадров";
  }
  if (g) {
    const pe = g.possession_estimate || {};
    const lbl = { own: "у своих", opp: "у соперника", loose: "ничей / в борьбе" };
    for (const [k, v] of Object.entries(pe)) rows[`мяч ${lbl[k] || k} (когда мяч известен)`] = Math.round(v * 100) + "%";
    if (g.possession_spells) rows["отрезков владения свои / соперник"] = `${g.possession_spells.own || 0} / ${g.possession_spells.opp || 0}`;
    if (g.focus_dist_to_ball_m) {
      rows["расстояние игрока до мяча, медиана"] = g.focus_dist_to_ball_m.median + " м";
      rows["игрок в 5 м от мяча"] = Math.round(g.focus_dist_to_ball_m.share_within_5m * 100) + "% времени";
    }
    if (g.focus_last_line_share != null) rows["игрок — последний полевой своей команды"] = Math.round(g.focus_last_line_share * 100) + "% времени";
  }
  kv($("pm-game-table"), rows);
}

function drawSpeed(m) {
  const c = $("pm-speed"), ctx = c.getContext("2d");
  c.width = c.clientWidth || 800;
  const W = c.width, H = c.height, tl = m.timeline, n = Math.max(tl.length, 1);
  const vmax = Math.max(6, ...tl.map((r) => r.speed_max_ms || 0));
  ctx.clearRect(0, 0, W, H);
  const zoneColor = { "шаг": "#e5e7eb", "трусца": "#bbf7d0", "бег": "#fde68a", "спринт": "#fecaca" };
  tl.forEach((r, i) => {
    const x = i / n * W, w = Math.ceil(W / n);
    if (!r.in_view) { ctx.fillStyle = "#f3f4f6"; ctx.fillRect(x, 0, w, H); return; }
    if (r.zone) { ctx.fillStyle = zoneColor[r.zone]; ctx.fillRect(x, H - (r.speed_ms / vmax) * (H - 14), w, (r.speed_ms / vmax) * (H - 14)); }
    if (r.ball_near) { ctx.fillStyle = "#111827"; ctx.fillRect(x, H - 4, w, 4); }
  });
  ctx.strokeStyle = "#2563eb"; ctx.lineWidth = 2; ctx.beginPath();
  let started = false;
  tl.forEach((r, i) => {
    const x = (i + 0.5) / n * W;
    if (r.speed_max_ms == null) { started = false; return; }
    const y = H - (r.speed_max_ms / vmax) * (H - 14);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#6b7280"; ctx.font = "11px system-ui";
  ctx.fillText(`скорость по секундам (линия — максимум), до ${vmax.toFixed(0)} м/с · чёрная полоса — мяч рядом`, 6, 12);
  c.onclick = (e) => { const r = c.getBoundingClientRect(); seek(Math.floor((e.clientX - r.left) / r.width * n)); };
}

function drawHeat(m) {
  const c = $("pm-heat"), ctx = c.getContext("2d"), heat = m.position && m.position.heatmap;
  ctx.fillStyle = "#0f3d1f"; ctx.fillRect(0, 0, c.width, c.height);
  if (!heat) return;
  const [bx, bz] = heat.bins, grid = heat.seconds, max = Math.max(...grid.flat(), 1e-6);
  const cw = c.width / bx, ch = c.height / bz;
  for (let z = 0; z < bz; z++) for (let x = 0; x < bx; x++) {
    const v = grid[z][x] / max;
    if (v <= 0) continue;
    ctx.fillStyle = `rgba(250, ${Math.round(220 - 170 * v)}, 40, ${0.15 + 0.8 * v})`;
    ctx.fillRect(x * cw, (bz - 1 - z) * ch, cw, ch);    // дальше от камеры — выше
  }
  ctx.fillStyle = "rgba(255,255,255,.8)"; ctx.font = "11px system-ui";
  ctx.fillText("↑ дальше от камеры", 6, 14); ctx.fillText("← слева · справа →", 6, c.height - 6);
}

$("pm-recalc").onclick = async () => {
  if (!currentJob) return;
  $("pm-msg").textContent = "пересчёт…";
  try {
    playerMetrics = await api(`/api/jobs/${currentJob.id}/player-metrics`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ height_m: parseFloat($("pm-height").value), hfov_deg: parseFloat($("pm-fov").value) }) });
    renderPlayerMetrics(playerMetrics); $("pm-msg").textContent = "";
  } catch (err) { $("pm-msg").textContent = err.message; }
};
$("rerun-btn").onclick = async () => {
  if (!currentJob || !confirm("Прогнать ролик заново (детектор, ~как первый запуск), с сохранением данных кадров для метрик и поправок?")) return;
  await api(`/api/jobs/${currentJob.id}/rerun`, { method: "POST" });
  $("result-body").classList.add("hidden");
  await refreshJob(true); startPolling();
};

// ---------- ИИ-разбор ----------
function mdToHtml(md) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    // [мм:сс] и диапазоны [мм:сс–мм:сс]: переход — на начало
    .replace(/\[(\d{1,2}):(\d{2})((?:\s*[–-]\s*\d{1,2}:\d{2})?)\]/g,
      (_, mm, ss, rest) => `<a class="ts" data-t="${parseInt(mm) * 60 + parseInt(ss)}">[${mm}:${ss}${rest}]</a>`);
  const out = []; let list = false;
  for (const line of md.split("\n")) {
    const t = line.trim();
    const li = t.match(/^([-*]|\d+\.)\s+(.*)/);
    if (li) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${inline(li[2])}</li>`); continue; }
    if (list) { out.push("</ul>"); list = false; }
    if (/^#{1,4}\s/.test(t)) out.push(`<h2>${inline(t.replace(/^#+\s/, ""))}</h2>`);
    else if (t) out.push(`<p>${inline(t)}</p>`);
  }
  if (list) out.push("</ul>");
  return out.join("");
}

function renderChat(messages) {
  const box = $("ai-chat"); box.innerHTML = "";
  for (const m of messages) {
    const div = document.createElement("div");
    div.className = "m " + m.role;
    div.innerHTML = m.role === "assistant" ? mdToHtml(m.content) : mdToHtml(m.content);
    box.appendChild(div);
  }
  for (const a of box.querySelectorAll("a.ts")) a.onclick = () => seek(parseInt(a.dataset.t));
  $("ai-form").classList.toggle("hidden", !messages.length);
  $("ai-run").textContent = messages.length ? "Новый разбор" : "Сделать разбор";
}

// ---------- схема поля ----------
const pitch = { camera: null, look: [0.5, 0.5], own_goal: null, length_m: 40, width_m: 25, saved: false };
const PITCH_MARGIN = 0.2;   // поле на холсте с полями вокруг — камера обычно вне поля

function pitchXY(canvas, x, y) {      // координаты схемы -> холст
  const mx = canvas.width * PITCH_MARGIN / (1 + 2 * PITCH_MARGIN), my = canvas.height * PITCH_MARGIN / (1 + 2 * PITCH_MARGIN);
  return [mx + x * (canvas.width - 2 * mx), my + y * (canvas.height - 2 * my)];
}
function pitchInv(canvas, cx, cy) {
  const mx = canvas.width * PITCH_MARGIN / (1 + 2 * PITCH_MARGIN), my = canvas.height * PITCH_MARGIN / (1 + 2 * PITCH_MARGIN);
  return [(cx - mx) / (canvas.width - 2 * mx), (cy - my) / (canvas.height - 2 * my)];
}

function drawPitch(canvas, setup, heat) {
  const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height;
  const L = setup.length_m || 40, Wd = setup.width_m || 25;
  ctx.fillStyle = "#14532d"; ctx.fillRect(0, 0, W, H);
  const [x0, y0] = pitchXY(canvas, 0, 0), [x1, y1] = pitchXY(canvas, 1, 1);
  ctx.fillStyle = "#15803d"; ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  if (heat) {
    const [bx, by] = heat.bins, grid = heat.seconds, max = Math.max(...grid.flat(), 1e-6);
    const cw = (x1 - x0) / bx, ch = (y1 - y0) / by;
    for (let j = 0; j < by; j++) for (let i = 0; i < bx; i++) {
      const v = grid[j][i] / max; if (v <= 0) continue;
      ctx.fillStyle = `rgba(250, ${Math.round(220 - 170 * v)}, 40, ${0.15 + 0.75 * v})`;
      ctx.fillRect(x0 + i * cw, y0 + j * ch, cw, ch);
    }
  }
  ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 2;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  const xm = (x0 + x1) / 2, ym = (y0 + y1) / 2, sx = (x1 - x0) / L;
  ctx.beginPath(); ctx.moveTo(xm, y0); ctx.lineTo(xm, y1); ctx.stroke();
  ctx.beginPath(); ctx.arc(xm, ym, Math.min(4.5, L / 8) * sx, 0, 2 * Math.PI); ctx.stroke();
  const boxD = Math.min(L * 0.16, 16.5) * sx, boxW = Math.min(Wd * 0.6, 40) * sx, goalW = Math.min(Wd * 0.2, 7.3) * sx;
  for (const [gx, dir, key] of [[x0, 1, "left"], [x1, -1, "right"]]) {
    ctx.strokeRect(dir > 0 ? gx : gx - boxD, ym - boxW / 2, boxD, boxW);
    ctx.fillStyle = setup.own_goal === key ? "#60a5fa" : setup.own_goal ? "#f87171" : "#e5e7eb";
    ctx.fillRect(dir > 0 ? gx - 8 : gx, ym - goalW / 2, 8, goalW);
    if (setup.own_goal) {
      ctx.font = "bold 12px system-ui"; ctx.fillStyle = "#fff"; ctx.textAlign = dir > 0 ? "left" : "right";
      ctx.fillText(setup.own_goal === key ? "свои ворота" : "чужие ворота", gx + dir * 6, ym - goalW / 2 - 6);
    }
  }
  ctx.textAlign = "left";
  if (setup.own_goal) {
    ctx.fillStyle = "rgba(255,255,255,.9)"; ctx.font = "12px system-ui";
    ctx.fillText(setup.own_goal === "left" ? "атака своей команды →" : "← атака своей команды", xm - 60, y1 + 16);
  }
  if (setup.camera) {
    const [cx, cy] = pitchXY(canvas, ...setup.camera);
    const [lx, ly] = pitchXY(canvas, ...(setup.look || [0.5, 0.5]));
    // угол обзора считается в метрах поля, рисуется в пикселях холста (масштабы по осям разные)
    const pxm = (x1 - x0) / L, pym = (y1 - y0) / Wd;
    const ang = Math.atan2((ly - cy) / pym, (lx - cx) / pxm);
    const half = (parseFloat($("pm-fov").value) || 70) / 2 * Math.PI / 180;
    const reachM = Math.hypot(L, Wd) * 1.5;
    ctx.fillStyle = "rgba(255,255,255,.12)"; ctx.beginPath(); ctx.moveTo(cx, cy);
    for (const a of [ang - half, ang + half]) ctx.lineTo(cx + Math.cos(a) * reachM * pxm, cy + Math.sin(a) * reachM * pym);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = "#fde047"; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(lx, ly); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#fde047"; ctx.beginPath(); ctx.arc(cx, cy, 8, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#111"; ctx.font = "bold 10px system-ui"; ctx.fillText("К", cx - 3.5, cy + 3.5);
    ctx.strokeStyle = "#fde047"; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(lx - 7, ly); ctx.lineTo(lx + 7, ly); ctx.moveTo(lx, ly - 7); ctx.lineTo(lx, ly + 7); ctx.stroke();
  }
}

function pitchReady() { return !!(pitch.camera && pitch.look && pitch.own_goal); }

function refreshPitch() {
  drawPitch($("pitch-canvas"), pitch);
  const miss = [!pitch.camera && "камеру", !pitch.own_goal && "свои ворота"].filter(Boolean);
  $("pitch-msg").textContent = miss.length ? "отметьте " + miss.join(" и ") : pitch.saved ? "схема сохранена" : "не сохранено";
  $("pitch-save").disabled = !pitchReady();
}

function loadPitch(j) {
  const p = (j.player && j.player.pitch) || null;
  const fmt = j.player && j.player.game_format;
  const dims = (profileOptions.fields && profileOptions.fields[fmt]) || [40, 25];
  Object.assign(pitch, { camera: p ? p.camera : null, look: p ? p.look : [0.5, 0.5], own_goal: p ? p.own_goal : null,
    length_m: (p && p.length_m) || dims[0], width_m: (p && p.width_m) || dims[1], saved: !!p });
  $("pitch-len").value = Math.round(pitch.length_m); $("pitch-wid").value = Math.round(pitch.width_m);
  $("pitch-frame").src = `/api/jobs/${j.id}/frame/0?r=${j.revision || 0}`;
  refreshPitch();
  initField(j);
}

$("pitch-canvas").onclick = (e) => {
  const c = e.target, r = c.getBoundingClientRect();
  const [x, y] = pitchInv(c, (e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height);
  const mode = document.querySelector("input[name=pitch-mode]:checked").value;
  const next = { camera: "look", look: "goal", goal: "goal" };
  if (mode === "camera") pitch.camera = [+x.toFixed(3), +y.toFixed(3)];
  else if (mode === "look") pitch.look = [+Math.min(Math.max(x, 0), 1).toFixed(3), +Math.min(Math.max(y, 0), 1).toFixed(3)];
  else pitch.own_goal = x < 0.5 ? "left" : "right";
  pitch.saved = false;
  document.querySelector(`input[name=pitch-mode][value=${next[mode]}]`).checked = true;
  refreshPitch();
};
for (const id of ["pitch-len", "pitch-wid"]) $(id).onchange = () => {
  pitch.length_m = parseFloat($("pitch-len").value) || pitch.length_m; pitch.width_m = parseFloat($("pitch-wid").value) || pitch.width_m;
  pitch.saved = false; refreshPitch();
};
$("pm-fov").addEventListener("change", refreshPitch);
$("pitch-save").onclick = async () => {
  if (!currentJob || !pitchReady()) return;
  $("pitch-msg").textContent = "сохранение…";
  try {
    currentJob = await api(`/api/jobs/${currentJob.id}/player`, { method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pitch: { camera: pitch.camera, look: pitch.look, own_goal: pitch.own_goal, length_m: pitch.length_m, width_m: pitch.width_m } }) });
    pitch.saved = true; refreshPitch();
    loadPlayerMetrics(currentJob); loadAnalysis(currentJob); loadBoard(currentJob);
  } catch (err) { $("pitch-msg").textContent = "ошибка: " + err.message; }
};

function renderPitchMetrics(m) {
  const pt = m.pitch;
  $("pm-pitch").classList.toggle("hidden", !pt);
  if (!pt) return;
  const rows = {};
  for (const [k, v] of Object.entries(pt.thirds)) rows[k] = Math.round(v * 100) + "%";
  for (const [k, v] of Object.entries(pt.channels)) rows[k] = Math.round(v * 100) + "%";
  rows["к чужим воротам / к своим"] = `${pt.towards_opponent_goal_m} м / ${pt.towards_own_goal_m} м`;
  rows["средняя позиция (0 — свои ворота, 1 — чужие)"] = pt.avg_progress;
  kv($("pm-pitch-table"), rows);
  drawPitch($("pm-pitch-heat"), { ...pt.setup }, pt.heatmap);
}

// ---------- ИИ-разбор: сохранённые версии ----------
let aiSessions = [], aiShown = null;

function renderSessions(a, pickId) {
  aiSessions = a.sessions || [];
  const sel = $("ai-version"); sel.innerHTML = "";
  aiSessions.forEach((x, i) => {
    const o = document.createElement("option"); o.value = x.id;
    o.textContent = `№${i + 1} · ${new Date(x.created * 1000).toLocaleString("ru-RU")}${x.stale ? " · по прежней версии слежения" : ""}`;
    sel.appendChild(o);
  });
  $("ai-version-wrap").classList.toggle("hidden", aiSessions.length < 2);
  $("ai-delete").classList.toggle("hidden", !aiSessions.length);
  const show = aiSessions.find((x) => x.id === pickId) || aiSessions[aiSessions.length - 1] || null;
  aiShown = show;
  if (show) sel.value = show.id;
  renderChat(show ? show.messages : []);
  const current = aiSessions.some((x) => !x.stale);
  $("ai-run").textContent = current ? "Новый разбор" : "Сделать разбор";
  $("ai-form").classList.toggle("hidden", !show || show.stale);
  if (show && show.stale) $("ai-msg").textContent = "это разбор по прежней версии слежения (до поправок) — вопросы задаются в новом разборе";
}
$("ai-version").onchange = () => renderSessions({ sessions: aiSessions }, parseInt($("ai-version").value));

async function loadAnalysis(j) {
  try {
    const st = await api("/api/ai/status");
    const p = j.player || {};
    $("ai-age").value = p.age || "";
    $("ai-position").value = p.position || "";
    $("ai-format").value = p.game_format || "";
    $("ai-color").value = p.team_color || "";
    $("ai-focus").value = p.focus || "";
    $("ai-msg").textContent = "";
    renderSessions(await api(`/api/jobs/${j.id}/analysis`));
    const blocked = !st.configured ? "ключ DeepSeek не задан (DEEPSEEK_API_KEY или data/secrets.json)"
      : !p.pitch ? "сначала отметьте и сохраните схему поля выше" : "";
    $("ai-run").disabled = !!blocked;
    if (blocked) $("ai-msg").textContent = blocked;
  } catch (err) { $("ai-msg").textContent = err.message; }
}

function renderChat(messages) {
  const box = $("ai-chat"); box.innerHTML = "";
  for (const m of messages) {
    const div = document.createElement("div");
    div.className = "m " + m.role;
    div.innerHTML = mdToHtml(m.content);
    box.appendChild(div);
  }
  for (const a of box.querySelectorAll("a.ts")) a.onclick = () => seek(parseInt(a.dataset.t));
}

function aiPlayer() {
  // правки здесь сохраняются в опрос задачи и пересчитывают метрики
  const out = { age: parseFloat($("ai-age").value) || null, position: $("ai-position").value || null,
    game_format: $("ai-format").value || null, team_color: $("ai-color").value.trim() || null, focus: $("ai-focus").value.trim() || null };
  const h = parseFloat($("pm-height").value);
  if (h && currentJob && currentJob.player && currentJob.player.height_m !== h) out.height_m = h;
  return out;
}

async function askAI(question, fresh) {
  if (!currentJob) return;
  const hasCurrent = aiSessions.some((x) => !x.stale);
  if (fresh && hasCurrent && !confirm("Разбор по текущему слежению уже сохранён. Сделать ещё один (новый запрос в DeepSeek)? Прежний останется.")) return;
  $("ai-run").disabled = true; $("ai-msg").textContent = "DeepSeek думает (до минуты)…";
  try {
    const body = { question, player: aiPlayer(), new: fresh && hasCurrent };
    if (!fresh && aiShown && !aiShown.stale) body.session_id = aiShown.id;
    const r = await api(`/api/jobs/${currentJob.id}/analysis`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    renderSessions(r, r.session_id);
    $("ai-msg").textContent = r.cached ? "показан сохранённый разбор по этим же данным (без запроса к модели)" : "";
    currentJob = await api(`/api/jobs/${currentJob.id}`);      // опрос мог измениться — метрики пересчитаны
    loadPlayerMetrics(currentJob);
  } catch (err) { $("ai-msg").textContent = "ошибка: " + err.message; }
  finally { $("ai-run").disabled = false; }
}
$("ai-run").onclick = () => askAI(null, true);
$("ai-delete").onclick = async () => {
  if (!currentJob || !aiShown || !confirm("Удалить эту версию разбора?")) return;
  await api(`/api/jobs/${currentJob.id}/analysis?session_id=${aiShown.id}`, { method: "DELETE" });
  renderSessions(await api(`/api/jobs/${currentJob.id}/analysis`));
};
$("ai-form").onsubmit = (e) => { e.preventDefault(); const q = $("ai-question").value.trim(); if (q) { $("ai-question").value = ""; askAI(q, false); } };

// ---------- тактическая доска ----------
let board = null, boardRaf = null;

async function loadBoard(j) {
  board = null;
  $("board-build").classList.add("hidden");
  $("board-msg").textContent = "загрузка доски…";
  try {
    board = await api(`/api/jobs/${j.id}/board`);
    const frames = board.frames;
    $("board-time").max = frames.length ? frames[frames.length - 1].t : 1;
    const b = board.ball || {};
    const notes = [];
    if (board.field.default && !(board.calibration && board.calibration.method === "field")) notes.push("схема поля не отмечена — камера условно у длинной бровки; отметьте схему, чтобы знать свои и чужие ворота");
    const cal = board.calibration;
    if (cal && cal.method === "field") notes.push(`положения на поле — по разметке поля на видео (${(cal.frames || [cal.frame]).length > 1 ? "кадры " + cal.frames.join(", ") : "кадр " + cal.frame})`
      + (cal.registered ? "" : "; точная привязка кадров ещё считается — положения уточнятся")
      + (cal.outside != null ? `; за границей поля ${Math.round(cal.outside * 100)}% положений` : ""));
    if (cal && cal.fitted) notes.push(`схема подогнана по расстановке игроков: глубина ×${cal.kz}, поперёк ×${cal.kx}; `
      + `за полем было ${Math.round(cal.outside_before * 100)}% положений, стало ${Math.round(cal.outside_after * 100)}%`);
    if (board.teams_note) notes.push("команды не назначены: " + board.teams_note);
    if (board.ball_pending) notes.push("мяч ещё ищется в фоне — доска пока без мяча");
    if (b.frames && !board.ball_pending) notes.push(`мяч найден на ${Math.round(b.detected_share * 100)}% кадров, с оценкой — ${Math.round(b.known_share * 100)}%`);
    $("board-msg").textContent = notes.join(" · ");
    renderBoardLegend();
    renderSideline();
    drawBoard(currentTime());
    drawOverlay();
  } catch (err) {
    $("board-msg").textContent = err.message;
    $("board-build").classList.toggle("hidden", !/Построить доску/.test(err.message));
    drawPitch($("board-canvas"), { length_m: 40, width_m: 25 });
  }
}

// человек вне игры на кадре: сначала — ещё не пересчитанные щелчки оператора, затем отрезки с сервера
const pendingMarks = {};            // id трека -> {role, scope, frame}
function isOffAt(id, frame) {
  const p = pendingMarks[id];
  if (p && (p.scope !== "from" || frame >= p.frame)) return p.role === "sideline";
  const m = trackMeta(id);
  if (!m.off) return m.role === "sideline";
  return m.off.some(([a, b]) => a <= frame && frame <= b);
}
function nowFrame() { const v = $("player"); return v && v.src ? videoFrameIndex() : Math.round(currentTime() * jobFps()); }

function renderSideline() {
  const box = $("board-off");
  if (!board) { box.innerHTML = ""; return; }
  const marked = board.roster_mode === "marked";
  $("roster-mode").value = board.roster_mode || "all";
  const tracks = Object.entries(board.tracks || {});
  // при разметке поля на видео: кто почти всё время за его границей — предложение выключить (решает оператор)
  const sug = marked ? [] : tracks.filter(([, m]) => m.suggest === "sideline" && m.role !== "sideline");
  const sugChips = sug.map(([id, m]) => `<span class="chip suggest" data-sid="${id}" title="${m.suggest_reason} — клик выключает">`
    + `<i style="background:${m.color}"></i>№${m.num ?? "?"} · ${m.seconds} с</span>`).join("");
  const sugHtml = sug.length ? `<div><b>Похоже, не играют</b> (стоят за границей отмеченного поля): ${sugChips}`
    + ` <a href="#" id="sug-all">выключить всех</a></div>` : "";
  // метки — только ваши пометки; всё, что выключено или включено следом за ними, — числом
  const mine = tracks.filter(([, m]) => m.manual);
  const chips = mine.map(([id, m]) => {
    const from = (m.off || []).length && m.off[0][0] > 0 && m.role !== "sideline" ? " · с " + fmtT(m.off[0][0] / jobFps()) : "";
    return `<span class="chip" data-id="${id}" title="ваша пометка — клик снимает её"><i style="background:${m.color}"></i>№${m.num ?? "?"}${from}</span>`;
  }).join("");
  const auto = { person: 0, similar: 0 };
  for (const [, m] of tracks) if (!m.manual && auto[m.why] != null && (marked ? m.role !== "sideline" : m.role === "sideline")) auto[m.why]++;
  const autoTxt = auto.person + auto.similar
    ? ` Следом за пометками ${marked ? "включено" : "выключено"} ещё ${auto.person + auto.similar} фрагм.: тот же человек — ${auto.person}, похожи по внешности — ${auto.similar}.` : "";
  const head = marked ? "<b>Отмечены игроками:</b>" : "<b>Выключены вами:</b>";
  const hint = marked
    ? "В игре только отмеченные вами и похожие на них. Поставьте на паузу кадр, где видно побольше игроков, и отметьте на нём ВСЕХ игроков — остальные люди этого кадра станут примерами «не играет»."
    : "На доске все найденные люди. Поставьте на паузу кадр, где видно побольше лишних (тренеры, зрители, запасные), и выключите их всех — человек остаётся выключенным на весь ролик, похожие на него тоже.";
  box.innerHTML = sugHtml + (mine.length ? `${head} ${chips}${autoTxt} ` : "") + `<span class="meta">${hint}</span>`
    + (mine.length ? ` <a href="#" id="roster-clear">снять все пометки</a>` : "");
  for (const chip of box.querySelectorAll(".chip[data-id]")) chip.onclick = () => markPerson(parseInt(chip.dataset.id), "auto");
  for (const chip of box.querySelectorAll(".chip[data-sid]")) chip.onclick = () => markPerson(parseInt(chip.dataset.sid), "sideline");
  const all = $("sug-all");
  if (all) all.onclick = (e) => { e.preventDefault(); for (const [id] of sug) markPerson(parseInt(id), "sideline"); };
  const clear = $("roster-clear");
  if (clear) clear.onclick = async (e) => {
    e.preventDefault();
    if (!confirm("Снять все пометки состава?")) return;
    await rosterCall("clear");
  };
}

// щелчки по людям сохраняются сразу (и сразу видны), а тяжёлый пересчёт доски идёт один раз — когда щелчки стихли
let rosterTimer = null, rosterBusy = false, rosterQueue = Promise.resolve();
function markPerson(trackId, role, scope = "all") {
  if (!currentJob) return;
  const frame = nowFrame();
  if (role === "auto") delete pendingMarks[trackId]; else pendingMarks[trackId] = { role, scope, frame };
  if (role === "auto") { const m = trackMeta(trackId); m.manual = false; }
  redrawAll();
  const job = currentJob.id;
  rosterQueue = rosterQueue.then(() => api(`/api/jobs/${job}/board/tracks`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ track_id: trackId, role, frame, scope, defer: true }) })).catch((err) => { $("board-msg").textContent = "ошибка: " + err.message; });
  clearTimeout(rosterTimer);
  $("board-msg").textContent = "пометка сохранена — отметьте остальных, пересчёт начнётся через пару секунд…";
  rosterTimer = setTimeout(() => rosterCall("apply"), 1800);
}
async function rosterCall(action, body) {
  if (!currentJob) return;
  if (rosterBusy) { clearTimeout(rosterTimer); rosterTimer = setTimeout(() => rosterCall(action, body), 800); return; }
  rosterBusy = true;
  $("board-msg").textContent = "⏳ пересчитываю состав, доску и метрики…";
  try {
    await rosterQueue;
    const sent = Object.keys(pendingMarks);
    board = await api(`/api/jobs/${currentJob.id}/board/roster/${action}`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined });
    for (const k of sent) delete pendingMarks[k];
    $("board-msg").textContent = "";
    renderBoardLegend(); renderSideline(); renderPick(); redrawAll();
    loadPlayerMetrics(currentJob);
  } catch (err) { $("board-msg").textContent = "ошибка: " + err.message; }
  rosterBusy = false;
}
const setTrackRole = (id, role) => markPerson(id, role);

function currentTime() { const v = $("player"); return v && v.src ? v.currentTime : parseFloat($("board-time").value) || 0; }

function renderBoardLegend() {
  if (!board) return;
  const items = [];
  for (const [k, t] of Object.entries(board.teams)) items.push(`<span><i class="dot" style="background:${t.color}"></i>${t.label}</span>`);
  items.push(`<span><i class="dot" style="background:#fde047;border-color:#fde047"></i>игрок в фокусе</span>`);
  items.push(`<span><i class="dot" style="background:#fff"></i>мяч найден</span>`);
  items.push(`<span><i class="dot" style="background:transparent;border-style:dashed"></i>мяч — оценка</span>`);
  $("board-legend").innerHTML = items.join("");
}

function boardFrameAt(t) {
  const fr = board.frames;
  if (!fr.length) return null;
  const k = Math.max(0, Math.min(fr.length - 1, Math.round(t * board.fps)));
  return fr[k];
}

function boardXY(canvas, x, y) { return pitchXY(canvas, x / board.field.length_m, y / board.field.width_m); }

// точка доски, прижатая к краю холста, если человек далеко за полем (третий элемент — «прижата»)
function boardPoint(canvas, x, y) {
  const [px, py] = boardXY(canvas, x, y);
  const cx = Math.min(Math.max(px, 9), canvas.width - 9), cy = Math.min(Math.max(py, 9), canvas.height - 9);
  return [cx, cy, cx !== px || cy !== py];
}

function numberTag(ctx, text, x, y) {
  ctx.font = "bold 10px system-ui";
  const w = ctx.measureText(text).width + 6;
  ctx.fillStyle = "rgba(17,24,39,.85)"; ctx.fillRect(x - w / 2, y - 11, w, 12);
  ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.fillText(text, x, y - 2); ctx.textAlign = "left";
}

function drawBoard(t) {
  const c = $("board-canvas");
  if (!board) return;
  const f = board.field;
  drawPitch(c, { length_m: f.length_m, width_m: f.width_m, own_goal: f.default ? null : f.own_goal });
  const ctx = c.getContext("2d");
  const [camx, camy] = boardXY(c, ...board.camera);
  ctx.fillStyle = "#fde047"; ctx.beginPath(); ctx.arc(camx, camy, 5, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,.8)"; ctx.font = "10px system-ui"; ctx.fillText("камера", camx + 7, camy + 3);
  const fr = boardFrameAt(t);
  if (!fr) return;
  const k = board.frames.indexOf(fr);
  const trail = parseFloat($("board-trail").value) || 0;
  const all = $("board-all").checked, ids = $("board-ids").checked, showOff = $("board-show-off").checked;
  const isOff = (id) => isOffAt(id, fr.f);
  const visible = (id) => !isOff(id) || showOff;
  const focusId = fr.focus ? fr.focus.id : null;
  const teamColor = (id) => {
    const tr = board.tracks[String(id)];
    if (!tr) return "#e5e7eb";
    const team = board.teams[String(tr.team)];
    return team ? team.color : "#e5e7eb";
  };
  // следы: позиции тех же треков за последние `trail` секунд
  if (trail > 0) {
    const from = Math.max(0, k - Math.round(trail * board.fps));
    const paths = new Map();
    const focusPath = [], ballPath = [];
    for (let i = from; i <= k; i++) {
      const q = board.frames[i];
      if (all) for (const [id, x, y] of q.p) { if (isOff(id)) continue; if (!paths.has(id)) paths.set(id, []); paths.get(id).push([x, y, i]); }
      if (q.focus) focusPath.push([q.focus.x, q.focus.y, i]);
      if (q.ball && i >= k - Math.round(2 * board.fps)) ballPath.push([q.ball[0], q.ball[1], i]);
    }
    const stroke = (pts, color, width) => {
      if (pts.length < 2) return;
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      pts.forEach(([x, y, i], n) => {
        const [px, py] = boardXY(c, x, y);
        // разрыв следа, если трек пропадал
        if (n === 0 || i - pts[n - 1][2] > 3) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    };
    ctx.globalAlpha = 0.35;
    for (const [id, pts] of paths) if (id !== focusId) stroke(pts, teamColor(id), 1.5);
    ctx.globalAlpha = 1;
    stroke(focusPath, "#fde047", 3.5);
    ctx.setLineDash([3, 3]); stroke(ballPath, "#ffffff", 1.5); ctx.setLineDash([]);
  }
  // люди: все найденные; выключенные — бледно и только по галочке; стоящие далеко за полем — у края холста
  if (all) for (const [id, x, y] of fr.p) {
    if (id === focusId || !visible(id)) continue;
    const [px, py, clamped] = boardPoint(c, x, y);
    const meta = board.tracks[String(id)] || {};
    ctx.globalAlpha = isOff(id) ? 0.35 : 1;
    ctx.fillStyle = meta.gk ? (meta.color || "#ccc") : teamColor(id); ctx.strokeStyle = "#111"; ctx.lineWidth = 1.5;
    if (clamped) ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.arc(px, py, 7, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.setLineDash([]);
    if (meta.gk) {   // вратарь: своя форма внутри, кольцо цветом команды снаружи
      ctx.strokeStyle = teamColor(id); ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(px, py, 10, 0, 2 * Math.PI); ctx.stroke();
    }
    if (ids && meta.num != null) numberTag(ctx, String(meta.num) + (meta.gk ? " вр" : ""), px, py < 24 ? py + 22 : py - 11);
    ctx.globalAlpha = 1;
  }
  // выбранный (первый щелчок) — ярко-красное кольцо, даже если он выключен и выключенные скрыты
  if (selectedId != null) {
    const q = fr.p.find(([id]) => id === selectedId);
    if (q) {
      const [px, py] = boardPoint(c, q[1], q[2]);
      ctx.strokeStyle = "#ff1a1a"; ctx.lineWidth = 3.5;
      ctx.beginPath(); ctx.arc(px, py, 13, 0, 2 * Math.PI); ctx.stroke();
      const num = (board.tracks[String(selectedId)] || {}).num;
      if (num != null) numberTag(ctx, String(num), px, py < 24 ? py + 26 : py - 15);
    }
  }
  // игрок в фокусе
  if (fr.focus) {
    const [px, py] = boardXY(c, fr.focus.x, fr.focus.y);
    ctx.fillStyle = focusId != null ? teamColor(focusId) : "#fde047";
    ctx.beginPath(); ctx.arc(px, py, 8, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = "#fde047"; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(px, py, 12, 0, 2 * Math.PI); ctx.stroke();
    const fnum = (board.tracks[String(focusId)] || {}).num;
    ctx.fillStyle = "#fde047"; ctx.font = "bold 11px system-ui";
    ctx.fillText(`${fnum != null ? fnum + " · " : ""}${fr.focus.state === "contested" ? "цель ?" : "цель"}`, px + 14, py - 10);
  }
  // мяч
  if (fr.ball) {
    const [px, py] = boardXY(c, fr.ball[0], fr.ball[1]);
    ctx.lineWidth = 2;
    if (fr.ball[2] === "d") { ctx.fillStyle = "#fff"; ctx.strokeStyle = "#111"; ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
    else { ctx.strokeStyle = "#fff"; ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.arc(px, py, 6, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]); }
  }
  ctx.fillStyle = "rgba(255,255,255,.9)"; ctx.font = "12px system-ui";
  ctx.fillText(fmtT(fr.t), 8, 16);
  if (document.activeElement !== $("board-time")) $("board-time").value = fr.t;
}

function boardLoop() {
  drawBoard(currentTime());
  drawOverlay();
  const v = $("player");
  boardRaf = v && !v.paused && !v.ended ? requestAnimationFrame(boardLoop) : null;
}
function redrawAll() { drawBoard(currentTime()); drawOverlay(); }
$("player").addEventListener("play", () => { if (!boardRaf) boardRaf = requestAnimationFrame(boardLoop); });
for (const ev of ["seeked", "pause", "loadeddata", "timeupdate"]) $("player").addEventListener(ev, redrawAll);
window.addEventListener("resize", () => drawOverlay());
$("overlay-on").onchange = () => drawOverlay();

// ---------- выбор человека: доска и видео ----------
// первый щелчок — выбрать (красная обводка на доске и на видео), второй по тому же — выключить / вернуть
let selectedId = null, vboxes = null;

function trackMeta(id) { return (board && board.tracks[String(id)]) || {}; }

function pickTrack(id, focusId) {
  if (id === focusId) {
    selectedId = null; renderPick("это игрок в фокусе — его нельзя выключить"); redrawAll(); return;
  }
  if (selectedId !== id) { selectedId = id; renderPick(); redrawAll(); return; }
  togglePerson(id, "all");
}
// «выключить / вернуть»: своя пометка того же смысла снимается, иначе ставится обратная
function togglePerson(id, scope) {
  const off = isOffAt(id, nowFrame()), m = trackMeta(id);
  selectedId = null;
  renderPick();
  // на этом треке уже стоит ваша пометка «весь ролик» — повторный щелчок её снимает; иначе ставится обратная
  if (scope === "all" && m.manual && !pendingMarks[id]) markPerson(id, "auto");
  else markPerson(id, off ? "play" : "sideline", scope);
}

function renderPick(note) {
  const box = $("pick-msg");
  if (note) { box.textContent = note; box.classList.remove("hidden"); return; }
  if (selectedId == null) { box.classList.add("hidden"); box.innerHTML = ""; return; }
  const m = trackMeta(selectedId);
  const off = isOffAt(selectedId, nowFrame());
  const team = board && board.teams[String(m.team)];
  const who = m.gk ? "вратарь" : team ? team.label : "команда не определена";
  const why = { manual: "ваша пометка", person: "тот же человек, что отмеченный вами", similar: "похож по внешности на отмеченных вами", default: board && board.roster_mode === "marked" ? "не отмечен как игрок" : "" }[m.why] || "";
  box.innerHTML = `Выбран №${m.num ?? "?"} (${who}, в кадре ${m.seconds ?? "?"} с) — сейчас <b>${off ? "вне игры" : "в игре"}</b>${why ? " (" + why + ")" : ""}. `
    + `Ещё щелчок по нему — ${off ? "вернуть в игру" : "выключить"}. `
    + `<button class="secondary" id="pick-toggle">${off ? "В игре" : "Вне игры"} — весь ролик</button>`
    + `<button class="secondary" id="pick-from" title="замена: с текущего кадра и дальше">${off ? "Вышел на поле" : "Ушёл с поля"} — с этого момента</button>`
    + `<button class="secondary" id="pick-cancel">Отмена</button>`;
  box.classList.remove("hidden");
  $("pick-toggle").onclick = () => togglePerson(selectedId, "all");
  $("pick-from").onclick = () => togglePerson(selectedId, "from");
  $("pick-cancel").onclick = () => { selectedId = null; renderPick(); redrawAll(); };
}

function videoFrameIndex() {
  const v = $("player");
  return Math.floor((v.currentTime || 0) * jobFps() + 0.01);
}

// прямоугольник, в который браузер вписал кадр внутри <video> (object-fit: contain);
// k — пикселей экрана на пиксель исходного кадра (рамки в координатах исходника, копия кадров может быть меньше)
function videoRect() {
  const v = $("player");
  const W = v.clientWidth, H = v.clientHeight, vw = v.videoWidth || 16, vh = v.videoHeight || 9;
  const s = Math.min(W / vw, H / vh);
  const meta = timelineFrames && timelineFrames.meta;
  const srcW = meta && meta.scale ? vw / meta.scale : vw;      // scale = кадр обработки / исходный кадр
  return { x: (W - vw * s) / 2, y: (H - vh * s) / 2, s, k: s * vw / srcW, W, H };
}

function boxStyle(id, focus) {
  if (focus) return { color: "#22c55e", width: 3.5, label: "#22c55e", glow: "rgba(34,197,94,.9)" };
  if (id === selectedId) return { color: "#ff1a1a", width: 4, label: "#ff4d4d", glow: "rgba(255,26,26,.95)", fill: "rgba(255,26,26,.18)" };
  const m = trackMeta(id);
  if (isOffAt(id, nowFrame())) return { color: "rgba(156,163,175,.45)", width: 1.5, dash: [4, 3], label: "rgba(209,213,219,.8)" };
  const team = board && board.teams[String(m.team)];
  if (team && team.role === "own") return { color: "rgba(248,113,113,.5)", width: 1.5, label: "#fca5a5" };
  if (team && team.role === "opp") return { color: "rgba(255,255,255,.55)", width: 1.5, label: "#ffffff" };
  return { color: "rgba(148,163,184,.6)", width: 1.3, label: "rgba(203,213,225,.9)" };
}

function drawOverlay() {
  const c = $("overlay"), v = $("player");
  if (!vboxes || !vboxes.clean_video || c.classList.contains("hidden")) return;
  const r = videoRect();
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(r.W * dpr) || c.height !== Math.round(r.H * dpr)) {
    c.width = Math.round(r.W * dpr); c.height = Math.round(r.H * dpr);
    c.style.width = r.W + "px"; c.style.height = r.H + "px";
  }
  $("overlay-click").style.height = Math.max(r.H - 56, 0) + "px";
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.W, r.H);
  if (!$("overlay-on").checked || !v.videoWidth) return;
  const idx = videoFrameIndex();
  drawFieldOnVideo(ctx, r, idx);
  const row = vboxes.frames[idx];
  if (!row) return;
  const rec = timelineFrames && timelineFrames.frames[idx];
  const focusId = rec && (rec.state === "active" || rec.state === "contested") ? rec.track_id : null;
  const k = r.k;
  const items = [];
  for (let i = 0; i + 4 < row.length; i += 5) items.push(row.slice(i, i + 5));
  // цель в кадрах, где её трек не детектирован (уточнение по ролику дорисовало рамку) — рамка из результата
  if (rec && rec.box && (rec.state === "active" || rec.state === "contested") && !items.some((q) => q[0] === focusId)) {
    items.push([focusId, ...rec.box.map((b) => Math.round(b))]);
  }
  // выбранный и цель рисуются последними — поверх остальных
  items.sort((a, b) => (a[0] === selectedId || a[0] === focusId) - (b[0] === selectedId || b[0] === focusId));
  for (const [id, x1, y1, x2, y2] of items) {
    const st = boxStyle(id, id === focusId);
    const X = r.x + x1 * k, Y = r.y + y1 * k, w = (x2 - x1) * k, h = (y2 - y1) * k;
    if (st.fill) { ctx.fillStyle = st.fill; ctx.fillRect(X, Y, w, h); }
    ctx.shadowColor = st.glow || "transparent"; ctx.shadowBlur = st.glow ? 8 : 0;
    ctx.strokeStyle = st.color; ctx.lineWidth = st.width; ctx.setLineDash(st.dash || []);
    ctx.strokeRect(X, Y, w, h);
    ctx.setLineDash([]); ctx.shadowBlur = 0;
    const num = trackMeta(id).num;
    const text = id === focusId ? `${num ?? ""} цель`.trim() : num != null ? String(num) : "";
    if (text) {
      ctx.font = `bold ${id === focusId || id === selectedId ? 12 : 10}px system-ui`;
      const tw = ctx.measureText(text).width + 6, th = id === focusId || id === selectedId ? 15 : 13;
      const ty = Math.max(Y - th - 1, 0);
      ctx.fillStyle = "rgba(17,24,39,.8)"; ctx.fillRect(X, ty, tw, th);
      ctx.fillStyle = st.label; ctx.fillText(text, X + 3, ty + th - 3);
    }
  }
  if (rec && rec.state === "lost") {
    ctx.font = "bold 13px system-ui"; ctx.fillStyle = "rgba(215,38,61,.9)";
    ctx.fillText("цель потеряна в этом кадре", r.x + 10, r.y + 20);
  }
}

$("overlay-click").onclick = (e) => {
  if (!vboxes || !board) return;
  const v = $("player");
  const rect = $("overlay-click").getBoundingClientRect();
  const r = videoRect();
  const k = r.k;
  const x = (e.clientX - rect.left - r.x) / k, y = (e.clientY - rect.top - r.y) / k;
  const idx = videoFrameIndex();
  const row = vboxes.frames[idx] || [];
  const rec = timelineFrames && timelineFrames.frames[idx];
  const focusId = rec && (rec.state === "active" || rec.state === "contested") ? rec.track_id : null;
  let best = null;
  for (let i = 0; i + 4 < row.length; i += 5) {
    const [id, x1, y1, x2, y2] = row.slice(i, i + 5);
    if (x < x1 - 4 || x > x2 + 4 || y < y1 - 4 || y > y2 + 4) continue;
    const area = (x2 - x1) * (y2 - y1);
    if (!best || area < best.area) best = { id, area };      // вложенные рамки — берём меньшую (дальнего игрока)
  }
  if (!v.paused) v.pause();
  if (!best) { selectedId = null; renderPick(); redrawAll(); return; }
  pickTrack(best.id, focusId);
};
$("board-canvas").onclick = (e) => {
  if (!board) return;
  const c = e.target, r = c.getBoundingClientRect();
  const cx = (e.clientX - r.left) * c.width / r.width, cy = (e.clientY - r.top) * c.height / r.height;
  const fr = boardFrameAt(currentTime());
  if (!fr) return;
  const focusId = fr.focus ? fr.focus.id : null;
  const showOff = $("board-show-off").checked;
  let best = null;
  for (const [id, x, y] of fr.p) {
    const off = isOffAt(id, fr.f);
    if (off && !showOff && id !== selectedId) continue;
    const [px, py] = boardPoint(c, x, y);
    const d = Math.hypot(px - cx, py - cy);
    if (d < 16 && (!best || d < best.d)) best = { id, d };
  }
  if (!best) { selectedId = null; renderPick(); redrawAll(); return; }
  pickTrack(best.id, focusId);
};
$("board-time").oninput = () => {
  const t = parseFloat($("board-time").value) || 0;
  const v = $("player");
  if (v && v.src) v.currentTime = t;
  drawBoard(t);
};
for (const id of ["board-trail", "board-all", "board-ids", "board-show-off"]) $(id).onchange = () => drawBoard(currentTime());
$("board-build").onclick = async () => {
  if (!currentJob) return;
  $("board-build").disabled = true;
  $("board-msg").textContent = "перестроение по сохранённым данным: журнал игроков и поиск мяча на увеличенных кадрах (для старой задачи — один раз, несколько минут)…";
  try {
    await api(`/api/jobs/${currentJob.id}/board/build`, { method: "POST" });
    startPolling();
  } catch (err) { $("board-msg").textContent = "ошибка: " + err.message; }
  finally { $("board-build").disabled = false; }
};

// ---------- разметка поля на кадре ----------
// углы 1–4: левый ближний, правый ближний, правый дальний, левый дальний (в координатах исходного кадра);
// rotation поворачивает их соответствие углам схемы (как в tracker/field.py). Разметок может быть несколько —
// по одной на место съёмки; кадр ролика считается по ближайшей по времени.
const HANDLE_CSS = 12, GRAB_CSS = 30;   // радиус ручки и зона захвата — в экранных пикселях (палец)
const fieldEd = { marks: [], cur: -1, frame: null, img: null, scale: 1, corners: null, rotation: 0, drag: -1, sel: 0,
  grab: [0, 0], pointer: null, margin: 0.3, zoom: 1, occ: null, occFrame: null, busy: false,
  mode: "corners", previewH: null, previewKey: "", previewTimer: null };

function pitchCornersJS(L, W, rot) {
  const ring = [[0, W], [L, W], [L, 0], [0, 0]];
  const k = ((rot % 4) + 4) % 4;
  return ring.slice(k).concat(ring.slice(0, k));
}

// перспективное преобразование по 4 точкам (решение 8×8 методом Гаусса)
function homographyJS(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  const n = 8;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    if (Math.abs(A[c][c]) < 1e-12) return null;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  const h = b.map((v, i) => v / A[i][i]);
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}
function applyH(H, x, y) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w, w];
}
function mul3(A, B) { return A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j])); }
function inv3(M) {
  const [[a, b, c], [d, e, f], [g, h, i]] = M;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
          [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
          [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}
// cams[i] — «кадр i -> сцена»: 8 чисел (гомография точной привязки) или 6 (подобие покадрового движения)
function camMat(i) {
  const c = vboxes.cams[Math.min(Math.max(i, 0), vboxes.cams.length - 1)];
  return c.length >= 8 ? [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], 1]] : [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [0, 0, 1]];
}
function frameToFrame(a, b) { return mul3(inv3(camMat(b)), camMat(a)); }   // кадр a -> сцена -> кадр b
function moveCorners(corners, a, b) {
  const T = frameToFrame(a, b);
  return corners.map(([x, y]) => applyH(T, x, y).slice(0, 2));
}

// линии поля в метрах: контур, средняя линия, центральный круг, штрафные, ворота (как на схеме drawPitch)
function pitchLines(L, W) {
  const lines = [], seg = (a, b, n = 24) => Array.from({ length: n + 1 }, (_, i) => [a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n]);
  const poly = (pts, n) => pts.flatMap((p, i, arr) => i ? seg(arr[i - 1], p, n).slice(1) : [p]);
  lines.push({ pts: poly([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], 24) });
  lines.push({ pts: seg([L / 2, 0], [L / 2, W]) });
  const r = Math.min(4.5, L / 8);
  lines.push({ pts: Array.from({ length: 49 }, (_, i) => [L / 2 + r * Math.cos(i / 48 * 2 * Math.PI), W / 2 + r * Math.sin(i / 48 * 2 * Math.PI)]) });
  const bd = Math.min(L * 0.16, 16.5), bw = Math.min(W * 0.6, 40), gw = Math.min(W * 0.2, 7.3);
  for (const [x0, dir, side] of [[0, 1, "left"], [L, -1, "right"]]) {
    const x1 = x0 + dir * bd;
    lines.push({ pts: poly([[x0, W / 2 - bw / 2], [x1, W / 2 - bw / 2], [x1, W / 2 + bw / 2], [x0, W / 2 + bw / 2]], 8) });
    lines.push({ pts: seg([x0, W / 2 - gw / 2], [x0, W / 2 + gw / 2], 6), goal: side });
  }
  return lines;
}
function goalColor(side, own, alpha = 1) {
  if (!own) return `rgba(253,224,71,${alpha})`;
  return side === own ? `rgba(96,165,250,${alpha})` : `rgba(248,113,113,${alpha})`;
}

function savedField() { return (currentJob && currentJob.player && currentJob.player.field) || null; }
function savedMarks() { const f = savedField(); return f ? (f.marks || (f.corners ? [f] : [])) : []; }
function formatDims() {
  const fmt = currentJob && currentJob.player && currentJob.player.game_format;
  return (profileOptions.fields && profileOptions.fields[fmt]) || [40, 25];
}
function fieldDims() {
  const d = formatDims();
  return [parseFloat($("field-len").value) || d[0], parseFloat($("field-wid").value) || d[1]];
}
function defaultCorners(w, h) {
  return [[0.05 * w, 0.92 * h], [0.95 * w, 0.92 * h], [0.78 * w, 0.52 * h], [0.22 * w, 0.52 * h]];
}
function regNote() {
  const j = currentJob;
  if (!j) return "";
  if (j.reg_stage === "running" || j.reg_stage === "queued") return ` · точная привязка кадров считается${j.reg_progress && j.total ? " " + Math.round(100 * j.reg_progress / j.total) + " %" : ""}…`;
  if (j.reg_stage === "failed") return " · точная привязка кадров не удалась — линии могут сползать";
  return "";
}

function renderFieldMarks() {
  const box = $("field-marks"); box.innerHTML = "";
  fieldEd.marks.forEach((m, k) => {
    const el = document.createElement("span");
    el.className = "chip" + (k === fieldEd.cur && !$("field-editor").classList.contains("hidden") ? " active" : "");
    el.textContent = `кадр ${m.frame} · ${fmtT(m.frame / jobFps())}${m.unsaved ? " · не сохранено" : ""}`;
    el.title = "открыть эту разметку";
    el.onclick = () => openFieldEditor(m.frame);
    box.appendChild(el);
  });
  const n = savedMarks().length;
  $("field-frame-label").textContent = (n ? (n > 1 ? `разметок: ${n}` : "поле отмечено") : "поле не отмечено") + regNote();
  // одни «свои ворота»: при разметке на видео старая схема поля не нужна
  const hide = n > 0 && !fieldEd.showPitch;
  $("pitch-panel").classList.toggle("hidden", hide);
  $("pitch-hidden-note").classList.toggle("hidden", !hide);
}
$("pitch-show").onclick = (e) => { e.preventDefault(); fieldEd.showPitch = true; renderFieldMarks(); };

function initField(j) {
  Object.assign(fieldEd, { marks: savedMarks().map((m) => ({ frame: m.frame, corners: m.corners.map((p) => [...p]), rotation: m.rotation || 0, mode: m.mode || "corners" })),
    cur: -1, img: null, corners: null, occ: null, occFrame: null, showPitch: false, zoom: 1 });
  $("field-editor").classList.add("hidden");
  $("field-check").classList.add("hidden");
  renderFieldMarks();
}

async function openFieldEditor(frame) {
  if (!currentJob) return;
  frame = Math.max(0, Math.round(frame));
  // разметка рядом (±1 с) — правим её, а не плодим вторую на том же месте
  const near = fieldEd.marks.findIndex((m) => Math.abs(m.frame - frame) <= jobFps());
  if (near >= 0) frame = fieldEd.marks[near].frame;
  $("field-editor").classList.remove("hidden");
  $("field-msg").textContent = "загрузка кадра…";
  const resp = await fetch(`/api/jobs/${currentJob.id}/frame/${frame}`);
  if (!resp.ok) { $("field-msg").textContent = "не удалось загрузить кадр"; return; }
  fieldEd.scale = parseFloat(resp.headers.get("X-Scale")) || 1;
  fieldEd.img = await createImageBitmap(await resp.blob());
  fieldEd.frame = frame;
  const w = fieldEd.img.width / fieldEd.scale, h = fieldEd.img.height / fieldEd.scale;      // исходный кадр
  const f = savedField();
  if (near >= 0) {
    fieldEd.cur = near;
  } else {
    // новая разметка: начинаем с ближайшей имеющейся, перенесённой движением камеры; нет — с углов по игрокам
    let corners = null, rotation = 0, mode = "corners";
    if (fieldEd.marks.length && vboxes && vboxes.cams) {
      const src = fieldEd.marks.reduce((a, b) => (Math.abs(b.frame - frame) < Math.abs(a.frame - frame) ? b : a));
      corners = moveCorners(src.corners, src.frame, frame); rotation = src.rotation; mode = src.mode || "corners";
      if (corners.some(([x, y]) => !isFinite(x) || !isFinite(y))) corners = null;
    }
    fieldEd.marks.push({ frame, corners: corners || defaultCorners(w, h), rotation, mode, unsaved: true, fresh: !corners });
    fieldEd.marks.sort((a, b) => a.frame - b.frame);
    fieldEd.cur = fieldEd.marks.findIndex((m) => m.frame === frame);
  }
  const m = fieldEd.marks[fieldEd.cur];
  fieldEd.corners = m.corners; fieldEd.rotation = m.rotation; fieldEd.sel = 0;
  fieldEd.mode = m.mode || "corners"; fieldEd.previewH = null; fieldEd.previewKey = "";
  $("field-mode").value = fieldEd.mode; renderModeHint();
  const fd = formatDims();
  $("field-len").value = Math.round((f && f.length_m) || fd[0]);
  $("field-wid").value = Math.round((f && f.width_m) || fd[1]);
  $("field-dims-note").textContent = f && f.length_m ? "" : "размеры — по формату игры, поправьте при необходимости";
  $("field-own").value = (f && f.own_goal) || (pitch.saved && pitch.own_goal) || "";
  $("field-msg").textContent = "";
  fitMargin(); renderFieldMarks(); drawField();
  await loadOccupancy();
  if (m.fresh) { m.fresh = false; await suggestCorners(false, true); }     // вместо трапеции «с потолка» — углы по игрокам
}

async function loadOccupancy() {
  if (fieldEd.occFrame === fieldEd.frame && fieldEd.occ) return;
  try {
    const r = await api(`/api/jobs/${currentJob.id}/field/occupancy?frame=${fieldEd.frame}`);
    fieldEd.occ = r.points; fieldEd.occFrame = r.frame;
  } catch (_) { fieldEd.occ = null; }
  drawField();
}

async function suggestCorners(expand, quiet) {
  if (!currentJob || !fieldEd.img || fieldEd.busy) return;
  const [L, W] = fieldDims();
  const body = { frame: fieldEd.frame, rotation: fieldEd.rotation, length_m: L, width_m: W, mode: fieldEd.mode };
  if (expand) body.corners = fieldEd.corners;
  if (!quiet) $("field-msg").textContent = "считаю по игрокам…";
  try {
    const r = await api(`/api/jobs/${currentJob.id}/field/suggest`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    setCorners(r.corners, r.rotation, r.marks_mode);
    const pct = (v) => Math.round(100 * v) + " %";
    $("field-msg").textContent = expand
      ? (r.outside_before === r.outside_after ? "игроки и так внутри разметки — углы не тронуты" : `раздвинуто: за полем было ${pct(r.outside_before)} точек игроков, стало ${pct(r.outside_after)}`)
      : (r.marks_mode === "sides" ? "ближние углы поля — далеко за кадром, поэтому включён режим «дальние углы + боковые линии»: 3 и 4 — дальние углы, 1 и 2 — на боковых линиях. "
        : "углы поставлены вокруг места, где играли. ") + "Это заготовка по движению игроков — поправьте точки по кадру и сохраните";
  } catch (err) { if (!quiet) $("field-msg").textContent = "не получилось: " + err.message; }
}
function setCorners(corners, rotation, mode) {
  const m = fieldEd.marks[fieldEd.cur];
  m.corners = corners.map((p) => [...p]); m.rotation = rotation; m.unsaved = true;
  if (mode) { m.mode = mode; fieldEd.mode = mode; $("field-mode").value = mode; renderModeHint(); }
  fieldEd.corners = m.corners; fieldEd.rotation = rotation;
  fitMargin(); renderFieldMarks(); drawField();
}

function renderModeHint() {
  $("field-mode-hint").textContent = fieldEd.mode === "sides"
    ? "3 и 4 — дальние углы поля; 1 и 2 — любые видимые места на боковых линиях (1 — на линии от угла 4, 2 — от угла 3). Остальное достроится по камере."
    : "все четыре точки — в углах поля; угол за краем кадра ставьте в серой зоне. Ближние углы совсем далеко за кадром — выберите второй режим.";
}
// как ляжет поле: в режиме corners — гомография на месте; в режиме sides модель камеры считает сервер
function fieldKey() { return JSON.stringify([fieldEd.frame, fieldEd.corners, fieldEd.rotation, fieldEd.mode, fieldDims()]); }
function requestPreview(now) {
  clearTimeout(fieldEd.previewTimer);
  fieldEd.previewTimer = setTimeout(async () => {
    if (!currentJob || !fieldEd.img) return;
    const key = fieldKey(), [L, W] = fieldDims();
    try {
      const r = await api(`/api/jobs/${currentJob.id}/field/preview`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: fieldEd.frame, corners: fieldEd.corners, rotation: fieldEd.rotation, mode: fieldEd.mode, length_m: L, width_m: W }) });
      if (key !== fieldKey()) return;                                   // точки уже сдвинули — ответ устарел
      fieldEd.previewH = r.H; fieldEd.previewKey = key;
      if (fieldEd.drag < 0 && !fieldEd.busy) {
        $("field-msg").textContent = `за границей поля ${Math.round(100 * r.outside)} % положений игроков`
          + (r.rms_m != null ? ` · точки согласуются с камерой до ${r.rms_m} м${r.rms_m > 2.5 ? " — многовато: проверьте дальние углы и что 1, 2 стоят на боковых линиях" : ""}` : "");
      }
      drawField();
    } catch (err) { if (fieldEd.drag < 0) $("field-msg").textContent = err.message; }
  }, now ? 0 : 120);
}

// запас холста вокруг кадра — чтобы поместились углы за краем кадра (ближние углы у ног оператора далеко внизу)
function fitMargin() {
  if (!fieldEd.img || !fieldEd.corners) return;
  const w = fieldEd.img.width / fieldEd.scale, h = fieldEd.img.height / fieldEd.scale;
  let m = 0.3;
  for (const [x, y] of fieldEd.corners) m = Math.max(m, -x / w + 0.08, (x - w) / w + 0.08, -y / h + 0.08, (y - h) / h + 0.08);
  fieldEd.margin = Math.min(m, 1.5);
}

function fieldView() {
  const c = $("field-canvas"), img = fieldEd.img;
  const w = img.width / fieldEd.scale, h = img.height / fieldEd.scale;
  const mx = w * fieldEd.margin, my = h * fieldEd.margin;
  const k = Math.min(1600 * Math.min(fieldEd.zoom, 2), (w + 2 * mx)) / (w + 2 * mx);          // пикселей холста на пиксель исходника
  const cw = Math.round((w + 2 * mx) * k), ch = Math.round((h + 2 * my) * k);
  if (c.width !== cw || c.height !== ch) { c.width = cw; c.height = ch; }
  c.style.width = Math.round(fieldEd.zoom * 100) + "%";
  const css = c.getBoundingClientRect().width || cw;                                           // экранных пикселей
  return { c, w, h, mx, my, k, dpr: cw / css, toC: ([x, y]) => [(x + mx) * k, (y + my) * k], fromC: (cx, cy) => [cx / k - mx, cy / k - my] };
}

function drawField() {
  if (!fieldEd.img || !fieldEd.corners) return;
  const V = fieldView(), ctx = V.c.getContext("2d"), px = V.dpr;                               // px — экранный пиксель в пикселях холста
  ctx.fillStyle = "#374151"; ctx.fillRect(0, 0, V.c.width, V.c.height);
  const [ix, iy] = V.toC([0, 0]);
  ctx.drawImage(fieldEd.img, ix, iy, V.w * V.k, V.h * V.k);
  ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.lineWidth = px; ctx.setLineDash([4 * px, 4 * px]); ctx.strokeRect(ix, iy, V.w * V.k, V.h * V.k); ctx.setLineDash([]);
  if (fieldEd.occ && $("field-occ").checked) {
    ctx.fillStyle = "rgba(34,211,238,.55)";
    const r = Math.max(1.2 * px, 1);
    for (const p of fieldEd.occ) { const [cx, cy] = V.toC(p); ctx.fillRect(cx - r, cy - r, 2 * r, 2 * r); }
  }
  const [L, W] = fieldDims();
  // метры поля -> исходный кадр
  let Hinv = null;
  if (fieldEd.mode === "sides") {
    if (fieldEd.previewH) Hinv = inv3(fieldEd.previewH);
    if (fieldEd.previewKey !== fieldKey()) requestPreview();
  } else Hinv = homographyJS(pitchCornersJS(L, W, fieldEd.rotation), fieldEd.corners);
  const sgn = Hinv && fieldEd.mode === "sides" ? Math.sign(applyH(fieldEd.previewH, ...fieldEd.corners[2])[2]) || 1 : 1;
  fieldEd.goalHits = [];
  if (Hinv) {
    const own = $("field-own").value;
    for (const ln of pitchLines(L, W)) {
      ctx.strokeStyle = ln.goal ? goalColor(ln.goal, own) : "rgba(255,255,255,.9)";
      ctx.lineWidth = (ln.goal ? 5 : 2) * px;
      ctx.beginPath();
      let pen = false, sx = 0, sy = 0, n = 0;
      for (const [x, y] of ln.pts) {
        const [u, v, wgt] = applyH(Hinv, x, y);
        // точка поля за камерой (или выше горизонта) на кадр не попадает; слишком далёкие не рисуем
        const back = fieldEd.mode === "sides" ? applyH(fieldEd.previewH, u, v)[2] * sgn <= 0 : wgt <= 0;
        if (back || !isFinite(u) || !isFinite(v) || Math.abs(u) > 20 * V.w || Math.abs(v) > 20 * V.h) { pen = false; continue; }
        const [cx, cy] = V.toC([u, v]);
        if (pen) ctx.lineTo(cx, cy); else { ctx.moveTo(cx, cy); pen = true; }
        if (cx > 0 && cy > 0 && cx < V.c.width && cy < V.c.height) { sx += cx; sy += cy; n++; }
      }
      ctx.stroke();
      if (ln.goal && n) {
        // подпись ворот: А — левые на схеме, Б — правые; щелчок по воротам выбирает свои
        const gx = sx / n, gy = sy / n, name = (ln.goal === "left" ? "ворота А" : "ворота Б") + (own === ln.goal ? " — свои" : own ? " — чужие" : "");
        ctx.font = `bold ${13 * px}px system-ui`; ctx.textAlign = "center";
        ctx.lineWidth = 3 * px; ctx.strokeStyle = "rgba(0,0,0,.75)"; ctx.strokeText(name, gx, gy - 9 * px);
        ctx.fillStyle = goalColor(ln.goal, own); ctx.fillText(name, gx, gy - 9 * px); ctx.textAlign = "left";
        fieldEd.goalHits.push({ side: ln.goal, x: gx, y: gy });
      }
    }
  }
  // четырёхугольник и точки
  ctx.strokeStyle = "#fde047"; ctx.lineWidth = 1.5 * px; ctx.setLineDash([6 * px, 4 * px]);
  ctx.beginPath();
  // sides: точки 1, 2 — не углы, ближней стороны у четырёхугольника нет (1 — 4 — 3 — 2)
  const ring = fieldEd.mode === "sides" ? [0, 3, 2, 1] : [0, 1, 2, 3];
  ring.forEach((k, i) => { const [cx, cy] = V.toC(fieldEd.corners[k]); i ? ctx.lineTo(cx, cy) : ctx.moveTo(cx, cy); });
  if (fieldEd.mode !== "sides") ctx.closePath();
  ctx.stroke(); ctx.setLineDash([]);
  fieldEd.corners.forEach((p, i) => {
    const [cx, cy] = V.toC(p), active = fieldEd.drag === i;
    ctx.beginPath();
    if (fieldEd.mode === "sides" && i < 2) {              // точка на боковой линии — ромб, угол — круг
      const r = HANDLE_CSS * px * 1.15;
      ctx.moveTo(cx, cy - r); ctx.lineTo(cx + r, cy); ctx.lineTo(cx, cy + r); ctx.lineTo(cx - r, cy); ctx.closePath();
    } else ctx.arc(cx, cy, HANDLE_CSS * px, 0, 2 * Math.PI);
    ctx.fillStyle = active ? "rgba(255,26,26,.35)" : i === fieldEd.sel ? "rgba(253,224,71,.95)" : "rgba(253,224,71,.7)"; ctx.fill();
    ctx.lineWidth = (i === fieldEd.sel ? 2.5 : 1.2) * px; ctx.strokeStyle = i === fieldEd.sel ? "#fff" : "#111"; ctx.stroke();
    if (active) {                                   // при перетаскивании ручка прозрачная с перекрестием — видно, куда ставим
      ctx.strokeStyle = "#ff1a1a"; ctx.lineWidth = px; ctx.beginPath();
      ctx.moveTo(cx - HANDLE_CSS * px, cy); ctx.lineTo(cx + HANDLE_CSS * px, cy); ctx.moveTo(cx, cy - HANDLE_CSS * px); ctx.lineTo(cx, cy + HANDLE_CSS * px); ctx.stroke();
    } else {
      ctx.fillStyle = "#111"; ctx.font = `bold ${13 * px}px system-ui`; ctx.textAlign = "center"; ctx.fillText(String(i + 1), cx, cy + 4.5 * px); ctx.textAlign = "left";
    }
  });
  if (fieldEd.drag >= 0) drawLoupe(ctx, V);
}

// лупа ×4 над пальцем: сам кадр вокруг точки, перекрестие — точное место угла
function drawLoupe(ctx, V) {
  const px = V.dpr, R = 62 * px, zoom = 4, [x, y] = fieldEd.corners[fieldEd.drag];
  const [cx, cy] = V.toC([x, y]);
  let lx = cx, ly = cy - R - 46 * px;
  if (ly - R < 0) ly = cy + R + 46 * px;
  lx = Math.min(Math.max(lx, R + 2), V.c.width - R - 2);
  ctx.save();
  ctx.beginPath(); ctx.arc(lx, ly, R, 0, 2 * Math.PI); ctx.clip();
  ctx.fillStyle = "#374151"; ctx.fillRect(lx - R, ly - R, 2 * R, 2 * R);
  const src = R / (zoom * V.k * px) * px;                                  // радиус лупы в пикселях исходного кадра
  const s = fieldEd.scale;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(fieldEd.img, (x - src) * s, (y - src) * s, 2 * src * s, 2 * src * s, lx - R, ly - R, 2 * R, 2 * R);
  ctx.restore();
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * px; ctx.beginPath(); ctx.arc(lx, ly, R, 0, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = "#ff1a1a"; ctx.lineWidth = px; ctx.beginPath();
  ctx.moveTo(lx - R, ly); ctx.lineTo(lx + R, ly); ctx.moveTo(lx, ly - R); ctx.lineTo(lx, ly + R); ctx.stroke();
}

function fieldPointer(e) {
  const V = fieldView(), r = V.c.getBoundingClientRect();
  const cx = (e.clientX - r.left) * V.c.width / r.width, cy = (e.clientY - r.top) * V.c.height / r.height;
  return { V, cx, cy, src: V.fromC(cx, cy) };
}
function markDirty() { const m = fieldEd.marks[fieldEd.cur]; if (m) { m.corners = fieldEd.corners; m.rotation = fieldEd.rotation; m.mode = fieldEd.mode; if (!m.unsaved) { m.unsaved = true; renderFieldMarks(); } } }

$("field-canvas").addEventListener("pointerdown", (e) => {
  if (!fieldEd.img || fieldEd.busy) return;
  const { V, cx, cy, src } = fieldPointer(e);
  let best = -1, bd = 1e9;
  fieldEd.corners.forEach((p, i) => { const [hx, hy] = V.toC(p); const d = Math.hypot(hx - cx, hy - cy) / V.dpr; if (d < bd) { bd = d; best = i; } });
  if (bd > GRAB_CSS) {
    // щелчок по воротам — выбор своих
    const g = (fieldEd.goalHits || []).find((h) => Math.hypot(h.x - cx, h.y - cy) / V.dpr < 36);
    if (g) { $("field-own").value = $("field-own").value === g.side ? "" : g.side; drawField(); }
    return;
  }
  fieldEd.drag = best; fieldEd.sel = best;
  fieldEd.grab = [fieldEd.corners[best][0] - src[0], fieldEd.corners[best][1] - src[1]];   // точка не прыгает под палец
  e.target.setPointerCapture(e.pointerId);
  e.target.focus({ preventScroll: true });
  e.preventDefault();
  drawField();
});
$("field-canvas").addEventListener("pointermove", (e) => {
  if (fieldEd.drag < 0) return;
  const { V, src } = fieldPointer(e);
  const lim = 1.5;
  fieldEd.corners[fieldEd.drag] = [Math.min(Math.max(src[0] + fieldEd.grab[0], -V.w * lim), V.w * (1 + lim)),
                                   Math.min(Math.max(src[1] + fieldEd.grab[1], -V.h * lim), V.h * (1 + lim))];
  markDirty(); drawField();
});
for (const ev of ["pointerup", "pointercancel"]) $("field-canvas").addEventListener(ev, () => {
  if (fieldEd.drag < 0) return;
  fieldEd.drag = -1; fitMargin(); drawField(); requestPreview(true);
});
// точная доводка с клавиатуры: стрелки — 1 пиксель кадра, с Shift — 10; Tab — следующая точка
$("field-canvas").addEventListener("keydown", (e) => {
  if (!fieldEd.img || fieldEd.busy) return;
  const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
  if (e.key === "Tab") { fieldEd.sel = (fieldEd.sel + (e.shiftKey ? 3 : 1)) % 4; e.preventDefault(); drawField(); return; }
  if (/^[1-4]$/.test(e.key)) { fieldEd.sel = +e.key - 1; drawField(); return; }
  if (!d) return;
  e.preventDefault();
  const k = e.shiftKey ? 10 : 1, p = fieldEd.corners[fieldEd.sel];
  fieldEd.corners[fieldEd.sel] = [p[0] + d[0] * k, p[1] + d[1] * k];
  markDirty(); fitMargin(); drawField(); requestPreview();
});

function setZoom(z) {
  const wrap = $("field-wrap"), before = fieldEd.zoom;
  fieldEd.zoom = Math.min(Math.max(z, 1), 6);
  $("field-zoom-label").textContent = Math.round(fieldEd.zoom * 100) + " %";
  // держим в центре выбранную точку
  const cxRel = (wrap.scrollLeft + wrap.clientWidth / 2) / (wrap.scrollWidth || 1), cyRel = (wrap.scrollTop + wrap.clientHeight / 2) / (wrap.scrollHeight || 1);
  drawField();
  if (fieldEd.img && fieldEd.corners) {
    const V = fieldView(), [hx, hy] = V.toC(fieldEd.corners[fieldEd.sel]);
    const fx = before === fieldEd.zoom ? cxRel : hx / V.c.width, fy = before === fieldEd.zoom ? cyRel : hy / V.c.height;
    wrap.scrollLeft = fx * wrap.scrollWidth - wrap.clientWidth / 2; wrap.scrollTop = fy * wrap.scrollHeight - wrap.clientHeight / 2;
  }
  drawField();
}
$("field-zoom-in").onclick = () => setZoom(fieldEd.zoom * 1.5);
$("field-zoom-out").onclick = () => setZoom(fieldEd.zoom / 1.5);
$("field-full").onclick = () => {
  const el = $("field-editor");
  if (document.fullscreenElement) document.exitFullscreen(); else if (el.requestFullscreen) el.requestFullscreen().catch(() => {});
};
document.addEventListener("fullscreenchange", () => setTimeout(drawField, 50));
window.addEventListener("resize", () => { if (!$("field-editor").classList.contains("hidden")) drawField(); });

$("field-here").onclick = () => {
  const v = $("player");
  if (v && !v.paused) v.pause();
  openFieldEditor(v && v.src ? videoFrameIndex() : 0);
};
$("field-auto").onclick = () => suggestCorners(false);
$("field-expand").onclick = () => suggestCorners(true);
$("field-occ").onchange = () => drawField();
$("field-rotate").onclick = () => { fieldEd.rotation = (fieldEd.rotation + 1) % 4; markDirty(); drawField(); requestPreview(true); };
$("field-mode").onchange = () => {
  fieldEd.mode = $("field-mode").value; fieldEd.previewH = null; fieldEd.previewKey = "";
  if (fieldEd.mode === "sides" && fieldEd.img) {
    // ближние точки — внутрь кадра, на те же боковые линии
    const V = fieldView(), inside = ([x, y]) => x >= 0.04 * V.w && x <= 0.96 * V.w && y >= 0.04 * V.h && y <= 0.96 * V.h;
    for (const k of [0, 1]) {
      const far = fieldEd.corners[3 - k], near = fieldEd.corners[k];
      for (let t = 1; t > 0.05 && !inside(fieldEd.corners[k]); t -= 0.01) fieldEd.corners[k] = [far[0] + (near[0] - far[0]) * t, far[1] + (near[1] - far[1]) * t];
    }
  }
  renderModeHint(); markDirty(); fitMargin(); drawField(); requestPreview(true);
};
$("field-reset").onclick = () => { if (!fieldEd.img) return; const V = fieldView(); setCorners(defaultCorners(V.w, V.h), 0); };
$("field-close").onclick = () => {
  fieldEd.marks = fieldEd.marks.filter((m) => !m.unsaved || savedMarks().some((s) => s.frame === m.frame));
  $("field-editor").classList.add("hidden"); fieldEd.cur = -1; renderFieldMarks();
};
for (const id of ["field-own", "field-len", "field-wid"]) $(id).onchange = () => { $("field-dims-note").textContent = ""; drawField(); if (id !== "field-own") requestPreview(true); };

function fieldBusy(on, msg) {
  fieldEd.busy = on;
  $("field-editor").classList.toggle("field-saving", on);
  for (const id of ["field-save", "field-delete", "field-auto", "field-expand"]) $(id).disabled = on;
  if (msg != null) $("field-msg").textContent = msg;
}
async function putField(marks) {
  const [L, W] = fieldDims();
  const body = { field: marks.length ? { marks: marks.map((m) => ({ frame: m.frame, rotation: m.rotation, mode: m.mode || "corners",
    corners: m.corners.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]) })),
    own_goal: $("field-own").value || null, length_m: L, width_m: W } : null };
  currentJob = await api(`/api/jobs/${currentJob.id}/player`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}
$("field-save").onclick = async () => {
  if (!currentJob || !fieldEd.img || fieldEd.busy) return;
  markDirty();
  fieldBusy(true, "⏳ сохраняю и пересчитываю доску — несколько секунд…");
  try {
    await putField(fieldEd.marks);
    fieldEd.marks.forEach((m) => { m.unsaved = false; });
    const board = await reloadAfterField();
    const out = board && board.calibration && board.calibration.outside;
    const parts = ["поле сохранено"];
    if (out != null) parts.push(`за границей поля ${Math.round(100 * out)} % положений игроков${out > 0.12 ? " — многовато: проверьте дальнюю бровку или нажмите «Раздвинуть по игрокам»" : ""}`);
    if (!$("field-own").value) parts.push("укажите свои ворота — будет разбор по третям и владение");
    fieldBusy(false, parts.join(" · "));
    renderFieldCheck();
  } catch (err) { fieldBusy(false, "ошибка: " + err.message); }
};
$("field-delete").onclick = async () => {
  if (!currentJob || fieldEd.cur < 0 || fieldEd.busy) return;
  const m = fieldEd.marks[fieldEd.cur], wasSaved = savedMarks().some((s) => s.frame === m.frame);
  const rest = fieldEd.marks.filter((_, k) => k !== fieldEd.cur);
  if (wasSaved) {
    const last = !rest.some((x) => savedMarks().some((s) => s.frame === x.frame));
    if (!confirm(last ? "Удалить разметку поля? Доска вернётся к оценке по росту игроков." : `Удалить разметку на кадре ${m.frame}?`)) return;
    fieldBusy(true, "⏳ удаляю и пересчитываю доску…");
    try { await putField(rest.filter((x) => !x.unsaved)); } catch (err) { fieldBusy(false, "ошибка: " + err.message); return; }
  }
  fieldEd.marks = rest; fieldEd.cur = -1;
  $("field-editor").classList.add("hidden");
  if (wasSaved) { await reloadAfterField(); renderFieldCheck(); }
  fieldBusy(false, "разметка удалена");
  renderFieldMarks();
};

async function reloadAfterField() {
  try { vboxes = await api(`/api/jobs/${currentJob.id}/tracks`); } catch (_) {}
  fieldLinesCache = null;
  await loadBoard(currentJob);
  loadPlayerMetrics(currentJob); loadAnalysis(currentJob);
  renderFieldMarks(); redrawAll();
  return board;
}

// проверка разметки по ролику: четыре кадра из разных мест с линиями поля — видно, где они сползают
async function renderFieldCheck() {
  const box = $("field-check"), f = vboxes && vboxes.field;
  box.innerHTML = "";
  box.classList.toggle("hidden", !f);
  if (!f || !vboxes.cams) return;
  const total = vboxes.total || vboxes.cams.length, job = currentJob.id;
  const note = document.createElement("div"); note.className = "note";
  note.textContent = "Проверьте линии в разных местах ролика. Сползли — откройте этот кадр (щелчок) и нажмите «Отметить на текущем кадре».";
  box.appendChild(note);
  for (const q of [0.08, 0.36, 0.64, 0.92]) {
    const n = Math.round((total - 1) * q);
    const fig = document.createElement("figure"), cv = document.createElement("canvas"), cap = document.createElement("figcaption");
    cap.textContent = `кадр ${n} · ${fmtT(n / jobFps())}`;
    fig.append(cv, cap); box.appendChild(fig);
    fig.onclick = () => { seekFrame(n); $("video-wrap").scrollIntoView({ behavior: "smooth", block: "center" }); };
    fetch(`/api/jobs/${job}/frame/${n}`).then(async (r) => {
      if (!r.ok || !currentJob || currentJob.id !== job) return;
      const scale = parseFloat(r.headers.get("X-Scale")) || 1, img = await createImageBitmap(await r.blob());
      cv.width = 460; cv.height = Math.round(460 * img.height / img.width);
      const ctx = cv.getContext("2d"); ctx.drawImage(img, 0, 0, cv.width, cv.height);
      drawFieldLines(ctx, { x: 0, y: 0, k: cv.width / (img.width / scale) }, n, 1.4);
    }).catch(() => {});
  }
}

// линии поля поверх кадра idx: метры -> опорный кадр ближайшей разметки -> сцена -> кадр idx
let fieldLinesCache = null;
function drawFieldLines(ctx, r, idx, widthK = 1) {
  const f = vboxes && vboxes.field;
  if (!f || !vboxes.cams) return;
  const marks = f.marks || [];
  if (!marks.length) return;
  const L = f.length_m || formatDims()[0], W = f.width_m || formatDims()[1];
  const key = JSON.stringify([f, L, W]);
  if (!fieldLinesCache || fieldLinesCache.key !== key) {
    // H разметки («кадр -> метры») присылает сервер (в режиме sides её даёт модель камеры); нет — считаем по углам
    fieldLinesCache = { key, lines: pitchLines(L, W), fwd: marks.map((m) => m.H || null),
      H: marks.map((m) => (m.H ? inv3(m.H) : homographyJS(pitchCornersJS(L, W, m.rotation || 0), m.corners))) };
  }
  let k = 0;
  marks.forEach((m, i) => { if (Math.abs(m.frame - idx) < Math.abs(marks[k].frame - idx)) k = i; });
  const Hinv = fieldLinesCache.H[k];
  if (!Hinv) return;
  const T = frameToFrame(marks[k].frame, idx);
  const fwd = fieldLinesCache.fwd[k], sgn = fwd ? Math.sign(applyH(fwd, ...marks[k].corners[2])[2]) || 1 : 1;
  for (const ln of fieldLinesCache.lines) {
    ctx.strokeStyle = ln.goal ? goalColor(ln.goal, f.own_goal, 0.9) : "rgba(253,224,71,.55)";
    ctx.lineWidth = (ln.goal ? 4 : 1.5) * widthK;
    ctx.beginPath();
    let pen = false;
    for (const [x, y] of ln.pts) {
      const [u0, v0, w0] = applyH(Hinv, x, y);
      if (fwd ? applyH(fwd, u0, v0)[2] * sgn <= 0 : w0 <= 0) { pen = false; continue; }
      if (!isFinite(u0) || Math.abs(u0) > 1e5 || Math.abs(v0) > 1e5) { pen = false; continue; }
      const [u, v, w1] = applyH(T, u0, v0);
      if (w1 <= 0 || !isFinite(u) || !isFinite(v)) { pen = false; continue; }
      const X = r.x + u * r.k, Y = r.y + v * r.k;
      if (pen) ctx.lineTo(X, Y); else { ctx.moveTo(X, Y); pen = true; }
    }
    ctx.stroke();
  }
}
function drawFieldOnVideo(ctx, r, idx) {
  if ($("overlay-field").checked) drawFieldLines(ctx, r, idx);
}
$("overlay-field").onchange = () => drawOverlay();

$("roster-mode").onchange = () => rosterCall("mode", { mode: $("roster-mode").value });
