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
  point = null; $("sel-point").value = ""; $("marker").classList.add("hidden"); $("start-btn").disabled = true;
}
$("sel-slider").oninput = () => { $("sel-time").value = $("sel-slider").value; };
$("sel-slider").onchange = loadFrame;
$("sel-time").onchange = () => { $("sel-slider").value = $("sel-time").value; loadFrame(); };

$("frame-img").onclick = (e) => {
  const img = e.target, rect = img.getBoundingClientRect();
  const kx = img.naturalWidth / rect.width, ky = img.naturalHeight / rect.height;
  const px = (e.clientX - rect.left) * kx, py = (e.clientY - rect.top) * ky;   // пиксели уменьшенного кадра
  point = [px / frameScale, py / frameScale];                                   // -> исходный кадр
  $("sel-point").value = `${point[0].toFixed(0)}, ${point[1].toFixed(0)}`;
  const m = $("marker");
  m.style.left = (e.clientX - rect.left) + "px"; m.style.top = (e.clientY - rect.top) + "px";
  m.classList.remove("hidden");
  $("start-btn").disabled = false;
};

$("start-btn").onclick = async () => {
  if (!currentVideo || !point) return;
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
    },
  };
  if (body.init.t_sec < 0) { $("start-msg").textContent = "Момент выбора раньше начала фрагмента"; return; }
  $("start-btn").disabled = true; $("start-msg").textContent = "запуск…";
  try {
    const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $("start-msg").textContent = "задача " + job.id;
    await loadJobs();
    openJob(job.id);
  } catch (err) {
    $("start-msg").textContent = "ошибка: " + err.message;
  } finally { $("start-btn").disabled = !point; }
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
  $("job-progress").textContent = `${j.progress}${j.total ? "/" + j.total : ""} кадров · ${j.fps_proc} кадр/с · ${j.state_now}`;
  $("cancel-btn").classList.toggle("hidden", !(j.status === "running" || j.status === "queued"));
  $("delete-btn").classList.toggle("hidden", j.status === "running" || j.status === "queued");
  $("job-error").classList.toggle("hidden", !j.error);
  $("job-error").textContent = j.error || "";
  renderEvents(j.events);
  if (j.status === "running" || j.status === "queued") { if (!pollTimer) startPolling(); }
  else if (!(j.status === "done" && wasDone && !force)) {
    if (j.status === "done") await renderResult(j);
    const anyRunning = (await api("/api/jobs")).some((x) => x.status === "running" || x.status === "queued");
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

async function renderResult(j) {
  $("result-body").classList.remove("hidden");
  const base = `/api/jobs/${j.id}/files/`;
  const v = $("player");
  if (j.options.render) { v.classList.remove("hidden"); if (!v.src.endsWith(base + "annotated.mp4")) v.src = base + "annotated.mp4"; }
  else v.classList.add("hidden");
  kv($("summary"), j.summary || {});
  renderMetrics(j.metrics || {});
  kv($("evaluation"), j.evaluation || {});
  $("downloads").innerHTML = ["track.json", "track.csv", "events.log", "metrics.json", "frame_stats.csv", "errors.md", "errors.jsonl", "evaluation.json", "annotated.mp4"]
    .map((n) => `<a href="${base}${n}" download>${n}</a>`).join("");
  await Promise.all([renderTimeline(j), renderErrors(j)]);
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
    const idx = Math.floor((e.clientX - rect.left) / rect.width * data.total);
    seek(idx / data.meta.fps);
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

loadVideos(); loadJobs();
