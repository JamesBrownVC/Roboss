/* V2R Factory demo frontend */
"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};
const fmt = (v, d = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));

const COLORS = ["#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#818cf8", "#f472b6", "#4ade80"];

/* MediaPipe 33-joint skeleton connections */
const POSE_EDGES = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
  [9, 10], [0, 7], [0, 8],
];

Chart.defaults.color = "#8ea0bf";
Chart.defaults.borderColor = "#1e2a42";
Chart.defaults.font.family = "Consolas, monospace";

/* ------------------------------------------------------------------ hero */
(function heroSkeleton() {
  const g = $("#heroSkel");
  g.innerHTML = `
    <circle cx="0" cy="-16" r="4.5" fill="none" stroke="#818cf8" stroke-width="2"/>
    <line x1="0" y1="-11" x2="0" y2="4" stroke="#818cf8" stroke-width="2"/>
    <line x1="0" y1="-7" x2="-10" y2="2" stroke="#818cf8" stroke-width="2" class="limb l1"/>
    <line x1="0" y1="-7" x2="10" y2="2" stroke="#818cf8" stroke-width="2" class="limb l2"/>
    <line x1="0" y1="4" x2="-7" y2="17" stroke="#818cf8" stroke-width="2" class="limb l3"/>
    <line x1="0" y1="4" x2="7" y2="17" stroke="#818cf8" stroke-width="2" class="limb l4"/>`;
  let t = 0;
  setInterval(() => {
    t += 0.12;
    const s = Math.sin(t) * 5;
    const lines = g.querySelectorAll(".limb");
    if (lines.length === 4) {
      lines[0].setAttribute("x2", -10 - s); lines[1].setAttribute("x2", 10 - s);
      lines[2].setAttribute("x2", -7 + s); lines[3].setAttribute("x2", 7 - s);
    }
  }, 50);
})();

async function loadOverview() {
  const o = await (await fetch("/api/overview")).json();
  const row = $("#statsRow");
  const stats = [
    [o.videos, "videos imported"],
    [o.timeseries, "timeseries files"],
    [o.workspaces, "episode workspaces"],
    [o.sessions, "multi-view sessions"],
    [o.stages, "pipeline stages"],
  ];
  for (const [v, k] of stats) {
    const s = el("div", "stat");
    s.append(el("div", "v", "0"), el("div", "k", k));
    row.append(s);
    animateCount(s.querySelector(".v"), v);
  }
}
function animateCount(node, target) {
  const t0 = performance.now(), dur = 1200;
  const tick = (now) => {
    const p = Math.min(1, (now - t0) / dur);
    node.textContent = Math.round(target * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ------------------------------------------------------- dataset explorer */
let VIDEOS = [];
async function loadVideos() {
  const data = await (await fetch("/api/videos")).json();
  VIDEOS = data.videos;
  $("#videosMock").hidden = !data.mock;
  const grid = $("#videoGrid");
  grid.innerHTML = "";
  for (const v of VIDEOS) {
    const card = el("div", "vcard reveal");
    card.dataset.id = v.id;
    if (v.url) {
      const vid = document.createElement("video");
      vid.src = v.url; vid.muted = true; vid.loop = true; vid.playsInline = true;
      vid.preload = "metadata";
      card.append(vid);
      card.addEventListener("mouseenter", () => vid.play().catch(() => {}));
      card.addEventListener("mouseleave", () => vid.pause());
    } else {
      card.append(el("div", "vthumb", v.subject === "human" ? "🕺" : "🐕"));
    }
    const body = el("div", "vcard-body");
    body.append(el("div", "vcard-name", v.filename));
    body.append(el("div", "vcard-meta",
      `${v.source_id} · ${v.duration_s}s · ${v.width}×${v.height} @ ${v.fps}fps · ${v.size_mb} MB`));
    const badges = el("div", "badges");
    badges.append(el("span", `badge ${v.subject}`, v.subject));
    if (v.has_timeseries) badges.append(el("span", "badge ts", "pose extracted"));
    if (v.mock) badges.append(el("span", "badge mock", "demo"));
    body.append(badges);
    card.append(body);
    card.addEventListener("click", () => selectVideo(v, card));
    grid.append(card);
  }
  observeReveals();
  const first = VIDEOS.find((v) => v.has_timeseries) || VIDEOS[0];
  if (first) selectVideo(first, grid.querySelector(`[data-id="${CSS.escape(first.id)}"]`));
}

/* -------------------------------------------------------- timeseries viewer */
let TS = null;           // current timeseries payload
let tsChart = null;
let overlayRAF = null;
let mockClock = { playing: false, t: 0, last: 0 };
let selectedJoints = new Set(["left_wrist", "right_wrist", "left_ankle", "right_ankle"]);

async function selectVideo(v, cardEl) {
  document.querySelectorAll(".vcard.selected").forEach((c) => c.classList.remove("selected"));
  if (cardEl) cardEl.classList.add("selected");

  $("#tsVideoTitle").textContent = `${v.subject === "human" ? "Skeleton overlay" : "Track overlay"} — ${v.filename}`;
  const video = $("#tsVideo");
  const placeholder = $("#stagePlaceholder");
  cancelAnimationFrame(overlayRAF);
  mockClock.playing = false;
  mockClock.t = 0;
  $("#btnPlay").textContent = "▶ Play";

  if (v.url) {
    video.src = v.url; video.hidden = false;
    placeholder.style.display = "none";
  } else {
    video.removeAttribute("src"); video.hidden = true;
    placeholder.style.display = "flex";
    placeholder.textContent = "synthetic preview (demo data)";
  }

  TS = await (await fetch(`/api/timeseries/${v.subject}/${encodeURIComponent(v.stem)}`)).json();
  TS._video = v;
  $("#tsMock").hidden = !TS.mock;
  $("#tsMeta").textContent =
    `${TS.n_frames} frames @ ${TS.fps} Hz · ${TS.subject} · ${TS.mock ? "synthetic demo signal" : "extracted from video"}`;

  buildChart();
  buildJointPicker();
  startOverlay();
}

$("#btnPlay").addEventListener("click", () => {
  const video = $("#tsVideo");
  if (TS && TS._video && TS._video.url) {
    if (video.paused) { video.play(); $("#btnPlay").textContent = "❚❚ Pause"; }
    else { video.pause(); $("#btnPlay").textContent = "▶ Play"; }
  } else {
    mockClock.playing = !mockClock.playing;
    mockClock.last = performance.now();
    $("#btnPlay").textContent = mockClock.playing ? "❚❚ Pause" : "▶ Play";
  }
});

function stageMapping(canvas) {
  /* map normalized video coords -> canvas px, accounting for object-fit: contain */
  const video = $("#tsVideo");
  const cw = canvas.width, ch = canvas.height;
  let vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  if (!TS._video.url) { vw = 16; vh = 9; }
  const scale = Math.min(cw / vw, ch / vh);
  const w = vw * scale, h = vh * scale;
  const ox = (cw - w) / 2, oy = (ch - h) / 2;
  return (nx, ny) => [ox + nx * w, oy + ny * h];
}

function frameAt(time) {
  if (!TS || !TS.frames.length) return null;
  const idx = Math.min(TS.frames.length - 1, Math.max(0, Math.round(time * TS.fps)));
  return TS.frames[idx];
}

function startOverlay() {
  const canvas = $("#overlayCanvas");
  const stage = $("#videoStage");
  const video = $("#tsVideo");
  const ctx = canvas.getContext("2d");

  const draw = () => {
    canvas.width = stage.clientWidth * devicePixelRatio;
    canvas.height = stage.clientHeight * devicePixelRatio;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!TS) { overlayRAF = requestAnimationFrame(draw); return; }

    let t;
    if (TS._video.url) {
      t = video.currentTime;
    } else {
      const now = performance.now();
      if (mockClock.playing) mockClock.t += (now - mockClock.last) / 1000;
      mockClock.last = now;
      const dur = TS.n_frames / TS.fps;
      t = mockClock.t % Math.max(0.1, dur);
    }

    /* draw overlay only when data matches source: real parquet over real video,
       or synthetic walker on the empty stage */
    const overlayOk = (!TS.mock && TS._video.url) || (!TS._video.url);
    const fr = frameAt(t);
    if (fr && overlayOk) {
      const map = stageMapping(canvas);
      if (TS.subject === "human") drawSkeleton(ctx, fr, map);
      else drawTracks(ctx, fr, map, canvas);
    }
    drawTimeCursor(t);
    overlayRAF = requestAnimationFrame(draw);
  };
  cancelAnimationFrame(overlayRAF);
  overlayRAF = requestAnimationFrame(draw);
}

function drawSkeleton(ctx, fr, map) {
  const J = fr.joints;
  ctx.lineWidth = 2.5 * devicePixelRatio;
  ctx.lineCap = "round";
  for (const [a, b] of POSE_EDGES) {
    const pa = J[a], pb = J[b];
    if (!pa || !pb || pa[0] === null || pb[0] === null) continue;
    if (pa[3] < 0.3 || pb[3] < 0.3) continue;
    const [x1, y1] = map(pa[0], pa[1]);
    const [x2, y2] = map(pb[0], pb[1]);
    const g = ctx.createLinearGradient(x1, y1, x2, y2);
    g.addColorStop(0, "#22d3ee"); g.addColorStop(1, "#a78bfa");
    ctx.strokeStyle = g;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  for (const p of J) {
    if (!p || p[0] === null || p[3] < 0.3) continue;
    const [x, y] = map(p[0], p[1]);
    ctx.fillStyle = "#e6ecf7";
    ctx.shadowColor = "#22d3ee"; ctx.shadowBlur = 6 * devicePixelRatio;
    ctx.beginPath(); ctx.arc(x, y, 3 * devicePixelRatio, 0, 7); ctx.fill();
    ctx.shadowBlur = 0;
  }
}

function drawTracks(ctx, fr, map) {
  for (let i = 0; i < (fr.entities || []).length; i++) {
    const e = fr.entities[i];
    const [cx, cy] = map(e.cx, e.cy);
    const [x0, y0] = map(e.cx - e.w / 2, e.cy - e.h / 2);
    const [x1, y1] = map(e.cx + e.w / 2, e.cy + e.h / 2);
    const color = COLORS[e.entity_id % COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5 * devicePixelRatio;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.fillStyle = color;
    ctx.font = `${12 * devicePixelRatio}px Consolas, monospace`;
    ctx.fillText(`${e.class_name} #${e.entity_id} ${(e.conf * 100).toFixed(0)}%`,
      x0, Math.max(12 * devicePixelRatio, y0 - 5 * devicePixelRatio));
    /* velocity vector */
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + e.vx * 900, cy + e.vy * 900); ctx.stroke();
  }
}

let cursorPlugin = null;
let lastCursorUpdate = 0;
function drawTimeCursor(t) {
  if (!tsChart) return;
  const now = performance.now();
  if (now - lastCursorUpdate < 80) return;   // ~12 fps is plenty for the cursor
  lastCursorUpdate = now;
  tsChart.$cursorT = t;
  tsChart.update("none");
}

const CHART_JOINTS = ["nose", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist",
  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"];

function buildJointPicker() {
  const picker = $("#jointPicker");
  picker.innerHTML = "";
  if (!TS || TS.subject !== "human") return;
  for (const j of CHART_JOINTS) {
    const chip = el("button", "jchip" + (selectedJoints.has(j) ? " on" : ""), j);
    chip.addEventListener("click", () => {
      selectedJoints.has(j) ? selectedJoints.delete(j) : selectedJoints.add(j);
      chip.classList.toggle("on");
      buildChart();
    });
    picker.append(chip);
  }
}

function buildChart() {
  if (tsChart) { tsChart.destroy(); tsChart = null; }
  if (!TS) return;
  const ctx = $("#tsChart").getContext("2d");
  const times = TS.frames.map((f) => f.t);
  let datasets = [];

  if (TS.subject === "human") {
    $("#tsChartTitle").textContent = "Joint trajectories (normalized y over time)";
    let ci = 0;
    for (const jname of CHART_JOINTS) {
      if (!selectedJoints.has(jname)) continue;
      const ji = TS.joint_names.indexOf(jname);
      datasets.push({
        label: jname,
        data: TS.frames.map((f) => {
          const p = f.joints[ji];
          return p && p[0] !== null ? p[1] : null;
        }),
        borderColor: COLORS[ci % COLORS.length],
        backgroundColor: "transparent",
        borderWidth: 1.8, pointRadius: 0, tension: 0.25, spanGaps: true,
      });
      ci++;
    }
  } else {
    $("#tsChartTitle").textContent = "Track center + speed over time";
    const ids = [...new Set(TS.frames.flatMap((f) => (f.entities || []).map((e) => e.entity_id)))].slice(0, 4);
    ids.forEach((id, ci) => {
      const grab = (key) => TS.frames.map((f) => {
        const e = (f.entities || []).find((x) => x.entity_id === id);
        return e ? e[key] : null;
      });
      datasets.push({ label: `#${id} cx`, data: grab("cx"), borderColor: COLORS[ci * 2 % COLORS.length],
        borderWidth: 1.8, pointRadius: 0, tension: 0.25, spanGaps: true });
      datasets.push({ label: `#${id} cy`, data: grab("cy"), borderColor: COLORS[(ci * 2 + 1) % COLORS.length],
        borderWidth: 1.8, pointRadius: 0, tension: 0.25, spanGaps: true, borderDash: [5, 4] });
    });
  }

  cursorPlugin = {
    id: "timeCursor",
    afterDraw(chart) {
      const t = chart.$cursorT;
      if (t === undefined) return;
      const xs = chart.scales.x;
      const px = xs.getPixelForValue(t);
      if (px < xs.left || px > xs.right) return;
      const c = chart.ctx;
      c.save();
      c.strokeStyle = "#e6ecf766"; c.lineWidth = 1; c.setLineDash([4, 4]);
      c.beginPath(); c.moveTo(px, chart.chartArea.top); c.lineTo(px, chart.chartArea.bottom); c.stroke();
      c.restore();
    },
  };

  tsChart = new Chart(ctx, {
    type: "line",
    data: { labels: times, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { type: "linear", title: { display: true, text: "time (s)" }, grid: { color: "#141d31" } },
        y: { reverse: TS.subject === "human", grid: { color: "#141d31" } },
      },
      plugins: { legend: { labels: { boxWidth: 12, font: { size: 10 } } } },
    },
    plugins: [cursorPlugin],
  });
}

/* --------------------------------------------------------- feasibility */
async function loadFeasibility() {
  const data = await (await fetch("/api/feasibility")).json();
  $("#feasMock").hidden = !data.mock;
  const grid = $("#feasGrid");
  grid.innerHTML = "";
  for (const r of data.reports) {
    const card = el("div", "panel feas-card reveal");
    const stripeColor = { proceed: "#34d399", reject: "#f87171", human_review: "#fbbf24" }[r.recommendation] || "#8ea0bf";
    const stripe = el("div", "verdict-stripe"); stripe.style.background = stripeColor;
    card.append(stripe);

    const head = el("div", "feas-head");
    head.append(el("div", "feas-title", r.episode_id + (r.mock ? ' <span class="badge mock">demo</span>' : "")));
    head.append(el("span", `verdict ${r.recommendation}`, r.recommendation.replace("_", " ")));
    card.append(head);

    const flags = el("div", "feas-flags");
    flags.append(el("span", "flag " + (r.physically_plausible ? "ok" : ""),
      (r.physically_plausible ? "✓" : "✗") + " physically_plausible"));
    flags.append(el("span", "flag " + (r.tracking_likely_valid ? "ok" : ""),
      (r.tracking_likely_valid ? "✓" : "✗") + " tracking_valid"));
    for (const a of r.ai_generated_artifacts || []) flags.append(el("span", "flag", "⚠ " + a));
    card.append(flags);

    const gauges = el("div", "gauges");
    gauges.append(gauge("confidence", r.confidence, "#22d3ee"));
    gauges.append(gauge("violations", r.physics_violation_frame_ratio, "#f87171", true));
    const pc = r.physics_checks || {};
    if (pc.vel_spike_ratio !== undefined) gauges.append(gauge("vel spikes", pc.vel_spike_ratio, "#fbbf24", true));
    if (pc.foot_slide_ratio !== undefined) gauges.append(gauge("foot slide", pc.foot_slide_ratio, "#a78bfa", true));
    card.append(gauges);

    if (r.notes) card.append(el("div", "feas-notes", `“${r.notes}” — judge: ${r.judge_source}`));
    grid.append(card);
  }
  observeReveals();
}

function gauge(label, value, color, inverse = false) {
  const wrap = el("div", "gauge");
  const canvas = document.createElement("canvas");
  canvas.width = 172; canvas.height = 120;
  wrap.append(canvas, el("div", "glabel", label));
  const ctx = canvas.getContext("2d");
  const v = Math.max(0, Math.min(1, value || 0));
  let p = 0;
  const target = v;
  const tick = () => {
    p = Math.min(target, p + Math.max(0.004, target / 40));
    ctx.clearRect(0, 0, 172, 120);
    ctx.lineWidth = 13; ctx.lineCap = "round";
    ctx.strokeStyle = "#141d31";
    ctx.beginPath(); ctx.arc(86, 100, 64, Math.PI, 2 * Math.PI); ctx.stroke();
    ctx.strokeStyle = color;
    ctx.beginPath(); ctx.arc(86, 100, 64, Math.PI, Math.PI * (1 + p)); ctx.stroke();
    ctx.fillStyle = "#e6ecf7"; ctx.font = "700 24px Consolas, monospace"; ctx.textAlign = "center";
    ctx.fillText(inverse ? (p * 100).toFixed(1) + "%" : (p * 100).toFixed(0) + "%", 86, 96);
    if (p < target - 1e-4) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  return wrap;
}

/* ----------------------------------------------------------- multi-view */
async function loadMultiview() {
  const data = await (await fetch("/api/multiview")).json();
  $("#mvMock").hidden = !data.mock;
  const container = $("#mvContainer");
  container.innerHTML = "";

  for (const s of data.sessions) {
    const wrap = el("div", "panel mv-session reveal");
    const title = el("div", "panel-title",
      `session: ${s.session_id} · tier <b style="color:#34d399">${s.tier}</b>` +
      (s.mock ? ' <span class="badge mock">demo</span>' : ""));
    wrap.append(title);

    const grid = el("div", "mv-grid");
    const stack = el("div", "cams-stack");
    const syncByCam = {};
    for (const c of (s.sync && s.sync.cameras) || []) syncByCam[c.cam_id] = c.offset_s;
    for (const cam of s.cameras) {
      const tile = el("div", "cam-tile");
      tile.append(el("div", "cam-icon", "🎥"));
      const info = el("div");
      info.append(el("div", "cam-name", cam));
      const off = syncByCam[cam];
      info.append(el("div", "cam-sub",
        `sync offset: ${off !== undefined ? (off >= 0 ? "+" : "") + (off * 1000).toFixed(1) + " ms" : "—"}` +
        ` · calib: ${s.calibration ? s.calibration.method : "—"}`));
      tile.append(info);
      stack.append(tile);
    }
    grid.append(stack);

    const right = el("div");
    const r = s.reproj || {};
    const kpis = el("div", "mv-kpis");
    const kpi = (k, v, cls = "accent") => {
      const d = el("div", `kpi ${cls}`);
      d.append(el("div", "v", v), el("div", "k", k));
      return d;
    };
    kpis.append(kpi("mean reproj error", fmt(r.mean_reproj_error_px) + " px"));
    kpis.append(kpi("p95 reproj error", fmt(r.p95_reproj_error_px) + " px"));
    if (r.monocular_shadow_mean_px != null)
      kpis.append(kpi("monocular shadow", fmt(r.monocular_shadow_mean_px) + " px", ""));
    if (r.triangulation_wins != null)
      kpis.append(kpi("triangulation wins", r.triangulation_wins ? "YES" : "NO", r.triangulation_wins ? "good" : ""));
    right.append(kpis);

    const cw = el("div", "chart-wrap"); cw.style.height = "240px";
    const canvas = document.createElement("canvas");
    cw.append(canvas); right.append(cw);
    grid.append(right);
    wrap.append(grid);
    container.append(wrap);

    const pf = r.per_frame || [];
    const datasets = [{
      label: "cross-view reproj error (px)",
      data: pf.map((p) => p.mean_error_px),
      borderColor: "#22d3ee", backgroundColor: "#22d3ee18",
      fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3,
    }];
    if (r.monocular_shadow_mean_px != null) {
      datasets.push({
        label: "monocular shadow (mean)",
        data: pf.map(() => r.monocular_shadow_mean_px),
        borderColor: "#f87171", borderDash: [6, 5], borderWidth: 1.5, pointRadius: 0,
      });
    }
    new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { labels: pf.map((p) => p.frame), datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "frame" }, grid: { color: "#141d31" } },
          y: { title: { display: true, text: "px" }, grid: { color: "#141d31" }, beginAtZero: true },
        },
        plugins: { legend: { labels: { boxWidth: 12, font: { size: 10 } } } },
      },
    });
  }
  observeReveals();
}

/* -------------------------------------------------------------- funnel */
const STAGE_LABELS = {
  ingest: "ingested", feasibility_judge: "feasibility_ok", geometry: "geometry_ok",
  human_body: "body_ok", hands: "hands_ok", objects: "objects_ok", contact: "contact_ok",
  semantics: "semantics_ok", retarget: "retarget_ok", physics_validate: "physics_ok",
  qa: "qa_ok", package: "exported",
};

async function loadFunnel() {
  const data = await (await fetch("/api/yield")).json();
  $("#funnelMock").hidden = !data.mock;
  const bars = $("#funnelBars");
  bars.innerHTML = "";
  const max = Math.max(1, ...data.funnel.map((f) => f.count));
  const rows = [];
  for (const f of data.funnel) {
    const row = el("div", "frow");
    row.append(el("div", "fname", STAGE_LABELS[f.stage] || f.stage));
    const track = el("div", "fbar-track");
    const bar = el("div", "fbar");
    track.append(bar);
    row.append(track);
    row.append(el("div", "fcount", String(f.count)));
    bars.append(row);
    rows.push([bar, (100 * f.count) / max]);
  }
  /* animate when scrolled into view */
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      rows.forEach(([bar, w], i) => setTimeout(() => (bar.style.width = w + "%"), i * 70));
      io.disconnect();
    }
  }, { threshold: 0.25 });
  io.observe(bars);

  const eps = $("#funnelEpisodes");
  eps.innerHTML = "";
  for (const e of data.episodes) {
    const row = el("div", "ep-row");
    row.append(el("span", "ep-name", e.episode_id));
    for (const stage of Object.keys(STAGE_LABELS)) {
      const pip = el("span", `stage-pip ${e.stages[stage] || ""}`);
      pip.title = `${stage}: ${e.stages[stage] || "pending"}`;
      row.append(pip);
    }
    if (e.failure_stage)
      row.append(el("div", "ep-fail", `✗ gated at ${e.failure_stage}: ${e.failure_reason || ""}`));
    eps.append(row);
  }
}

/* -------------------------------------------------------------- exports */
async function loadExports() {
  const data = await (await fetch("/api/exports")).json();
  $("#expMock").hidden = !data.mock;
  const grid = $("#exportGrid");
  grid.innerHTML = "";
  for (const x of data.exports) {
    const card = el("div", "panel exp-card reveal");
    const head = el("div", "exp-head");
    head.append(el("div", "exp-name", "📦 " + x.episode_id + (x.mock ? ' <span class="badge mock">demo</span>' : "")));
    head.append(el("span", `tier-badge ${x.tier}`, x.tier));
    card.append(head);
    if (x.tier_description) card.append(el("div", "exp-desc", x.tier_description));
    const kv = el("div", "exp-kv");
    const pairs = [
      ["format", x.format],
      ["robots", (x.robots || []).join(", ") || "—"],
      ["features", String(x.n_features)],
      ["provenance", x.synthetic ? "synthetic (CI mode)" : "estimated / measured"],
    ];
    for (const [k, v] of pairs) { kv.append(el("div", "k", k)); kv.append(el("div", "v", v)); }
    card.append(kv);
    if (x.features && x.features.length) {
      const fl = el("div", "feat-list");
      for (const f of x.features) fl.append(el("span", "feat", f));
      card.append(fl);
    }
    grid.append(card);
  }
  observeReveals();
}

/* --------------------------------------------------------------- reveal */
let revealIO = null;
function observeReveals() {
  if (!revealIO) {
    revealIO = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { e.target.classList.add("vis"); revealIO.unobserve(e.target); }
    }, { threshold: 0.12 });
  }
  document.querySelectorAll(".reveal:not(.vis)").forEach((n) => revealIO.observe(n));
}

/* ------------------------------------------------------------------ boot */
loadOverview();
loadVideos();
loadFeasibility();
loadMultiview();
loadFunnel();
loadExports();
