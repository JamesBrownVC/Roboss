/* V2R Factory — single linear "watch the factory work" flow */
"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v) => (v === null || v === undefined ? "—" : (v * 100).toFixed(0) + "%");
const fmt = (v, d = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));

Chart.defaults.color = "#8ea0bf";
Chart.defaults.borderColor = "#1e2a42";
Chart.defaults.font.family = "Consolas, monospace";

const PHASE_ORDER = ["starting", "requested", "generating", "generated",
  "verifying", "verified", "ingesting", "ingested", "delivered"];
const rank = (p) => PHASE_ORDER.indexOf(p);

const V2R_STAGES = ["ingest", "feasibility_judge", "geometry", "human_body",
  "hands", "objects", "contact", "semantics", "retarget", "physics_validate",
  "qa", "package"];

/* ------------------------------------------------------------------ state */
let currentJobId = null;
let pollTimer = null;
let stageSigs = {};            // stage id -> content signature (skip rebuilds)
const trajDone = new Set();    // episode ids whose trajectory chart is drawn

/* ------------------------------------------------------------- stage defs */
const STAGES = [
  {
    id: "director", num: "01", icon: "🎬", title: "Director — prompt expansion",
    sub: "Gemini turns one sentence into a multi-camera, multi-event shot plan",
    status(j) {
      if (j.variants.length) return "done";
      return j.runner && j.runner.running ? "active" : "pending";
    },
    sig: (j) => j.variants.map((v) => v.variant_id).join() + j.director,
    render: renderDirector,
  },
  {
    id: "generation", num: "02", icon: "📼", title: "Video generation",
    sub: "each planned shot is rendered by the video backend",
    status(j) {
      if (!j.variants.length) return "pending";
      if (rank(j.phase) >= rank("generated") ||
          j.variants.every((v) => v.generated)) return "done";
      return j.phase === "generating" ? "active" : "pending";
    },
    sig: (j) => j.variants.map((v) => (v.video_url || "-") + (v.gen_error || "")).join() + j.phase,
    render: renderGeneration,
  },
  {
    id: "verification", num: "03", icon: "🔍", title: "Agentic verification",
    sub: "VLM judge + physics tools accept or reject every generated video",
    status(j) {
      if (rank(j.phase) >= rank("verified")) return "done";
      return j.phase === "verifying" ? "active" : "pending";
    },
    sig: (j) => j.variants.map((v) => (v.verdict || "-") + (v.vlm_notes || "").length).join() + j.phase,
    render: renderVerification,
  },
  {
    id: "pipeline", num: "04", icon: "🏭", title: "V2R pipeline",
    sub: "accepted videos run the full 12-stage factory (geometry → retarget → QA)",
    status(j) {
      if (rank(j.phase) >= rank("ingested")) {
        // nothing passed verification: this stage was skipped, not completed
        return j.pipeline.length ? "done" : "skipped";
      }
      return j.phase === "ingesting" ? "active" : "pending";
    },
    sig: (j) => JSON.stringify(j.pipeline) + j.phase + (j.accepted ?? ""),
    render: renderPipeline,
  },
  {
    id: "multiview", num: "05", icon: "📐", title: "Multi-view triangulation",
    sub: "same event, ≥2 cameras → measured cross-view reprojection error",
    skip: (j) => j.n_cameras < 2 && !j.reproj.length,
    status(j) {
      if (j.reproj.length && rank(j.phase) >= rank("ingested")) return "done";
      if (rank(j.phase) >= rank("ingested")) return "skipped";
      return j.phase === "ingesting" ? "active" : "pending";
    },
    sig: (j) => JSON.stringify(j.reproj) + j.phase,
    render: renderMultiview,
  },
  {
    id: "delivery", num: "06", icon: "📦", title: "Delivery — robot training data",
    sub: "feasibility-filtered LeRobot episodes, tier-tagged and ready to train on",
    status(j) {
      if (j.phase === "delivered") return "done";
      return j.phase === "ingested" ? "active" : "pending";
    },
    sig: (j) => j.phase + j.episodes.join() + j.episodes_detail.length +
      j.variants.map((v) => v.verdict || "-").join() + (j.dataset_card || "").length,
    render: renderDelivery,
  },
];

/* --------------------------------------------------------------- skeleton */
function buildSkeleton(job) {
  const root = $("#stages");
  root.innerHTML = "";
  stageSigs = {};
  trajDone.clear();
  for (const st of STAGES) {
    const sec = el("section", "stage pending");
    sec.id = "stage_" + st.id;
    sec.innerHTML = `
      <div class="stage-rail"><div class="stage-dot">${st.icon}</div><div class="stage-line"></div></div>
      <div class="stage-body">
        <div class="stage-head">
          <span class="stage-num">${st.num}</span>
          <h2>${st.title}</h2>
          <span class="stage-state"></span>
        </div>
        <p class="stage-sub">${st.sub}</p>
        <div class="stage-content"></div>
      </div>`;
    root.append(sec);
  }
}

/* ------------------------------------------------------------ stage render */
function renderDirector(j, box) {
  box.innerHTML = "";
  const meta = el("div", "chip-row");
  meta.append(el("span", "chip",
    `director: <b>${j.director === "gemini" ? "Gemini (LLM)" : "deterministic"}</b>`));
  meta.append(el("span", "chip", `${j.n_events} event${j.n_events > 1 ? "s" : ""} × ${j.n_cameras} camera${j.n_cameras > 1 ? "s" : ""} = <b>${j.variants.length} videos planned</b>`));
  box.append(meta);
  if (j.synthetic_scenario) {
    const s = j.synthetic_scenario;
    const scenario = el("div", "mini-card pop");
    scenario.append(el("div", "mini-title", `synthetic scenario · ${esc(s.scenario_id)}`));
    scenario.append(el("div", "mini-prompt",
      `${esc(s.subject)} · ${esc(s.motion)} · ${esc(s.scene)} · source=${esc(s.source)}`));
    const chips = el("div", "chip-row tight");
    for (const label of s.expected_labels || []) {
      chips.append(el("span", "chip sm skill", esc(label)));
    }
    if (s.synthetic_controls) {
      chips.append(el("span", "chip sm", `generation ${esc(s.synthetic_controls.generation)}`));
      chips.append(el("span", "chip sm", esc(s.synthetic_controls.video_asset)));
      chips.append(el("span", "chip sm", `robot ${esc(s.synthetic_controls.retarget_robot)}`));
    }
    scenario.append(chips);
    box.append(scenario);
  }
  const camsById = {};
  for (const c of j.cameras || []) camsById[c.cam_id] = c;
  const grid = el("div", "card-grid");
  for (const v of j.variants) {
    const c = camsById[v.cam_id] || {};
    const card = el("div", "mini-card pop");
    card.append(el("div", "mini-title",
      `${esc(v.variant_id)} <span class="dim">· ${esc(v.cam_id)}</span>`));
    if (c.description) card.append(el("div", "mini-cam", "🎥 " + esc(c.description)));
    const chips = el("div", "chip-row tight");
    if (c.height_m != null) chips.append(el("span", "chip sm", `h ${c.height_m}m`));
    if (c.distance_m != null) chips.append(el("span", "chip sm", `d ${c.distance_m}m`));
    if (c.azimuth_deg != null) chips.append(el("span", "chip sm", `az ${c.azimuth_deg}°`));
    if (c.fov_deg != null) chips.append(el("span", "chip sm", `fov ${c.fov_deg}°`));
    if (v.duration_s) chips.append(el("span", "chip sm", `${v.duration_s}s`));
    card.append(chips);
    card.append(el("div", "mini-prompt", esc(v.prompt)));
    grid.append(card);
  }
  box.append(grid);
}

function renderGeneration(j, box) {
  box.innerHTML = "";
  const grid = el("div", "vid-grid");
  for (const v of j.variants) {
    const slot = el("div", "vid-slot pop");
    if (v.video_url) {
      const vid = document.createElement("video");
      vid.src = v.video_url; vid.muted = true; vid.loop = true;
      vid.playsInline = true; vid.controls = true; vid.preload = "metadata";
      slot.append(vid);
    } else if (v.gen_error) {
      slot.append(el("div", "vid-wait err", "✗ " + esc(v.gen_error.slice(0, 80))));
    } else {
      slot.append(el("div", "vid-wait",
        `<span class="spin"></span> generating with ${esc(v.backend || j.backend || "backend")}… (~40s/clip)`));
    }
    slot.append(el("div", "vid-cap",
      `${esc(v.variant_id)} · ${esc(v.backend || j.backend || "?")}`));
    grid.append(slot);
  }
  box.append(grid);
}

function renderVerification(j, box) {
  box.innerHTML = "";
  const grid = el("div", "card-grid");
  for (const v of j.variants) {
    const card = el("div", "mini-card pop" + (v.verdict ? "" : " dim-card"));
    const head = el("div", "verify-head");
    head.append(el("span", "mini-title", esc(v.variant_id)));
    const badgeCls = v.verdict === "accept" ? "ok" : v.verdict === "reject" ? "bad" : "wait";
    head.append(el("span", `verdict-badge ${badgeCls}`,
      v.verdict ? esc(v.verdict) : "pending"));
    card.append(head);
    if (v.vlm_notes) {
      card.append(el("div", "vlm-notes",
        `“${esc(v.vlm_notes)}”` +
        (v.vlm_confidence != null ? ` <span class="dim">· VLM confidence ${pct(v.vlm_confidence)}${v.vlm_judge ? " · " + esc(v.vlm_judge) : ""}</span>` : "")));
    }
    if (v.physics) {
      const chips = el("div", "chip-row tight");
      chips.append(el("span", "chip sm " + (v.physics.physics_ok ? "good" : "warn"),
        "physics " + (v.physics.physics_ok ? "✓" : "✗")));
      if (v.physics.flow_consistency != null)
        chips.append(el("span", "chip sm", `flow consistency ${pct(v.physics.flow_consistency)}`));
      if (v.physics.velocity_spike_ratio != null)
        chips.append(el("span", "chip sm", `vel spikes ${pct(v.physics.velocity_spike_ratio)}`));
      if (v.physics.pose_detection_rate != null)
        chips.append(el("span", "chip sm", `pose detect ${pct(v.physics.pose_detection_rate)}`));
      card.append(chips);
    }
    if (v.skills && v.skills.length) {
      const chips = el("div", "chip-row tight");
      for (const s of v.skills) chips.append(el("span", "chip sm skill", esc(s)));
      card.append(chips);
    }
    grid.append(card);
  }
  box.append(grid);
}

function renderPipeline(j, box) {
  box.innerHTML = "";
  if (!j.pipeline.length) {
    if (rank(j.phase) >= rank("ingested")) {
      const n = j.variants.filter((v) => v.verdict === "reject").length;
      box.append(el("div", "dim",
        `⊘ skipped — no videos passed verification${n ? ` (${n} rejected by the feasibility gate)` : ""}, so nothing entered the factory.`));
    } else {
      box.append(el("div", "dim", "waiting for accepted videos to enter the pipeline…"));
    }
    return;
  }
  const wrap = el("div", "pipe-rows pop");
  for (const ep of j.pipeline) {
    const row = el("div", "pipe-row");
    row.append(el("span", "pipe-name", esc(ep.episode_id)));
    const pips = el("span", "pips");
    for (const st of V2R_STAGES) {
      const s = ep.stages[st] || "pending";
      const pip = el("span", `pip ${s}`);
      pip.title = `${st}: ${s}`;
      pips.append(pip);
    }
    row.append(pips);
    row.append(el("span", "pipe-flag " + (ep.accepted ? "ok" : ep.accepted === false ? "bad" : ""),
      ep.accepted ? "✓ accepted" : ep.accepted === false
        ? `✗ ${esc(ep.failure_stage || "rejected")}` : "…"));
    wrap.append(row);
  }
  box.append(wrap);
  box.append(el("div", "pipe-legend dim",
    V2R_STAGES.join(" · ")));
}

function renderMultiview(j, box) {
  box.innerHTML = "";
  if (!j.reproj.length) {
    box.append(el("div", "dim", rank(j.phase) >= rank("ingested")
      ? "⊘ skipped — needs ≥2 accepted cameras of the same event to triangulate."
      : "triangulating across cameras…"));
    return;
  }
  for (const r of j.reproj) {
    const row = el("div", "kpi-row pop");
    row.append(el("div", "kpi big", `<div class="v">${fmt(r.mean_reproj_error_px)} px</div><div class="k">mean reprojection error</div>`));
    row.append(el("div", "kpi big", `<div class="v">${fmt(r.p95_reproj_error_px)} px</div><div class="k">p95 reprojection error</div>`));
    row.append(el("div", "kpi", `<div class="v">${r.n_frames ?? "—"}</div><div class="k">frames</div>`));
    row.append(el("div", "kpi", `<div class="v">${r.n_joints ?? "—"}</div><div class="k">joints</div>`));
    row.append(el("div", "kpi wide", `<div class="v mono">${esc(r.session_id)}</div><div class="k">session — accuracy is measured, not asserted</div>`));
    box.append(row);
  }
}

function renderDelivery(j, box) {
  box.innerHTML = "";
  if (j.phase !== "delivered") {
    box.append(el("div", "dim", "packaging LeRobot episodes…"));
    return;
  }
  /* funnel chips */
  const funnel = el("div", "chip-row");
  for (const [k, v] of Object.entries(j.funnel || {})) {
    funnel.append(el("span", "chip", `${esc(k)}: <b>${v}</b>`));
  }
  box.append(funnel);

  /* all rejected: say so loudly, with the reasons, instead of an empty stage */
  if (!j.episodes_detail.length && !j.episodes.length) {
    const panel = el("div", "reject-panel");
    panel.append(el("div", "reject-title",
      "🛡 0 episodes delivered — the feasibility gate rejected every video, so none of it ships as training data."));
    for (const v of j.variants.filter((x) => x.verdict && x.verdict !== "accept")) {
      panel.append(el("div", "reject-row",
        `<b>${esc(v.variant_id)}</b> · ${esc(v.verdict)} — ${esc((v.verdict_reasons || []).join("; ") || "see verification stage")}` +
        (v.vlm_notes ? `<div class="dim">“${esc(v.vlm_notes)}”</div>` : "")));
    }
    panel.append(el("div", "dim",
      "This is the gate working as designed: infeasible AI-generated motion never reaches the dataset. Try a simpler or more physically grounded prompt."));
    box.append(panel);
    if (j.delivery_path) {
      box.append(el("div", "path mono big-path",
        "rejection details: " + esc(j.delivery_path) + "\\rejected.json"));
    }
    return;
  }

  /* episode cards */
  const grid = el("div", "card-grid");
  (j.episodes_detail.length ? j.episodes_detail
    : j.episodes.map((e) => ({ episode_id: e }))).forEach((ep, i) => {
    const card = el("div", "mini-card deliver-card pop");
    const head = el("div", "verify-head");
    head.append(el("span", "mini-title", "📦 " + esc(ep.episode_id)));
    if (ep.tier) head.append(el("span", `tier-badge ${esc(ep.tier)}`, esc(ep.tier)));
    card.append(head);
    if (ep.caption) card.append(el("div", "mini-prompt", `“${esc(ep.caption)}”`));
    const chips = el("div", "chip-row tight");
    if (ep.format) chips.append(el("span", "chip sm", esc(ep.format)));
    for (const r of ep.robots || []) chips.append(el("span", "chip sm", "🤖 " + esc(r)));
    for (const s of ep.skills || []) chips.append(el("span", "chip sm skill", esc(s)));
    card.append(chips);
    if (ep.path) card.append(el("div", "path mono", esc(ep.path)));
    if (ep.has_trajectory) {
      const cw = el("div", "traj-wrap");
      const canvas = document.createElement("canvas");
      canvas.id = `traj_${i}`;
      cw.append(canvas);
      card.append(cw);
      drawTrajectory(ep.episode_id, canvas);
    }
    grid.append(card);
  });
  box.append(grid);

  if (j.delivery_path) {
    box.append(el("div", "path mono big-path",
      "delivery folder: " + esc(j.delivery_path)));
  }
  if (j.dataset_card) {
    const det = el("details", "card-details");
    det.innerHTML = `<summary>dataset card (README.md)</summary><pre>${esc(j.dataset_card)}</pre>`;
    box.append(det);
  }
}

async function drawTrajectory(episodeId, canvas) {
  if (trajDone.has(episodeId + canvas.id)) return;
  trajDone.add(episodeId + canvas.id);
  try {
    const d = await (await fetch(`/api/syngen/trajectory/${encodeURIComponent(episodeId)}`)).json();
    if (!d.hands || !d.hands.length) return;
    const colors = { px: "#22d3ee", py: "#a78bfa", pz: "#34d399" };
    const datasets = [];
    const hand = d.hands.find((h) => h.hand === "right") || d.hands[0];
    for (const axis of ["px", "py", "pz"]) {
      datasets.push({
        label: `${hand.hand} ${axis}`, data: hand[axis],
        borderColor: colors[axis], borderWidth: 1.6, pointRadius: 0, tension: 0.25,
      });
    }
    new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { labels: hand.t, datasets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { type: "linear", title: { display: true, text: "t (s)" }, grid: { color: "#141d31" } },
          y: { title: { display: true, text: `${d.robot} end-effector (m)` }, grid: { color: "#141d31" } },
        },
        plugins: { legend: { labels: { boxWidth: 10, font: { size: 9 } } } },
      },
    });
  } catch (e) { /* trajectory preview is best-effort */ }
}

/* ---------------------------------------------------------------- job head */
function renderJobHead(j) {
  const head = $("#jobHead");
  const running = j.runner && j.runner.running;
  const failed = j.runner && !j.runner.running && j.runner.returncode !== 0 && j.phase !== "delivered";
  const nothingShipped = j.phase === "delivered" &&
    !j.episodes.length && !j.episodes_detail.length;
  const phaseCls = nothingShipped ? "warn"
    : j.phase === "delivered" ? "ok" : failed ? "bad" : running ? "live" : "";
  const phaseTxt = failed ? "failed"
    : nothingShipped ? "completed — 0 episodes (all rejected)" : esc(j.phase);
  head.innerHTML = `
    <div class="job-line">
      <span class="job-id mono">${esc(j.job_id)}</span>
      <span class="phase-badge ${phaseCls}">${phaseTxt}${running ? " …" : ""}</span>
      <span class="chip sm">backend: ${esc(j.backend || "?")}</span>
      ${j.variants.length && running
        ? `<span class="chip sm">📼 ${j.variants.filter((v) => v.generated).length}/${j.variants.length} generated</span>` : ""}
      ${j.created_at ? `<span class="chip sm">${esc(String(j.created_at).replace("T", " ").slice(0, 19))}</span>` : ""}
    </div>
    <div class="job-prompt">“${esc(j.prompt)}”</div>
    ${running && j.runner.log_tail && j.runner.log_tail.length
      ? `<div class="job-log mono">${esc(j.runner.log_tail[j.runner.log_tail.length - 1])}</div>` : ""}
    ${failed ? `<div class="job-log mono bad">exit code ${j.runner.returncode} — ${esc((j.runner.log_tail || []).slice(-1)[0] || "see demo/.cache logs")}</div>` : ""}`;
}

/* ------------------------------------------------------------- render loop */
function renderJob(j, { replay = false } = {}) {
  $("#emptyHint").hidden = true;
  $("#flowRoot").hidden = false;
  if ($("#stages").dataset.job !== j.job_id) {
    buildSkeleton(j);
    $("#stages").dataset.job = j.job_id;
  }
  renderJobHead(j);

  let replayDelay = 0;
  for (const st of STAGES) {
    const sec = $("#stage_" + st.id);
    if (!sec) continue;
    sec.hidden = !!(st.skip && st.skip(j));
    if (sec.hidden) continue;
    const status = st.status(j);
    sec.classList.toggle("pending", status === "pending");
    sec.classList.toggle("active", status === "active");
    sec.classList.toggle("done", status === "done");
    sec.classList.toggle("skipped", status === "skipped");
    sec.querySelector(".stage-state").textContent =
      status === "done" ? "✓" : status === "active" ? "working…"
        : status === "skipped" ? "⊘ skipped" : "";

    if (status !== "pending") {
      const sig = st.sig(j);
      if (stageSigs[st.id] !== sig) {
        stageSigs[st.id] = sig;
        const box = sec.querySelector(".stage-content");
        if (replay) {
          setTimeout(() => { st.render(j, box); sec.classList.add("lit"); },
            replayDelay += 240);
        } else {
          st.render(j, box);
          sec.classList.add("lit");
        }
      }
    }
  }
}

async function fetchJobs() {
  try {
    const [syngenRes, demoRes] = await Promise.all([
      fetch("/api/syngen"),
      fetch("/api/label-demo/jobs"),
    ]);
    const syngen = syngenRes.ok ? (await syngenRes.json()).jobs || [] : [];
    const demo = demoRes.ok ? (await demoRes.json()).jobs || [] : [];
    return [...demo, ...syngen];
  } catch (e) { return []; }
}

async function refresh() {
  const jobs = await fetchJobs();
  populateHistory(jobs);
  if (!currentJobId) return;
  const j = jobs.find((x) => x.job_id === currentJobId);
  if (!j) return;
  renderJob(j);
  const running = j.runner && j.runner.running;
  const settled = j.phase === "delivered" || (j.runner && !j.runner.running);
  if (!running && settled && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function startPolling() {
  if (!pollTimer) pollTimer = setInterval(refresh, 3000);
}

/* ---------------------------------------------------------------- history */
function populateHistory(jobs) {
  const sel = $("#jobHistory");
  const prev = sel.value;
  const sorted = [...jobs].sort((a, b) =>
    String(b.created_at).localeCompare(String(a.created_at)));
  sel.innerHTML = '<option value="">— previous jobs —</option>';
  for (const j of sorted) {
    const opt = document.createElement("option");
    opt.value = j.job_id;
    const p = j.prompt.length > 42 ? j.prompt.slice(0, 42) + "…" : j.prompt;
    opt.textContent = `${j.job_id} — ${p || "(no prompt)"} [${j.phase}]`;
    sel.append(opt);
  }
  sel.value = currentJobId && sorted.some((j) => j.job_id === currentJobId)
    ? currentJobId : prev && sorted.some((j) => j.job_id === prev) ? prev : "";
}

$("#jobHistory").addEventListener("change", async (ev) => {
  const id = ev.target.value;
  if (!id) return;
  currentJobId = id;
  const jobs = await fetchJobs();
  const j = jobs.find((x) => x.job_id === id);
  if (!j) return;
  $("#stages").dataset.job = "";          // force fresh skeleton
  renderJob(j, { replay: j.phase === "delivered" });
  if (j.runner && j.runner.running) startPolling();
  window.scrollTo({ top: $("#flowRoot").offsetTop - 90, behavior: "smooth" });
});

/* ------------------------------------------------------------------ submit */
$("#sgForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = $("#sgRunBtn"), status = $("#sgStatus");
  const body = {
    prompt: $("#sgPrompt").value.trim(),
    variants: parseInt($("#sgVariants").value, 10) || 1,
    cameras: parseInt($("#sgCameras").value, 10) || 2,
    backend: $("#sgBackend").value,
  };
  if (!body.prompt) return;
  btn.disabled = true; btn.textContent = "launching…";
  status.hidden = false;
  status.textContent = "starting the factory…";
  try {
    const res = await fetch("/api/syngen/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const out = await res.json();
    if (!res.ok) throw new Error(out.detail || res.statusText);
    status.textContent = `job ${out.job_id} running — ${out.plan.variants} event(s) × ${out.plan.cameras} camera(s), backend=${out.plan.backend}`;
    currentJobId = out.job_id;
    $("#stages").dataset.job = "";
    await refresh();
    startPolling();
    setTimeout(() => window.scrollTo({ top: $("#flowRoot").offsetTop - 90, behavior: "smooth" }), 150);
  } catch (e) {
    status.textContent = "failed to start: " + e.message;
  } finally {
    btn.disabled = false; btn.textContent = "▶ Generate";
  }
});

$("#labelDemoBtn").addEventListener("click", async () => {
  const btn = $("#labelDemoBtn"), status = $("#sgStatus");
  $("#sgPrompt").value =
    "A dog runs forward across the frame with a clear quadruped gait cycle; label and package the local demo video.";
  btn.disabled = true;
  btn.textContent = "launching…";
  status.hidden = false;
  status.textContent = "starting parallel dog label demo…";
  try {
    const res = await fetch("/api/label-demo/run", { method: "POST" });
    const out = await res.json();
    if (!res.ok) throw new Error(out.detail || res.statusText);
    status.textContent =
      `job ${out.job_id} running — local ai_dog.mp4, generation skipped`;
    currentJobId = out.job_id;
    $("#stages").dataset.job = "";
    await refresh();
    startPolling();
    setTimeout(() => window.scrollTo({ top: $("#flowRoot").offsetTop - 90, behavior: "smooth" }), 150);
  } catch (e) {
    status.textContent = "failed to start demo: " + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "▶ Dog label demo";
  }
});

/* -------------------------------------------------------------------- boot */
(async function boot() {
  const jobs = await fetchJobs();
  populateHistory(jobs);
  /* if the factory is already running a job, attach to it */
  const running = jobs.find((j) => j.runner && j.runner.running);
  if (running) {
    currentJobId = running.job_id;
    renderJob(running);
    startPolling();
  }
})();
