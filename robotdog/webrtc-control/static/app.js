/* Go2 Control — frontend logic */
"use strict";

// ------------------------------------------------------------------ state
let ws = null;
let CONST = null;           // constants dump from server
let reqId = 0;
const pending = new Map();  // id -> {resolve, reject}
const topics = {};          // topic -> last data
const keys = new Set();     // held keyboard keys

const $ = (id) => document.getElementById(id);
const logEl = $("log");

function log(msg, isErr = false) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  logEl.textContent = (line + logEl.textContent).slice(0, 8000);
  if (isErr) console.warn(msg);
}

// ------------------------------------------------------------- websocket
function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => log("websocket connected");
  ws.onclose = () => { log("websocket closed, retrying...", true); setTimeout(connectWS, 2000); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") { topics[msg.topic] = msg.data; renderTopic(msg.topic, msg.data); }
    else if (msg.type === "status") renderStatus(msg);
    else if (msg.type === "lidar") renderLidar(msg.positions, msg.meta);
    else if (msg.type === "response") {
      const p = pending.get(msg.id);
      if (p) { pending.delete(msg.id); msg.ok ? p.resolve(msg.result) : p.reject(new Error(msg.error)); }
    }
  };
}

function send(msg) {
  return new Promise((resolve, reject) => {
    if (!ws || ws.readyState !== 1) return reject(new Error("ws not connected"));
    msg.id = ++reqId;
    pending.set(msg.id, { resolve, reject });
    ws.send(JSON.stringify(msg));
    setTimeout(() => { if (pending.delete(msg.id)) reject(new Error("timeout")); }, 12000);
  });
}

async function cmd(msg, label) {
  try {
    const r = await send(msg);
    log(`✅ ${label || msg.action}`);
    return r;
  } catch (e) {
    log(`❌ ${label || msg.action}: ${e.message}`, true);
    throw e;
  }
}

const sport = (c, parameter) => cmd({ action: "sport", cmd: c, parameter }, c);

// ------------------------------------------------------------- status bar
function renderStatus(s) {
  const badge = $("conn-badge");
  badge.textContent = s.connected ? "connected" : "disconnected";
  badge.className = "badge " + (s.connected ? "on" : "off");
  $("robot-ip").textContent = s.robot_ip;
  $("motion-mode").textContent = s.motion_mode;
  if (!pendingAvoid) $("avoid-toggle").checked = s.avoid_enabled;
}

// ----------------------------------------------------------- command grid
// acrobatic commands — disabled for safety (excluded from UI + dropdown)
const ACRO_BLOCKLIST = new Set(["FrontFlip", "BackFlip", "LeftFlip", "RightFlip",
  "FrontJump", "FrontPounce", "Handstand", "HandStand", "BackStand", "StandOut"]);
const GROUPS = {
  "grp-posture": ["StandUp", "StandDown", "Sit", "RiseSit", "BalanceStand",
    "RecoveryStand", "Damp", "StopMove", "Pose"],
  "grp-tricks": ["Hello", "Stretch", "Dance1", "Dance2", "WiggleHips",
    "FingerHeart", "Heart", "Scrape", "Content", "Wallow"],
  "grp-gaits": ["StaticWalk", "TrotRun", "EconomicGait", "ClassicWalk", "FreeWalk",
    "FreeBound", "FreeJump", "FreeAvoid", "ContinuousGait", "CrossStep",
    "CrossWalk", "MoonWalk", "OnesidedStep", "Bound", "LeadFollow"],
};
// commands that expect {"data": true}
const BOOL_PARAM = new Set(["StandOut", "Handstand", "HandStand", "BackStand",
  "FreeWalk", "FreeBound", "FreeJump", "FreeAvoid", "ClassicWalk", "CrossStep",
  "CrossWalk", "MoonWalk", "OnesidedStep", "Bound", "WiggleHips", "LeadFollow",
  "FrontFlip", "BackFlip", "LeftFlip", "RightFlip"]);

function buildCommandButtons() {
  const known = new Set(Object.keys(CONST.SPORT_CMD).concat(Object.keys(CONST.SPORT_CMD_MCF)));
  ACRO_BLOCKLIST.forEach((n) => known.delete(n));
  for (const [gridId, names] of Object.entries(GROUPS)) {
    const grid = $(gridId);
    for (const name of names) {
      if (!known.has(name)) continue;
      const btn = document.createElement("button");
      btn.textContent = name;
      btn.onclick = () => sport(name, BOOL_PARAM.has(name) ? { data: true } : undefined);
      grid.appendChild(btn);
    }
  }
  // "any command" dropdown = every SPORT_CMD + SPORT_CMD_MCF entry
  const sel = $("any-cmd");
  for (const name of [...known].sort()) {
    const o = document.createElement("option");
    o.value = o.textContent = name;
    sel.appendChild(o);
  }
  // generic RPC topic dropdown = every RTC_TOPIC
  const tsel = $("rpc-topic");
  for (const [key, topic] of Object.entries(CONST.RTC_TOPIC)) {
    const o = document.createElement("option");
    o.value = topic; o.textContent = `${key} (${topic})`;
    tsel.appendChild(o);
  }
  // raw topic viewer
  const rsel = $("raw-topic-select");
  rsel.onchange = () => renderRaw();
}

$("any-send").onclick = () => {
  let p;
  const t = $("any-param").value.trim();
  if (t) { try { p = JSON.parse(t); } catch { return log("❌ invalid parameter JSON", true); } }
  sport($("any-cmd").value, p);
};

$("rpc-send").onclick = async () => {
  let p;
  const t = $("rpc-param").value.trim();
  if (t) { try { p = JSON.parse(t); } catch { return log("❌ invalid parameter JSON", true); } }
  try {
    const r = await cmd({
      action: "request", topic: $("rpc-topic").value,
      api_id: parseInt($("rpc-api").value, 10), parameter: p,
    }, "RPC");
    $("rpc-result").textContent = JSON.stringify(r, null, 2);
  } catch (e) { $("rpc-result").textContent = String(e); }
};

// ------------------------------------------------------------------ drive
function speeds() {
  const boost = keys.has("shift") ? 1.6 : 1;
  return { x: parseFloat($("speed-x").value) * boost, yaw: parseFloat($("speed-yaw").value) * boost };
}

const MOVES = {
  fwd: () => ({ x: speeds().x, y: 0, z: 0 }),
  back: () => ({ x: -speeds().x, y: 0, z: 0 }),
  left: () => ({ x: 0, y: speeds().x, z: 0 }),
  right: () => ({ x: 0, y: -speeds().x, z: 0 }),
  yawl: () => ({ x: 0, y: 0, z: speeds().yaw }),
  yawr: () => ({ x: 0, y: 0, z: -speeds().yaw }),
};

document.querySelectorAll("[data-move]").forEach((btn) => {
  const kind = btn.dataset.move;
  if (kind === "stop") { btn.onclick = () => cmd({ action: "stop" }, "StopMove"); return; }
  let timer = null;
  const start = (e) => {
    e.preventDefault();
    const fire = () => send({ action: "move", ...MOVES[kind]() }).catch(() => {});
    fire(); timer = setInterval(fire, 250);
  };
  const stop = () => { if (timer) { clearInterval(timer); timer = null; } };
  btn.addEventListener("mousedown", start);
  btn.addEventListener("touchstart", start);
  ["mouseup", "mouseleave", "touchend"].forEach((ev) => btn.addEventListener(ev, stop));
});

// keyboard drive
const KEYMAP = { w: "fwd", s: "back", a: "left", d: "right", q: "yawl", e: "yawr" };
let keyTimer = null;
function keyLoop() {
  let x = 0, y = 0, z = 0;
  const sp = speeds();
  if (keys.has("w")) x += sp.x;
  if (keys.has("s")) x -= sp.x;
  if (keys.has("a")) y += sp.x;
  if (keys.has("d")) y -= sp.x;
  if (keys.has("q")) z += sp.yaw;
  if (keys.has("e")) z -= sp.yaw;
  if (x || y || z) send({ action: "move", x, y, z }).catch(() => {});
}
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); cmd({ action: "stop" }, "StopMove"); return; }
  if (k === "shift") { keys.add("shift"); return; }
  if (!(k in KEYMAP)) return;
  e.preventDefault();
  keys.add(k);
  if (!keyTimer) { keyLoop(); keyTimer = setInterval(keyLoop, 250); }
});
window.addEventListener("keyup", (e) => {
  keys.delete(e.key.toLowerCase());
  if (![...keys].some((k) => k in KEYMAP) && keyTimer) { clearInterval(keyTimer); keyTimer = null; }
});

// sliders / params
function bindSlider(id, valId, fn) {
  $(id).addEventListener("input", () => { $(valId).textContent = $(id).value; });
  $(id).addEventListener("change", () => fn(parseFloat($(id).value)));
}
bindSlider("speed-x", "speed-x-val", () => {});
bindSlider("speed-yaw", "speed-yaw-val", () => {});
bindSlider("body-height", "body-height-val", (v) => sport("BodyHeight", { data: v }));
bindSlider("foot-raise", "foot-raise-val", (v) => sport("FootRaiseHeight", { data: v }));
$("speed-level").onchange = () => sport("SpeedLevel", { data: parseInt($("speed-level").value, 10) });

let pendingAvoid = false;
$("avoid-toggle").onchange = async () => {
  pendingAvoid = true;
  try {
    const r = await cmd({ action: "avoid_set", enable: $("avoid-toggle").checked }, "obstacle avoidance");
    if (r && typeof r.enable === "boolean") $("avoid-toggle").checked = r.enable;
  } finally {
    pendingAvoid = false;
  }
};
$("mode-select").onchange = () =>
  cmd({ action: "set_mode", name: $("mode-select").value }, `mode → ${$("mode-select").value}`);
$("video-toggle").onchange = () => {
  cmd({ action: "video_channel", enable: $("video-toggle").checked }, "video channel");
  $("video").src = $("video-toggle").checked ? "/video?" + Date.now() : "";
};

// --------------------------------------------------------------------- vui
document.querySelectorAll("#led-colors button").forEach((b) => {
  b.onclick = () => cmd({ action: "vui_color", color: b.dataset.color, time: 10 }, `LED ${b.dataset.color}`);
});
bindSlider("vui-brightness", "vui-brightness-val", (v) => cmd({ action: "vui_brightness", level: v }, "brightness"));
bindSlider("vui-volume", "vui-volume-val", (v) => cmd({ action: "vui_volume", level: v }, "volume"));

// --------------------------------------------------------------- audio hub
$("audio-refresh").onclick = async () => {
  try {
    const r = await cmd({ action: "audio_list" }, "audio list");
    const list = r.audio_list || r.list || [];
    const ul = $("audio-list");
    ul.innerHTML = "";
    for (const item of list) {
      const li = document.createElement("li");
      const name = item.CUSTOM_NAME || item.name || item.unique_id;
      li.innerHTML = `<span>${name}</span>`;
      const play = document.createElement("button");
      play.textContent = "▶";
      play.onclick = () => cmd({ action: "audio_play", uuid: item.UNIQUE_ID || item.unique_id }, `play ${name}`);
      li.appendChild(play);
      ul.appendChild(li);
    }
    if (!list.length) ul.innerHTML = "<li>(no audio files)</li>";
  } catch {}
};
$("audio-pause").onclick = () => cmd({ action: "audio_pause" });
$("audio-resume").onclick = () => cmd({ action: "audio_resume" });
$("audio-play-mode").onchange = () => cmd({ action: "audio_play_mode", mode: $("audio-play-mode").value });
$("megaphone-on").onclick = () => cmd({ action: "megaphone", enable: true });
$("megaphone-off").onclick = () => cmd({ action: "megaphone", enable: false });

// ------------------------------------------------------------------- lidar
$("lidar-toggle").onchange = () => cmd({ action: "lidar", enable: $("lidar-toggle").checked }, "lidar");
function renderLidar(pos, meta) {
  const sport = topics[CONST.RTC_TOPIC.LF_SPORT_MOD_STATE] || {};
  const yaw = sport.imu_state && sport.imu_state.rpy ? sport.imu_state.rpy[2] : undefined;
  if (window.lidar3d) window.lidar3d.update(pos, meta, sport.position, yaw);
  $("lidar-count").textContent = `${(pos.length / 3) | 0} pts`;
}

// ------------------------------------------------------------- R3-1 remote
const R3_KEYS = ["R1", "L1", "START", "SELECT", "R2", "L2", "F1", "F2",
  "A", "B", "X", "Y", "↑", "→", "↓", "←"];
(function initR3Keys() {
  $("r3-keys").innerHTML = R3_KEYS.map((k) => `<span data-k="${k}">${k}</span>`).join("");
})();
$("r3-enable").onclick = () => sport("SwitchJoystick", { data: true });
$("r3-disable").onclick = () => sport("SwitchJoystick", { data: false });

function drawStick(id, x, y) {
  const cv = $(id), ctx = cv.getContext("2d");
  const c = cv.width / 2;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.strokeStyle = "#30363d";
  ctx.beginPath(); ctx.arc(c, c, c - 2, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(c, 4); ctx.lineTo(c, cv.height - 4);
  ctx.moveTo(4, c); ctx.lineTo(cv.width - 4, c); ctx.stroke();
  ctx.fillStyle = "#58a6ff";
  ctx.beginPath();
  ctx.arc(c + x * (c - 8), c - y * (c - 8), 6, 0, Math.PI * 2);
  ctx.fill();
}
drawStick("r3-left", 0, 0);
drawStick("r3-right", 0, 0);

function renderR3(d) {
  if (!d) return;
  $("r3-status").textContent =
    `lx ${fmt(d.lx)} ly ${fmt(d.ly)} | rx ${fmt(d.rx)} ry ${fmt(d.ry)}`;
  drawStick("r3-left", d.lx || 0, d.ly || 0);
  drawStick("r3-right", d.rx || 0, d.ry || 0);
  const bits = d.keys || 0;
  document.querySelectorAll("#r3-keys span").forEach((el, i) => {
    el.classList.toggle("pressed", !!(bits & (1 << i)));
  });
}

// --------------------------------------------------------------- telemetry
const fmt = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : v ?? "—");
function kv(el, obj) {
  el.innerHTML = Object.entries(obj).map(([k, v]) => `<div>${k}<b>${v}</b></div>`).join("");
}

function renderTopic(topic, data) {
  const T = CONST ? CONST.RTC_TOPIC : {};
  if (topic === T.LOW_STATE) renderLowState(data);
  else if (topic === T.LF_SPORT_MOD_STATE || topic === T.SPORT_MOD_STATE) renderSportState(data);
  else if (topic === T.WIRELESS_CONTROLLER) renderR3(data);
  // raw viewer
  const sel = $("raw-topic-select");
  if (![...sel.options].some((o) => o.value === topic)) {
    const o = document.createElement("option");
    o.value = o.textContent = topic;
    sel.appendChild(o);
  }
  if (sel.value === topic) renderRaw();
}
function renderRaw() {
  const t = $("raw-topic-select").value;
  $("raw-json").textContent = JSON.stringify(topics[t], null, 2)?.slice(0, 6000) || "—";
}

function renderLowState(d) {
  if (!d) return;
  const bms = d.bms_state || {};
  $("battery-soc").textContent = bms.soc != null ? bms.soc + "%" : "—";
  kv($("bms-kv"), {
    "SOC %": bms.soc, "current mA": bms.current, "cycles": bms.cycle,
    "BQ NTC °C": (bms.bq_ntc || []).join(" / "), "MCU NTC °C": (bms.mcu_ntc || []).join(" / "),
    "power V": fmt(d.power_v), "board NTC1 °C": d.temperature_ntc1,
    "BMS ver": bms.version_high != null ? `${bms.version_high}.${bms.version_low}` : "—",
  });
  // motors
  const tbody = $("motor-table").querySelector("tbody");
  const ms = d.motor_state || [];
  let html = "";
  for (let i = 0; i < 6; i++) {
    const a = ms[i] || {}, b = ms[i + 6] || {};
    const hot = (t) => (t >= 70 ? ' class="hot"' : "");
    html += `<tr><td>${i}</td><td${hot(a.temperature)}>${a.temperature ?? "—"}</td><td>${fmt(a.q)}</td>` +
      `<td>${i + 6}</td><td${hot(b.temperature)}>${b.temperature ?? "—"}</td><td>${fmt(b.q)}</td></tr>`;
  }
  tbody.innerHTML = html;
  kv($("foot-kv"), { "foot force": (d.foot_force || []).join(" / ") });
}

function renderSportState(d) {
  if (!d) return;
  const imu = d.imu_state || {};
  const rpy = imu.rpy || [];
  kv($("imu-kv"), {
    "roll °": fmt(rpy[0] * 57.3, 1), "pitch °": fmt(rpy[1] * 57.3, 1), "yaw °": fmt(rpy[2] * 57.3, 1),
    "IMU °C": imu.temperature ?? "—",
  });
  kv($("sport-kv"), {
    "mode": d.mode, "gait": d.gait_type, "progress": d.progress,
    "body h m": fmt(d.body_height), "foot raise m": fmt(d.foot_raise_height),
    "pos x/y/z": (d.position || []).map((v) => fmt(v, 1)).join(" / "),
    "vel x/y/z": (d.velocity || []).map((v) => fmt(v, 1)).join(" / "),
    "yaw speed": fmt(d.yaw_speed), "error": d.error_code,
    "obstacles m": (d.range_obstacle || []).map((v) => fmt(v, 1)).join(" / "),
  });
}

// ------------------------------------------------------------------- init
(async function init() {
  CONST = await (await fetch("/api/constants")).json();
  buildCommandButtons();
  connectWS();
  log("UI ready — waiting for robot connection");
})();
