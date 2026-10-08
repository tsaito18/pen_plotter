// 筆跡 — 集める（iPad＋Apple Pencil）・見直す・学習。スタジオと同じサーバー・同じ部品。

import { $, STORAGE, api, bindTheme, el, formatDuration, icon, toast, toastUndo } from "./common.js";
import { StrokeOrder, drawStrokes, unflatten } from "./glyph.js";
import { LOGICAL, WritingPad } from "./pad.js";

const state = {
  profile: null,
  profiles: [],
  mode: "write",
  current: null, // /api/collect/next の結果
  forced: null, // 見直しから「この字を書く」
  queue: [], // スタジオからの依頼
  queueActive: false,
  queueIndex: 0,
  session: 0,
  glyphs: new Map(),
  stats: null,
  issues: null,
  filter: "all",
  drawerChar: null,
  training: null,
  models: { models: [], active: null },
  lossHistory: [],
};

let pad;
let guide;

// ------------------------------------------------------------------ 起動

async function boot() {
  pad = new WritingPad($("pad"));
  guide = new StrokeOrder($("guideCanvas"));
  bindTheme($("themeBtn"), () => guide.draw(1));
  bindModes();
  bindProfileMenu();
  bindWrite();
  bindReview();
  bindTrain();
  bindKeyboard();

  const params = new URLSearchParams(location.search);
  await loadProfiles(params.get("profile"));
  if (params.get("char")) state.forced = params.get("char").slice(0, 1);
  if (params.get("queue")) state.queueActive = true;
  const mode = location.hash.slice(1);
  setMode(["write", "review", "train"].includes(mode) ? mode : "write", { push: false });
  window.setInterval(pollQueue, 6000);
  window.setInterval(() => state.mode === "train" && loadTraining(), 1500);
}

// ------------------------------------------------------------------ 書き手

async function loadProfiles(wanted) {
  const boot = await api("/api/bootstrap");
  state.profiles = boot.profiles;
  const saved = wanted || STORAGE.get("profile");
  const ids = state.profiles.map((p) => p.id);
  state.profile = ids.includes(saved) ? saved : ids[0] || null;
  renderProfiles();
}

function renderProfiles() {
  const id = state.profile;
  $("profileName").textContent = id || "未設定";
  $("profileAvatar").textContent = id ? id[0].toUpperCase() : "+";
  const list = $("profileList");
  list.replaceChildren(
    ...state.profiles.map((p) =>
      el(
        "button",
        {
          class: "profile-item",
          type: "button",
          role: "menuitemradio",
          "aria-checked": String(p.id === id),
          onclick: () => selectProfile(p.id),
        },
        [el("span", { class: "avatar", text: p.id[0].toUpperCase() }), el("b", { text: p.id }), el("span", { text: `${p.characters}字` })],
      ),
    ),
  );
  if (!state.profiles.length) list.append(el("p", { class: "menu-head", text: "まだ誰もいません。下で追加してください" }));
}

async function selectProfile(id) {
  state.profile = id;
  STORAGE.set("profile", id);
  state.forced = null;
  state.queueActive = false;
  renderProfiles();
  $("profileMenu").classList.remove("is-open");
  toast(`書き手を「${id}」にしました`, "ok", 1800);
  await refreshMode();
}

function bindProfileMenu() {
  const menu = $("profileMenu");
  const toggle = menu.querySelector(".profile-pill");
  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = menu.classList.toggle("is-open");
    toggle.setAttribute("aria-expanded", String(open));
    if (open) $("profileInput").focus({ preventScroll: true });
  });
  menu.addEventListener("click", (e) => e.stopPropagation());
  document.addEventListener("click", () => {
    menu.classList.remove("is-open");
    toggle.setAttribute("aria-expanded", "false");
  });
  $("welcome").addEventListener("submit", (e) => {
    e.preventDefault();
    createProfile($("welcomeInput").value.trim());
  });
  $("profileForm").addEventListener("submit", (e) => {
    e.preventDefault();
    createProfile($("profileInput").value.trim());
  });
}

async function createProfile(id) {
  if (!id) return;
  try {
    await api("/api/profiles", { method: "POST", body: { id } });
    $("profileInput").value = "";
    await loadProfiles(id);
    await selectProfile(id);
  } catch (error) {
    toast(`追加できません: ${error.message}`, "error");
  }
}

function needProfile() {
  if (state.profile) return true;
  toast("まず書き手を追加してください", "warn");
  setMode("write");
  $("welcomeInput").focus();
  return false;
}

// ------------------------------------------------------------------ モード

function bindModes() {
  document.querySelectorAll(".modes .seg-btn").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  window.addEventListener("hashchange", () => {
    const m = location.hash.slice(1);
    if (["write", "review", "train"].includes(m) && m !== state.mode) setMode(m, { push: false });
  });
}

function setMode(mode, { push = true } = {}) {
  state.mode = mode;
  $("app").dataset.mode = mode;
  document.querySelector(".modes").dataset.active = mode;
  document.querySelectorAll(".modes .seg-btn").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  $("viewWrite").hidden = mode !== "write";
  $("viewReview").hidden = mode !== "review";
  $("viewTrain").hidden = mode !== "train";
  if (push) history.replaceState(null, "", `${location.pathname}${location.search}#${mode}`);
  refreshMode();
}

async function refreshMode() {
  if (state.mode === "write") {
    await pollQueue();
    await loadNext();
  } else if (state.mode === "review") {
    await loadReview();
  } else {
    await Promise.all([loadTraining(), loadModels()]);
  }
}

// ------------------------------------------------------------------ 集める

function bindWrite() {
  pad.addEventListener("change", updatePadState);
  pad.addEventListener("strokestart", () => $("pad").classList.add("has-ink"));
  pad.addEventListener("gestureundo", () => toast("1 画戻しました", "info", 1200));
  $("saveBtn").addEventListener("click", save);
  $("undoBtn").addEventListener("click", () => pad.undo());
  $("clearBtn").addEventListener("click", () => pad.clear());
  $("skipBtn").addEventListener("click", skip);
  $("guideBtn").addEventListener("click", () => guide.play());
  $("ghostToggle").checked = Boolean(STORAGE.get("ghost", false));
  $("ghostToggle").addEventListener("change", (e) => {
    STORAGE.set("ghost", e.target.checked);
    applyGhost();
  });
  $("touchToggle").checked = Boolean(STORAGE.get("touch", false));
  pad.allowTouch = $("touchToggle").checked;
  $("touchToggle").addEventListener("change", (e) => {
    pad.allowTouch = e.target.checked;
    STORAGE.set("touch", e.target.checked);
  });
  $("requestStart").addEventListener("click", () => {
    state.queueActive = true;
    state.queueIndex = 0;
    state.forced = null;
    renderRequest();
    loadNext();
  });
  $("requestStop").addEventListener("click", () => {
    state.queueActive = false;
    renderRequest();
    loadNext();
  });
  $("requestClear").addEventListener("click", async () => {
    await api("/api/collect/queue", { method: "POST", body: { profile: state.profile, chars: "" } });
    state.queue = [];
    state.queueActive = false;
    renderRequest();
    loadNext();
  });
}

async function glyphOf(char) {
  if (!state.glyphs.has(char)) {
    const data = await api("/api/glyph", { params: { char } }).catch(() => ({ strokes: [] }));
    state.glyphs.set(char, data.strokes || []);
  }
  return state.glyphs.get(char);
}

async function loadNext() {
  $("welcome").hidden = Boolean(state.profile);
  if (!state.profile) {
    renderSubject(null);
    return;
  }
  let info;
  let next = null;
  const queue = state.queueActive ? state.queue.slice(state.queueIndex) : [];
  if (state.forced) {
    info = await api("/api/collect/next", { params: { profile: state.profile, char: state.forced } });
    info.mode = "forced";
  } else if (queue.length) {
    info = await api("/api/collect/next", { params: { profile: state.profile, char: queue[0] } });
    info.mode = "queue";
    next = queue[1] || null;
  } else {
    if (state.queueActive) {
      state.queueActive = false;
      renderRequest();
      toast("依頼の字をすべて書きました", "ok", 3000);
    }
    info = await api("/api/collect/next", {
      params: { profile: state.profile, prefer: state.current?.next || undefined },
    });
    next = info.next;
    updateTally(info);
  }
  info.next = next;
  state.current = info;
  renderSubject(info);
}

async function renderSubject(info) {
  const charEl = $("subjectChar");
  if (!info || !info.char) {
    charEl.textContent = info ? "完" : "？";
    $("subjectTier").textContent = info ? "全部そろいました" : "書き手を選んでください";
    $("subjectDots").replaceChildren();
    $("subjectInfo").textContent = "";
    $("nextChar").textContent = "—";
    guide.set([]);
    $("saveBtn").disabled = true;
    return;
  }
  charEl.textContent = info.char;
  charEl.classList.remove("swap");
  void charEl.offsetWidth;
  charEl.classList.add("swap");
  const tier = $("subjectTier");
  if (info.mode === "queue") {
    tier.textContent = `依頼 ${state.queueIndex + 1} / ${state.queue.length}`;
    tier.className = "chip queue";
    tier.removeAttribute("data-tier");
  } else if (info.mode === "forced") {
    tier.textContent = "書き直し";
    tier.className = "chip queue";
    tier.removeAttribute("data-tier");
  } else {
    tier.textContent = info.tier_label;
    tier.className = "chip";
    tier.dataset.tier = info.tier;
  }
  renderDots(info.count, info.target);
  $("nextChar").textContent = info.next || "—";
  const strokes = await glyphOf(info.char);
  if (state.current !== info) return;
  $("guideBtn").classList.toggle("empty", !strokes.length);
  guide.set(strokes);
  $("subjectInfo").textContent = strokes.length ? `お手本 ${strokes.length} 画` : "お手本なし";
  applyGhost();
  updatePadState();
}

function renderDots(count, target) {
  const dots = $("subjectDots");
  dots.replaceChildren(...Array.from({ length: Math.max(target, count) }, (_, i) => el("i", { class: i < count ? "on" : "" })));
  dots.setAttribute("aria-label", `${count} / ${target} サンプル`);
}

function applyGhost() {
  const char = state.current?.char;
  pad.setGhost($("ghostToggle").checked && char ? state.glyphs.get(char) : null);
}

function updatePadState() {
  const n = pad.count;
  $("pad").classList.toggle("has-ink", n > 0);
  $("saveBtn").disabled = n === 0 || !state.current?.char;
  $("undoBtn").disabled = n === 0;
  $("clearBtn").disabled = n === 0;
  const expected = state.current?.char ? (state.glyphs.get(state.current.char) || []).length : 0;
  const chip = $("strokeChip");
  if (n === 0) {
    chip.dataset.state = "idle";
    $("strokeText").textContent = expected ? `お手本は ${expected} 画` : "0 画";
  } else if (!expected) {
    chip.dataset.state = "idle";
    $("strokeText").textContent = `${n} 画`;
  } else if (n === expected) {
    chip.dataset.state = "match";
    $("strokeText").textContent = `${n} 画 — お手本と同じ`;
  } else {
    chip.dataset.state = "differ";
    $("strokeText").textContent = `${n} 画（お手本は ${expected} 画）`;
  }
}

function updateTally(info) {
  $("tallyDone").textContent = info.completed;
  $("tallyTotal").textContent = info.total;
  $("tallyFill").style.width = `${(info.completed / Math.max(info.total, 1)) * 100}%`;
  $("tallySession").textContent = state.session;
}

let saving = false;

async function save() {
  if (saving || pad.empty || !state.current?.char || !needProfile()) return;
  saving = true;
  const info = state.current;
  const strokes = pad.data();
  try {
    const result = await api("/api/collect/samples", {
      method: "POST",
      body: { profile: state.profile, character: info.char, strokes },
    });
    flyAway();
    renderDots(result.count, info.target);
    const on = $("subjectDots").querySelectorAll("i.on");
    on[on.length - 1]?.classList.add("pop");
    state.session += 1;
    $("tallySession").textContent = state.session;
    toast(`「${info.char}」を保存しました`, "ok", 3200, {
      label: "取り消す",
      run: async () => {
        await api("/api/collect/undo", { method: "POST", body: { profile: state.profile } }).catch(() => null);
        state.session = Math.max(0, state.session - 1);
        toast(`「${info.char}」を取り消しました`, "info", 1600);
        loadNext();
      },
    });
    if (info.mode === "forced") state.forced = null;
    if (info.mode === "queue") await pollQueue();
    window.setTimeout(loadNext, 260);
  } catch (error) {
    toast(`保存できませんでした: ${error.message}`, "error", 5000);
  } finally {
    saving = false;
  }
}

/** 書いた字を小さくしてサンプル数の点へ送り込む。 */
function flyAway() {
  const frame = $("fly");
  const img = el("img", { src: pad.snapshot(), alt: "" });
  frame.append(img);
  pad.clear();
  $("pad").classList.add("flash");
  window.setTimeout(() => $("pad").classList.remove("flash"), 600);
  const from = frame.getBoundingClientRect();
  const dots = $("subjectDots").getBoundingClientRect();
  const dx = dots.left + dots.width / 2 - (from.left + from.width / 2);
  const dy = dots.top + dots.height / 2 - (from.top + from.height / 2);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const anim = img.animate(
    reduce
      ? [{ opacity: 1 }, { opacity: 0 }]
      : [
          { transform: "translate(0,0) scale(1)", opacity: 1 },
          { transform: `translate(${dx * 0.35}px, ${dy * 0.35 - 30}px) scale(0.45)`, opacity: 0.9, offset: 0.45 },
          { transform: `translate(${dx}px, ${dy}px) scale(0.04)`, opacity: 0 },
        ],
    { duration: reduce ? 200 : 620, easing: "cubic-bezier(.5,0,.2,1)" },
  );
  anim.onfinish = () => img.remove();
}

function skip() {
  if (state.forced) state.forced = null;
  else if (state.queueActive && state.queue.length) state.queueIndex = (state.queueIndex + 1) % state.queue.length;
  pad.clear();
  loadNext();
}

async function pollQueue() {
  if (!state.profile) return;
  try {
    const { chars } = await api("/api/collect/queue", { params: { profile: state.profile } });
    const changed = chars.join("") !== state.queue.join("");
    state.queue = chars;
    if (state.queueIndex >= chars.length) state.queueIndex = 0;
    if (changed || !chars.length) renderRequest();
  } catch {
    // 一時的な失敗は次の問い合わせで回復する
  }
}

function renderRequest() {
  const box = $("request");
  box.hidden = state.queue.length === 0;
  box.classList.toggle("is-active", state.queueActive);
  $("requestCount").textContent = state.queue.length;
  $("requestChars").textContent = state.queue.join("");
  $("requestLead").textContent = state.queueActive ? "字、スタジオからの依頼を書いています" : "字、スタジオから「書いて教えて」";
  $("requestStart").hidden = state.queueActive;
  $("requestStop").hidden = !state.queueActive;
}

// ------------------------------------------------------------------ 見直す

function bindReview() {
  $("filters").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-filter]");
    if (!b) return;
    state.filter = b.dataset.filter;
    $("filters").querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    renderReview();
  });
  $("searchInput").addEventListener("input", (e) => {
    const c = e.target.value.trim();
    if (c) openDrawer(c);
    renderReview();
  });
  $("drawerClose").addEventListener("click", closeDrawer);
  $("scrim").addEventListener("click", closeDrawer);
  $("drawerWrite").addEventListener("click", () => {
    state.forced = state.drawerChar;
    closeDrawer();
    setMode("write");
  });
  $("drawerDeleteAll").addEventListener("click", deleteAll);
}

async function loadReview() {
  if (!state.profile) return;
  const [stats, issues] = await Promise.all([
    api("/api/collect/stats", { params: { profile: state.profile } }),
    api("/api/collect/issues", { params: { profile: state.profile } }),
  ]);
  state.stats = stats;
  state.issues = issues;
  renderReview();
}

function issueChars() {
  const set = new Set();
  for (const a of state.issues?.anomalies || []) set.add(a.character);
  for (const m of state.issues?.mismatches || []) set.add(m.character);
  return set;
}

function issueTotal() {
  const i = state.issues || { anomalies: [], mismatches: [] };
  return i.anomalies.length + i.mismatches.length;
}

function renderReview() {
  const stats = state.stats;
  if (!stats) return;
  const tiers = stats.tiers;
  const done = Object.values(tiers).reduce((n, t) => n + t.completed, 0);
  const total = Object.values(tiers).reduce((n, t) => n + t.total, 0);
  $("figDone").textContent = done;
  $("figTotal").textContent = total;
  $("figSamples").textContent = stats.total_samples.toLocaleString();
  $("figIssues").textContent = issueTotal();
  $("figIssues").parentElement.classList.toggle("zero", issueTotal() === 0);
  const badge = $("issueCount");
  badge.hidden = issueTotal() === 0;
  badge.textContent = issueTotal();
  const names = ["レポート頻出", "基本", "標準"];
  $("tiers").replaceChildren(
    ...["tier1", "tier2", "tier3"].map((key, i) => {
      const t = tiers[key];
      const bar = el("span");
      bar.style.width = `${(t.completed / Math.max(t.total, 1)) * 100}%`;
      return el("div", { class: "tier", "data-tier": i }, [
        el("span", { text: names[i] }),
        el("div", { class: "tier-track" }, bar),
        el("b", { text: `${t.completed}/${t.total}` }),
      ]);
    }),
  );

  const showIssues = state.filter === "issues";
  $("cells").hidden = showIssues;
  $("issues").hidden = !showIssues;
  if (showIssues) {
    renderIssues();
    return;
  }
  const flagged = issueChars();
  const query = $("searchInput").value.trim();
  const target = stats.target;
  const entries = Object.entries(stats.char_counts).filter(([c, n]) => {
    if (query && c !== query) return false;
    if (state.filter === "todo") return n < target;
    if (state.filter === "done") return n >= target;
    return true;
  });
  const cells = entries.map(([c, n], i) => {
    const cell = el(
      "button",
      {
        class: `cell${flagged.has(c) ? " has-issue" : ""}`,
        type: "button",
        "data-level": Math.min(n, 3),
        title: `${c}: ${n} サンプル`,
        onclick: () => openDrawer(c),
      },
      [document.createTextNode(c), el("span", { class: "cell-bar" }, [el("i"), el("i"), el("i")])],
    );
    cell.style.setProperty("--i", Math.min(i, 200));
    return cell;
  });
  if (!cells.length) cells.push(el("div", { class: "cells-empty", text: query ? `「${query}」はまだありません` : "該当する字はありません" }));
  $("cells").replaceChildren(...cells);
}

function sampleThumb(sample, { onDelete, outlier = false } = {}) {
  const canvas = el("canvas");
  const secs = sampleSeconds(sample.strokes);
  const meta = `${sample.stroke_count}画${secs ? ` · ${secs.toFixed(1)}秒` : ""}`;
  const node = el("div", { class: `thumb${outlier ? " outlier" : ""}` }, [canvas, el("div", { class: "thumb-meta", text: meta })]);
  if (onDelete) {
    node.append(
      el("button", { class: "thumb-del", type: "button", "aria-label": "このサンプルを消す", html: icon("i-trash"), onclick: () => onDelete(node) }),
    );
  }
  // 古いサンプルは書き込み欄の大きさが違うので、外接矩形に合わせる（小さな字は小さいまま）
  requestAnimationFrame(() => drawStrokes(canvas, sample.strokes, { minSpan: LOGICAL * 0.55, margin: 0.14 }));
  return node;
}

function sampleSeconds(strokes) {
  const first = strokes[0]?.[0]?.timestamp;
  const lastStroke = strokes[strokes.length - 1];
  const last = lastStroke?.[lastStroke.length - 1]?.timestamp;
  return first !== undefined && last !== undefined ? Math.max(0, (last - first) / 1000) : 0;
}

async function trash(char, filename, node) {
  await api("/api/collect/samples", { method: "DELETE", params: { profile: state.profile, char, file: filename } });
  node?.classList.add("is-gone");
  toastUndo(`「${char}」のサンプルを消しました`, async () => {
    await api("/api/collect/samples/restore", { method: "POST", body: { profile: state.profile, char, files: [filename] } });
    refreshAfterEdit(char);
  });
  window.setTimeout(() => refreshAfterEdit(char), 260);
}

function refreshAfterEdit(char) {
  loadReview();
  if (state.drawerChar === char) openDrawer(char);
}

function renderIssues() {
  const box = $("issues");
  const { anomalies, mismatches } = state.issues;
  if (!anomalies.length && !mismatches.length) {
    box.replaceChildren(el("div", { class: "cells-empty", html: `${icon("i-check")}<br>確認が必要なサンプルはありません` }));
    return;
  }
  const parts = [];
  if (anomalies.length) {
    parts.push(el("h4", { class: "issues-title", text: `形があやしい（${anomalies.length}）` }));
    parts.push(
      el(
        "div",
        { class: "issue-grid" },
        anomalies.map((a) => {
          const card = el("div", { class: "issue" });
          card.append(
            el("div", { class: "issue-top" }, [
              el("span", { class: "issue-char", text: a.character }),
              el("div", { class: "reasons" }, a.reasons.map((r) => el("span", { class: "reason", text: r }))),
            ]),
            sampleThumb({ ...a, strokes: a.strokes }),
            el("div", { class: "issue-actions" }, [
              el("button", {
                class: "btn btn-quiet small",
                type: "button",
                html: `${icon("i-check")}問題ない`,
                onclick: async () => {
                  await api("/api/collect/samples/metadata", {
                    method: "POST",
                    body: { profile: state.profile, char: a.character, file: a.filename, key: "ignore_anomaly", value: true },
                  });
                  card.classList.add("is-gone");
                  window.setTimeout(loadReview, 260);
                },
              }),
              el("button", {
                class: "btn btn-danger small",
                type: "button",
                html: `${icon("i-trash")}消す`,
                onclick: () => trash(a.character, a.filename, card),
              }),
            ]),
          );
          return card;
        }),
      ),
    );
  }
  if (mismatches.length) {
    parts.push(el("h4", { class: "issues-title", text: `画数がそろっていない（${mismatches.length}）` }));
    for (const m of mismatches) {
      const card = el("div", { class: "mismatch" }, [
        el("div", { class: "issue-top" }, [
          el("span", { class: "issue-char", text: m.character }),
          el("span", { class: "reason mild", text: `多いのは ${m.mode_count} 画` }),
        ]),
        el(
          "div",
          { class: "mismatch-row" },
          m.samples.map((s) =>
            sampleThumb(s, { outlier: s.is_outlier, onDelete: s.is_outlier ? (node) => trash(m.character, s.filename, node) : null }),
          ),
        ),
        el("div", { class: "issue-actions" }, [
          el("button", {
            class: "btn btn-quiet small",
            type: "button",
            html: `${icon("i-check")}どれも正しい`,
            onclick: async () => {
              for (const s of m.samples.filter((x) => x.is_outlier)) {
                await api("/api/collect/samples/metadata", {
                  method: "POST",
                  body: { profile: state.profile, char: m.character, file: s.filename, key: "ignore_stroke_mismatch", value: true },
                });
              }
              card.classList.add("is-gone");
              window.setTimeout(loadReview, 260);
            },
          }),
        ]),
      ]);
      card.classList.add("issue");
      parts.push(card);
    }
  }
  box.replaceChildren(...parts);
}

async function openDrawer(char) {
  if (!state.profile) return;
  state.drawerChar = char;
  $("drawer").hidden = false;
  $("scrim").hidden = false;
  $("drawerChar").textContent = char;
  $("drawerSamples").replaceChildren(...Array.from({ length: 3 }, () => el("div", { class: "skeleton" })));
  $("drawerPreview").replaceChildren(...Array.from({ length: 3 }, () => el("div", { class: "skeleton" })));
  try {
    const [info, samples] = await Promise.all([
      api("/api/collect/next", { params: { profile: state.profile, char } }),
      api("/api/collect/samples", { params: { profile: state.profile, char } }),
    ]);
    if (state.drawerChar !== char) return;
    $("drawerTier").textContent = info.tier_label;
    $("drawerTier").dataset.tier = info.tier;
    $("drawerCount").textContent = `${samples.length} / ${info.target} サンプル`;
    $("drawerDeleteAll").disabled = samples.length === 0;
    $("drawerSamples").replaceChildren(
      ...(samples.length
        ? samples.map((s) => sampleThumb(s, { onDelete: (node) => trash(char, s.filename, node) }))
        : [el("p", { class: "preview-note", text: "まだ書いていません。「この字を書く」から書けます" })]),
    );
  } catch (error) {
    toast(error.message, "error");
  }
  loadPreview(char);
}

async function loadPreview(char) {
  try {
    const data = await api("/api/collect/preview", { params: { profile: state.profile, char, n: 3 } });
    if (state.drawerChar !== char) return;
    const label = {
      user_strokes: "あなたの筆跡から",
      ml_inference: "ML があなた風に変形",
      kanjivg: "お手本の字形のまま（筆跡を集めると変わります）",
      geometric: "幾何字形",
      missing_glyphs: "字形がありません",
    }[data.source];
    const thumbs = data.variants.map((v) => {
      const canvas = el("canvas");
      requestAnimationFrame(() => drawStrokes(canvas, v.map(unflatten), { yUp: true, minSpan: 6, margin: 0.16 }));
      return el("div", { class: "thumb" }, canvas);
    });
    $("drawerPreview").replaceChildren(...thumbs, el("p", { class: "preview-note", text: label }));
  } catch {
    $("drawerPreview").replaceChildren(el("p", { class: "preview-note", text: "試し書きできませんでした" }));
  }
}

function closeDrawer() {
  state.drawerChar = null;
  $("drawer").hidden = true;
  $("scrim").hidden = true;
  $("searchInput").value = "";
}

async function deleteAll() {
  const char = state.drawerChar;
  if (!char) return;
  const { files } = await api("/api/collect/samples", { method: "DELETE", params: { profile: state.profile, char } });
  toastUndo(`「${char}」のサンプルを ${files.length} 件消しました`, async () => {
    await api("/api/collect/samples/restore", { method: "POST", body: { profile: state.profile, char, files } });
    refreshAfterEdit(char);
  });
  refreshAfterEdit(char);
}

// ------------------------------------------------------------------ 学習

function bindTrain() {
  for (const id of ["trainDataset", "trainKind"]) {
    $(id).addEventListener("click", (e) => {
      const b = e.target.closest(".seg-btn");
      if (!b) return;
      $(id).dataset.active = b.dataset.value;
      $(id).querySelectorAll(".seg-btn").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      $("trainBaseWrap").hidden = $("trainKind").dataset.active !== "finetune";
    });
  }
  $("trainStart").addEventListener("click", startTraining);
  $("trainCancel").addEventListener("click", async () => {
    renderTraining(await api("/api/training/cancel", { method: "POST" }));
  });
  $("useNew").addEventListener("click", () => useModel(state.training?.checkpoint_name));
}

async function startTraining() {
  if (!needProfile()) return;
  const num = (id) => (($(id).value || "").trim() ? Number($(id).value) : null);
  const body = {
    profile: state.profile,
    kind: $("trainKind").dataset.active,
    dataset: $("trainDataset").dataset.active,
    epochs: num("trainEpochs"),
    batch_size: num("trainBatch"),
    learning_rate: num("trainLr"),
    deformer_type: $("trainDeformer").value,
    device: $("trainDevice").value.trim() || null,
    output: $("trainOutput").value.trim() || null,
    checkpoint: $("trainKind").dataset.active === "finetune" ? $("trainBase").value || null : null,
  };
  try {
    state.lossHistory = [];
    renderTraining(await api("/api/training/start", { method: "POST", body }));
    toast("学習を始めました", "ok", 2000);
  } catch (error) {
    toast(`始められません: ${error.message}`, "error", 5000);
  }
}

async function loadTraining() {
  try {
    renderTraining(await api("/api/training"));
  } catch {
    // サーバー再起動中など
  }
}

function renderTraining(s) {
  const prev = state.training?.state;
  state.training = s;
  $("trainProfile").textContent = state.profile ? `書き手: ${state.profile}` : "";
  const running = s.state === "running";
  $("statusCard").dataset.state = s.state;
  $("trainStart").hidden = running;
  $("trainCancel").hidden = !running;
  const pct = s.total_epochs ? Math.floor((s.epoch / s.total_epochs) * 100) : 0;
  const shown = s.state === "succeeded" ? 100 : pct;
  $("trainRing").style.strokeDasharray = `${shown} 100`;
  $("trainRing").classList.toggle("has-value", shown > 0);
  $("trainPct").textContent = shown;
  const titles = { idle: "待機中", running: "学習しています", succeeded: "学習が終わりました", failed: "学習に失敗しました", cancelled: "止めました" };
  $("trainTitle").textContent = titles[s.state] || s.state;
  let sub = "学習を始めると、ここに進み具合が出ます";
  if (running) {
    const elapsed = s.started_at ? Date.now() / 1000 - s.started_at : 0;
    const remain = s.epoch > 0 ? (elapsed / s.epoch) * (s.total_epochs - s.epoch) : NaN;
    sub =
      s.epoch === 0
        ? "準備しています（筆跡とお手本を読み込み中）"
        : `エポック ${s.epoch} / ${s.total_epochs}${Number.isFinite(remain) ? ` · 残り 約${formatDuration(remain)}` : ""}`;
  } else if (s.state === "succeeded") {
    sub = `保存しました: ${s.checkpoint_name || s.checkpoint_path}`;
  } else if (s.state === "failed") {
    sub = s.error || "";
  }
  $("trainSub").textContent = sub;
  $("trainLoss").textContent = s.loss === null || s.loss === undefined ? "—" : Number(s.loss).toFixed(4);
  const losses = (s.logs || []).map((l) => /loss=([\d.eE+-]+)/.exec(l)).filter(Boolean).map((m) => Number(m[1]));
  drawSpark(losses);
  $("trainLog").textContent = (s.logs || []).join("\n");
  const fresh = s.state === "succeeded" && s.checkpoint_name && s.checkpoint_name !== state.models.active;
  $("useNewWrap").hidden = !fresh;
  if (prev === "running" && s.state !== "running") {
    loadModels();
    if (s.state === "succeeded") toast("学習が終わりました。「このモデルを使う」でスタジオに反映できます", "ok", 6000);
    if (s.state === "failed") toast(`学習に失敗しました: ${s.error}`, "error", 6000);
  }
}

function drawSpark(values) {
  if (values.length < 2) {
    $("sparkLine").setAttribute("d", "");
    $("sparkArea").setAttribute("d", "");
    return;
  }
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values.map((v, i) => [(i / (values.length - 1)) * 300, 58 - ((v - lo) / span) * 50]);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
  $("sparkLine").setAttribute("d", line);
  $("sparkArea").setAttribute("d", `${line}L300,64L0,64Z`);
}

async function loadModels() {
  try {
    state.models = await api("/api/models");
  } catch {
    return;
  }
  const { models, active } = state.models;
  $("modelsMeta").textContent = active ? `使用中: ${active}` : "ML を使っていません";
  const list = $("models");
  const newest = state.training?.state === "succeeded" ? state.training.checkpoint_name : null;
  const rows = models.map((m) => {
    const isActive = m.name === active;
    const date = new Date(m.modified * 1000).toLocaleString("ja-JP", { dateStyle: "short", timeStyle: "short" });
    return el("li", { class: `model${isActive ? " is-active" : ""}${m.name === newest ? " is-new" : ""}` }, [
      el("div", {}, [el("div", { class: "model-name", text: m.name }), el("div", { class: "model-meta", text: `${date} · ${(m.size / 1e6).toFixed(1)} MB` })]),
      isActive
        ? el("span", { class: "badge", html: `${icon("i-check")}使用中` })
        : el("button", { class: "btn btn-quiet small", type: "button", text: "使う", onclick: () => useModel(m.name) }),
    ]);
  });
  if (!rows.length) rows.push(el("li", { class: "models-empty", text: "まだモデルがありません。左で学習させると、ここに並びます" }));
  if (active) {
    rows.push(el("li", { class: "model" }, [el("div", { class: "model-meta", text: "ML を使わずに清書する" }), el("button", { class: "btn btn-quiet small", type: "button", text: "使わない", onclick: () => useModel(null) })]));
  }
  list.replaceChildren(...rows);
  const base = $("trainBase");
  base.replaceChildren(...models.map((m) => el("option", { value: m.name, text: m.name })));
  if (models.some((m) => m.name === "pretrain_checkpoint.pt")) base.value = "pretrain_checkpoint.pt";
  if (state.training) $("useNewWrap").hidden = !(state.training.state === "succeeded" && state.training.checkpoint_name && state.training.checkpoint_name !== active);
}

async function useModel(name) {
  try {
    await api("/api/models/use", { method: "POST", body: { name } });
    toast(name ? `スタジオの清書に「${name}」を使います` : "ML を使わずに清書します", "ok", 3000);
    await loadModels();
  } catch (error) {
    toast(error.message, "error");
  }
}

// ------------------------------------------------------------------ キーボード

function bindKeyboard() {
  document.addEventListener("keydown", (e) => {
    if (e.target.closest?.("input, textarea, select")) return;
    if (e.key === "Escape" && !$("drawer").hidden) {
      closeDrawer();
      return;
    }
    if (state.mode !== "write") return;
    const mod = e.metaKey || e.ctrlKey;
    if (e.key === "Enter") {
      e.preventDefault();
      save();
    } else if ((mod && e.key.toLowerCase() === "z") || e.key === "Backspace") {
      e.preventDefault();
      pad.undo();
    } else if (e.key === "Escape") {
      pad.clear();
    } else if (e.key === "ArrowRight") {
      skip();
    }
  });
}

boot().catch((error) => {
  console.error(error);
  toast(`起動に失敗しました: ${error.message || error}`, "error", 10000);
});

