// Pen Plotter — 手書きスタジオ
// 書く（原稿＋ライブ下書き）→ 清書（手書きストローク＋同一の G-code）→ 描く（WebSerial でプロッタへ）。

import { Editor } from "./editor.js";
import { PaperView } from "./paper.js";
import { Plotter, estimateLineSeconds, normalizeGcode, parseGcode, unsupportedReason } from "./plotter.js";

const $ = (id) => document.getElementById(id);
const isMac = /Mac|iPhone|iPad/.test(navigator.platform);
const MOD = isMac ? "⌘" : "Ctrl";

const STORAGE = {
  get(key, fallback = null) {
    try {
      const v = localStorage.getItem(`pp.${key}`);
      return v === null ? fallback : JSON.parse(v);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(`pp.${key}`, JSON.stringify(value));
    } catch {
      // 保存できなくても動作は続ける
    }
  },
};

// 仕上がりの雰囲気（筆跡 3 軸の組み合わせ）
const PRESETS = [
  { id: "neat", label: "端正", values: { temperature: 0.1, messiness: 0.2, instance_variation: 0.05 }, path: "M4 11.5h36" },
  { id: "natural", label: "自然", values: { temperature: 0.2, messiness: 0.4, instance_variation: 0.1 }, path: "M4 12c6-2 10 1 16-1s10-1 16-1" },
  { id: "casual", label: "くだけた", values: { temperature: 0.5, messiness: 0.9, instance_variation: 0.3 }, path: "M4 13c4-5 8 3 12-1s6-6 10-2 6 3 10-1" },
  { id: "rough", label: "走り書き", values: { temperature: 0.9, messiness: 1.4, instance_variation: 0.55 }, path: "M4 15c3-9 6 7 9-2s4-8 7 0 4 7 7-3 5-5 9 1" },
];

const COVERAGE_TIERS = [
  { key: "user_strokes", label: "あなたの筆跡", color: "var(--accent)" },
  { key: "composed", label: "部品から組み立て", color: "#7a9e7e" },
  { key: "ml_inference", label: "ML で変形", color: "var(--pencil)" },
  { key: "kanjivg", label: "KanjiVG 字形", color: "var(--text-2)" },
  { key: "geometric", label: "幾何字形", color: "#c2a46b" },
  { key: "missing_glyphs", label: "未収録", color: "var(--danger)" },
];

const store = {
  boot: null,
  settings: {},
  profile: null,
  japaneseOnly: false,
  seed: 0,
  draft: null,
  render: null, // {pages, coverage, seed, elapsed, key}
  rendering: false,
  page: 0,
  tab: "style",
  source: "render", // ジョブの元: render | upload
  upload: null,
  selected: new Set(),
  lastOutcome: null,
  knownPort: null,
};

const plotter = new Plotter();
let editor;
let paper;

// ------------------------------------------------------------------ helpers

function randomSeed() {
  return Math.floor(Math.random() * 90000) + 10000;
}

function inputKey() {
  return JSON.stringify([editor.value, store.settings, store.profile, store.japaneseOnly, store.seed]);
}

const isFresh = () => Boolean(store.render && store.render.key === inputKey());
const hasText = () => editor.value.trim().length > 0;

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "";
  if (seconds < 60) return "1分未満";
  const m = Math.round(seconds / 60);
  if (m < 60) return `${m}分`;
  return `${Math.floor(m / 60)}時間${m % 60 ? `${m % 60}分` : ""}`;
}

function toast(message, kind = "info", timeout = 3400) {
  const icons = { ok: "i-check", warn: "i-alert", error: "i-alert", info: "i-sparkle" };
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.innerHTML = `<svg class="icon"><use href="#${icons[kind] || icons.info}"/></svg><span></span>`;
  node.querySelector("span").textContent = message;
  $("toasts").append(node);
  window.setTimeout(() => {
    node.classList.add("is-leaving");
    node.addEventListener("animationend", () => node.remove(), { once: true });
  }, timeout);
}

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), ms);
  };
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) node.append(c);
  return node;
}

// ------------------------------------------------------------------ 起動

async function boot() {
  const res = await fetch("/api/bootstrap");
  store.boot = await res.json();
  const defaults = store.boot.settings;
  store.settings = { ...defaults, ...pick(STORAGE.get("settings", {}), defaults) };
  const profiles = store.boot.profiles.map((p) => p.id);
  const savedProfile = STORAGE.get("profile");
  store.profile = profiles.includes(savedProfile) ? savedProfile : profiles[0] || null;
  store.japaneseOnly = Boolean(STORAGE.get("japaneseOnly", false));
  store.seed = Number(STORAGE.get("seed")) || randomSeed();

  editor = new Editor($("textInput"), $("editorLayer"));
  editor.value = STORAGE.get("text", "") || "";

  paper = new PaperView($("paperCanvas"), store.boot.paper);
  if (store.boot.paper.background) paper.setBackground("/api/paper");
  paper.on((type, detail) => {
    if (type === "zoom") $("zoomText").textContent = `${detail}%`;
    if (type === "animationend") updateStage();
  });
  applyInsets();
  window.addEventListener("resize", applyInsets);

  buildExamples();
  buildSyntax();
  buildPresets();
  buildSections();
  buildProfile();
  bindEditor();
  bindStage();
  bindInspector();
  bindPlotter();
  bindKeyboard();
  bindFlow();

  $("japaneseOnly").checked = store.japaneseOnly;
  $("seedInput").value = store.seed;
  if (!isMac) {
    document.querySelectorAll(".kbd-hint, .shortcuts kbd").forEach((k) => {
      k.textContent = k.textContent.replace("⌘", "Ctrl");
    });
    document.querySelector(".modal-note")?.remove();
  }
  refreshAll();
  requestLayout();
}

function pick(obj, template) {
  const out = {};
  for (const k of Object.keys(template)) {
    if (obj && obj[k] !== undefined && typeof obj[k] === typeof template[k]) out[k] = obj[k];
  }
  return out;
}

function applyInsets() {
  const small = window.innerWidth <= 640;
  paper.insets = small
    ? { top: 56, bottom: 72, left: 12, right: 12 }
    : { top: 64, bottom: 84, left: 28, right: 28 };
  if (paper.fitted) paper.fit();
}

// ------------------------------------------------------------------ 原稿

function buildExamples() {
  const menu = $("examplesMenu");
  const pop = menu.querySelector(".menu-pop");
  const toggle = menu.querySelector("button");
  for (const ex of store.boot.examples) {
    const first = ex.text.split("\n").find((l) => l.trim() && !l.startsWith("#")) || ex.text;
    pop.append(
      el("button", { class: "menu-item", type: "button", role: "menuitem", onclick: () => insertExample(ex) }, [
        el("b", { text: ex.label }),
        el("span", { text: first.slice(0, 40) }),
      ]),
    );
  }
  const close = () => {
    menu.classList.remove("is-open");
    toggle.setAttribute("aria-expanded", "false");
  };
  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = !menu.classList.contains("is-open");
    menu.classList.toggle("is-open", open);
    toggle.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", close);
  menu.addEventListener("keydown", (e) => e.key === "Escape" && close());
}

function insertExample(ex) {
  const text = editor.value.trim();
  editor.value = text ? `${editor.value.replace(/\s+$/, "")}\n\n${ex.text}` : ex.text;
  editor.textarea.focus();
  toast(`「${ex.label}」を挿入しました`, "ok", 2200);
}

function buildSyntax() {
  const grid = $("syntaxGrid");
  for (const row of store.boot.syntax) {
    const code = el("code", { text: row.example, title: "クリックで挿入" });
    code.addEventListener("click", () => {
      if (row.example.startsWith("（")) return;
      editor.insert(row.example);
    });
    grid.append(el("div", { class: "syntax-name", text: row.name }), el("div", { class: "syntax-ex" }, [code, el("small", { text: row.note })]));
  }
  $("syntaxBtn").addEventListener("click", () => {
    const panel = $("syntaxPanel");
    panel.hidden = !panel.hidden;
    $("syntaxBtn").setAttribute("aria-expanded", String(!panel.hidden));
  });
}

function bindEditor() {
  const save = debounce(() => STORAGE.set("text", editor.value), 400);
  editor.textarea.addEventListener("input", () => {
    save();
    updateStats();
    requestLayout();
    refreshAll();
  });
  $("clearBtn").addEventListener("click", () => {
    if (!hasText()) return;
    const previous = editor.value;
    editor.value = "";
    const node = toastUndo("原稿を消去しました", () => {
      editor.value = previous;
    });
    return node;
  });
  $("emptyExample").addEventListener("click", () => insertExample(store.boot.examples[0]));
}

function toastUndo(message, undo) {
  const node = document.createElement("div");
  node.className = "toast";
  node.innerHTML = '<svg class="icon"><use href="#i-trash"/></svg><span></span>';
  node.querySelector("span").textContent = message;
  const btn = el("button", { class: "link-btn", type: "button", text: "元に戻す" });
  btn.style.color = "inherit";
  btn.addEventListener("click", () => {
    undo();
    node.remove();
  });
  node.append(btn);
  $("toasts").append(node);
  window.setTimeout(() => {
    node.classList.add("is-leaving");
    node.addEventListener("animationend", () => node.remove(), { once: true });
  }, 6000);
}

function updateStats() {
  const chars = editor.value.replace(/\s/g, "").length;
  $("statChars").textContent = chars.toLocaleString();
  const pages = hasText() && store.draft ? store.draft.pages.length : 0;
  $("statPages").textContent = pages;
}

// ------------------------------------------------------------------ 下書き（組版のみ・入力に追従）

let layoutAbort = null;
const requestLayout = debounce(async () => {
  layoutAbort?.abort();
  layoutAbort = new AbortController();
  try {
    const res = await fetch("/api/layout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: editor.value, settings: store.settings }),
      signal: layoutAbort.signal,
    });
    store.draft = await res.json();
    showValidation(store.draft.errors);
    paper.setRuled(store.draft.ruled);
    if (store.page >= displayPages().length) store.page = Math.max(0, displayPages().length - 1);
    updateStats();
    refreshAll();
  } catch (error) {
    if (error.name !== "AbortError") console.warn(error);
  }
}, 180);

function showValidation(errors) {
  const box = $("validation");
  box.hidden = !errors.length;
  box.innerHTML = errors.length ? `<ul>${errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>` : "";
}

const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ------------------------------------------------------------------ 清書

async function doRender({ reroll = false } = {}) {
  if (store.rendering || !hasText() || plotter.running) return;
  if (store.draft?.errors?.length) {
    toast("設定に問題があります。右のパネルを確認してください", "warn");
    return;
  }
  if (reroll) {
    store.seed = randomSeed();
    $("seedInput").value = store.seed;
    STORAGE.set("seed", store.seed);
  }
  store.rendering = true;
  const key = inputKey();
  setRenderProgress(0, "準備中");
  refreshAll();
  try {
    const res = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: editor.value,
        settings: store.settings,
        profile: store.profile,
        japanese_only: store.japaneseOnly,
        seed: store.seed,
      }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let result = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let i;
      while ((i = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, i);
        buffer = buffer.slice(i + 1);
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "progress") setRenderProgress(event.fraction, event.message);
        else if (event.type === "error") throw new Error(event.message);
        else if (event.type === "result") result = event;
      }
    }
    if (!result) throw new Error("サーバーから結果が返りませんでした");
    store.render = { ...result, key };
    store.page = Math.min(store.page, result.pages.length - 1);
    store.source = "render";
    store.selected = new Set(result.pages.map((_, i) => i));
    store.lastOutcome = null;
    paper.setInk(result.pages[store.page], { animate: true });
    showCoverage(result.coverage);
    const strokes = result.pages.reduce((n, p) => n + p.strokes.length, 0);
    toast(`${result.pages.length}ページを清書しました — ${strokes.toLocaleString()}画・${result.elapsed}秒`, "ok");
  } catch (error) {
    toast(error.message || String(error), "error", 5200);
  } finally {
    store.rendering = false;
    $("renderProgress").hidden = true;
    refreshAll();
  }
}

function setRenderProgress(fraction, message) {
  $("renderProgress").hidden = false;
  $("rpFill").style.width = `${Math.round(fraction * 100)}%`;
  $("rpPct").textContent = `${Math.round(fraction * 100)}%`;
  $("rpText").textContent = message;
}

function showCoverage(coverage) {
  const total = COVERAGE_TIERS.reduce((n, t) => n + (coverage[t.key]?.count || 0), 0);
  $("coverageCard").hidden = total === 0;
  if (!total) return;
  $("coverageMeta").textContent = `${total.toLocaleString()}字`;
  const bar = $("coverageBar");
  const legend = $("coverageLegend");
  bar.replaceChildren();
  legend.replaceChildren();
  for (const t of COVERAGE_TIERS) {
    const n = coverage[t.key]?.count || 0;
    if (!n) continue;
    const seg = el("span", { title: `${t.label}: ${coverage[t.key].chars.slice(0, 80)}` });
    seg.style.background = t.color;
    seg.style.flexGrow = String(n);
    bar.append(seg);
    const dot = el("i");
    dot.style.background = t.color;
    legend.append(el("li", {}, [dot, el("span", { text: t.label }), el("b", { text: n.toLocaleString() })]));
  }
  const missing = coverage.missing_glyphs?.chars || "";
  $("coverageMissing").hidden = !missing;
  $("coverageMissing").textContent = missing ? `字形が無く空白になった字: ${missing}` : "";
}

function downloadGcode() {
  if (!store.render) return;
  const pages = store.render.pages;
  pages.forEach((page, i) => {
    const name = pages.length === 1 ? "handwriting.gcode" : `handwriting_p${i + 1}.gcode`;
    window.setTimeout(() => {
      const blob = new Blob([`${page.gcode.join("\n")}\n`], { type: "text/plain" });
      const a = el("a", { href: URL.createObjectURL(blob), download: name });
      document.body.append(a);
      a.click();
      a.remove();
      window.setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    }, i * 350);
  });
  toast(`G-code を${pages.length > 1 ? ` ${pages.length} ファイル` : ""}保存しました`, "ok", 2400);
}

// ------------------------------------------------------------------ 設定パネル

function buildPresets() {
  const box = $("presets");
  for (const p of PRESETS) {
    const btn = el(
      "button",
      { class: "preset", type: "button", role: "radio", "aria-checked": "false", "data-id": p.id },
      [el("span", { html: `<svg viewBox="0 0 44 22"><path d="${p.path}"/></svg>` }), el("span", { text: p.label })],
    );
    btn.addEventListener("click", () => {
      Object.assign(store.settings, p.values);
      settingsChanged();
      syncControls();
    });
    box.append(btn);
  }
}

function syncPresets() {
  for (const btn of $("presets").children) {
    const p = PRESETS.find((x) => x.id === btn.dataset.id);
    const active = Object.entries(p.values).every(([k, v]) => Math.abs(store.settings[k] - v) < 1e-6);
    btn.setAttribute("aria-checked", String(active));
  }
}

const controlInputs = new Map();

function buildSections() {
  const root = $("sections");
  const openState = STORAGE.get("sectionsOpen", {});
  for (const section of store.boot.sections) {
    const group = el("div", { class: "field-group" });
    const body = el("div", { class: "group-body" });
    if (section.collapsed) {
      group.classList.add("is-collapsible");
      if (openState[section.id]) group.classList.add("is-open");
      const head = el("button", { class: "group-head", type: "button", "aria-expanded": String(Boolean(openState[section.id])) }, [
        el("h3", { class: "group-title", text: section.title }),
        el("span", { html: '<svg class="icon"><use href="#i-chevron"/></svg>' }),
      ]);
      head.addEventListener("click", () => {
        const open = group.classList.toggle("is-open");
        head.setAttribute("aria-expanded", String(open));
        STORAGE.set("sectionsOpen", { ...STORAGE.get("sectionsOpen", {}), [section.id]: open });
      });
      group.append(head);
    } else {
      group.append(el("div", { class: "group-head" }, [el("h3", { class: "group-title", text: section.title })]));
    }
    for (const c of section.controls) body.append(c.kind === "toggle" ? toggleControl(c) : rangeControl(c));
    group.append(body);
    root.append(group);
  }
}

function rangeControl(c) {
  const id = `ctl-${c.field}`;
  const input = el("input", { type: "range", id, min: c.minimum, max: c.maximum, step: c.step });
  const value = el("output", { class: "range-value", for: id });
  const label = el("label", { class: "field-label", for: id, text: c.label, title: "ダブルクリックで既定値" });
  if (c.hardware_note) label.append(el("span", { class: "hw-tag", text: "実機は0推奨" }));
  label.addEventListener("dblclick", () => {
    store.settings[c.field] = store.boot.settings[c.field];
    settingsChanged();
    syncControls();
  });
  input.addEventListener("input", () => {
    store.settings[c.field] = parseFloat(input.value);
    paintRange(c, input, value);
    settingsChanged();
    syncPresets();
  });
  controlInputs.set(c.field, () => {
    input.value = store.settings[c.field];
    paintRange(c, input, value);
  });
  const field = el("div", { class: "range-field" }, [el("div", { class: "range-top" }, [label, value]), input]);
  if (c.info) field.append(el("span", { class: "field-info", text: c.info }));
  return field;
}

function paintRange(c, input, value) {
  const v = parseFloat(input.value);
  const pct = ((v - c.minimum) / (c.maximum - c.minimum)) * 100;
  input.style.setProperty("--fill", `${pct}%`);
  const digits = c.step >= 1 ? 0 : c.step >= 0.1 ? 1 : 2;
  value.textContent = `${v.toFixed(digits)}${c.unit ? ` ${c.unit}` : ""}`;
  value.classList.toggle("is-changed", Math.abs(v - store.boot.settings[c.field]) > 1e-9);
}

function toggleControl(c) {
  const input = el("input", { type: "checkbox", class: "switch" });
  input.addEventListener("change", () => {
    store.settings[c.field] = input.checked;
    settingsChanged();
  });
  controlInputs.set(c.field, () => {
    input.checked = Boolean(store.settings[c.field]);
  });
  return el("label", { class: "switch-row" }, [
    el("span", {}, [el("span", { class: "field-label", text: c.label }), c.info ? el("span", { class: "field-info", text: c.info }) : null]),
    input,
  ]);
}

function syncControls() {
  for (const sync of controlInputs.values()) sync();
  syncPresets();
}

const persistSettings = debounce(() => STORAGE.set("settings", store.settings), 300);

function settingsChanged() {
  persistSettings();
  requestLayout();
  refreshAll();
}

function buildProfile() {
  const profiles = store.boot.profiles;
  $("profileField").hidden = profiles.length === 0;
  const select = $("profileSelect");
  for (const p of profiles) {
    select.append(el("option", { value: p.id, text: `${p.id}（${p.characters}字・${p.samples}サンプル）` }));
  }
  if (store.profile) select.value = store.profile;
  select.addEventListener("change", () => {
    store.profile = select.value;
    STORAGE.set("profile", store.profile);
    refreshAll();
  });
}

function bindInspector() {
  syncControls();
  $("japaneseOnly").addEventListener("change", (e) => {
    store.japaneseOnly = e.target.checked;
    STORAGE.set("japaneseOnly", store.japaneseOnly);
    refreshAll();
  });
  const seedInput = $("seedInput");
  seedInput.addEventListener("change", () => {
    const v = parseInt(seedInput.value.replace(/\D/g, ""), 10);
    store.seed = Number.isFinite(v) ? v : store.seed;
    seedInput.value = store.seed;
    STORAGE.set("seed", store.seed);
    refreshAll();
  });
  $("seedDice").addEventListener("click", () => {
    store.seed = randomSeed();
    seedInput.value = store.seed;
    STORAGE.set("seed", store.seed);
    refreshAll();
  });
  $("resetBtn").addEventListener("click", () => {
    store.settings = { ...store.boot.settings };
    settingsChanged();
    syncControls();
    toast("設定を既定値に戻しました", "ok", 2000);
  });
  for (const tab of [$("tabStyle"), $("tabPlot")]) {
    tab.addEventListener("click", () => selectTab(tab === $("tabPlot") ? "plot" : "style"));
  }
}

function selectTab(name) {
  const plot = name === "plot";
  $("tabStyle").setAttribute("aria-selected", String(!plot));
  $("tabPlot").setAttribute("aria-selected", String(plot));
  $("panelStyle").hidden = plot;
  $("panelPlot").hidden = !plot;
  document.querySelector(".seg").dataset.active = name;
  store.tab = name;
  refreshAll();
}

/** プロッタ画面で G-code ファイルを選んでいるときは、それを用紙に出す。 */
const showingUpload = () => store.source === "upload" && store.upload && store.tab === "plot";

function setView(view) {
  $("app").dataset.view = view;
  document.querySelectorAll(".mobile-nav button").forEach((b) => b.setAttribute("aria-current", String(b.dataset.view === view)));
}

// ------------------------------------------------------------------ 用紙

function displayPages() {
  if (plotter.running && store.job) return store.job.pages.map((p) => p.view);
  if (showingUpload()) return store.upload.pages.map((p) => p.view);
  if (isFresh()) return store.render.pages;
  return hasText() && store.draft ? store.draft.pages : [];
}

function bindStage() {
  $("zoomIn").addEventListener("click", () => paper.zoomAt(1.25));
  $("zoomOut").addEventListener("click", () => paper.zoomAt(0.8));
  $("zoomFit").addEventListener("click", () => paper.fit());
  $("renderBtn").addEventListener("click", () => doRender());
  $("rerollBtn").addEventListener("click", () => doRender({ reroll: true }));
  $("downloadBtn").addEventListener("click", downloadGcode);
  $("plotBtn").addEventListener("click", openPlot);
  $("dockPause").addEventListener("click", togglePause);
  $("dockEstop").addEventListener("click", () => plotter.emergencyStop());
  document.querySelectorAll(".mobile-nav button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  setView("paper");
}

function goPage(i) {
  const pages = displayPages();
  if (!pages.length || plotter.running) return;
  store.page = Math.max(0, Math.min(pages.length - 1, i));
  refreshAll();
}

let shownInk = null;

function updateStage() {
  const running = plotter.running && store.job;
  const fresh = isFresh();
  const text = hasText();
  const draftPages = store.draft?.pages || [];
  $("stageEmpty").hidden = text || running || showingUpload();

  if (running) {
    const p = plotter.progress;
    const page = store.job.pages[p ? p.pageIndex : 0];
    if (shownInk !== page.view) {
      paper.setInk(page.view);
      shownInk = page.view;
    }
    paper.showDraft = false;
    paper.setInkAlpha(1);
    paper.setPlotProgress(p ? p.line : 0);
  } else {
    paper.setPlotProgress(null);
    const upload = showingUpload();
    const renderPage = upload
      ? store.upload.pages[Math.min(store.page, store.upload.pages.length - 1)].view
      : store.render?.pages[store.page] || null;
    if (shownInk !== renderPage && !(fresh && paper.anim)) {
      paper.setInk(renderPage);
      shownInk = renderPage;
    } else if (shownInk !== renderPage) {
      shownInk = renderPage;
    }
    paper.showDraft = !upload && !fresh && text;
    paper.setInkAlpha(upload || fresh ? 1 : 0.16);
    paper.setDraft(text ? draftPages[store.page] || null : null, store.settings.line_spacing);
  }

  // モード表示
  const chip = $("modeChip");
  let mode = "empty";
  let label = "白紙";
  if (running) {
    mode = "plot";
    label = `描画中 · ${plotter.progress?.pageNo ?? 1}ページ目`;
  } else if (store.rendering) {
    mode = "busy";
    label = "清書中…";
  } else if (showingUpload()) {
    mode = "ink";
    label = `G-code · ${store.upload.name}`;
  } else if (fresh) {
    mode = "ink";
    label = `清書 · No.${store.render.seed}`;
  } else if (text && store.render) {
    mode = "stale";
    label = "下書き（清書は古い）";
  } else if (text) {
    mode = "draft";
    label = "下書き";
  }
  chip.dataset.mode = mode;
  $("modeText").textContent = label;

  // ページタブ
  const pages = displayPages();
  const tabs = $("pageTabs");
  const want = pages.length > 1 ? pages.length : 0;
  if (tabs.children.length !== want) {
    tabs.replaceChildren(
      ...Array.from({ length: want }, (_, i) =>
        el("button", { class: "page-tab", type: "button", role: "tab", text: String(i + 1), onclick: () => goPage(i) }),
      ),
    );
  }
  const current = running ? plotter.progress?.pageIndex ?? 0 : store.page;
  [...tabs.children].forEach((t, i) => t.setAttribute("aria-selected", String(i === current)));
}

// ------------------------------------------------------------------ ドック・流れ

function refreshAll() {
  if (!editor) return;
  updateStage();
  updateDock();
  updateFlow();
  updatePlotPanel();
}

function updateDock() {
  const running = plotter.running;
  const fresh = isFresh();
  const text = hasText();
  const show = (id, v) => {
    $(id).hidden = !v;
  };
  show("renderBtn", !running && !fresh);
  show("rerollBtn", !running && fresh);
  show("downloadBtn", !running && fresh);
  show("dockSep", !running && fresh);
  show("plotBtn", !running && fresh);
  show("dockRun", running);
  show("dockPause", running);
  show("dockEstop", running);

  const renderBtn = $("renderBtn");
  renderBtn.disabled = !text || store.rendering || Boolean(store.draft?.errors?.length);
  $("renderLabel").textContent = store.rendering ? "清書しています…" : store.render ? "清書し直す" : "清書する";
  renderBtn.classList.toggle("is-emphasis", text && !store.render && !store.rendering);
  $("plotLabel").textContent = plotter.connected ? "プロッタで描く" : "接続して描く";

  if (running) {
    const p = plotter.progress;
    const pct = p ? (p.sent / p.total) * 100 : 0;
    $("dockRunFill").style.width = `${pct}%`;
    $("dockRunText").textContent =
      plotter.status === "paused" ? "一時停止中" : plotter.status === "paper" ? "用紙交換待ち" : plotter.status === "pausing" ? "止まります…" : `描画中 ${Math.floor(pct)}%`;
    $("dockRunEta").textContent = p ? `残り ${formatDuration(p.remaining)}` : "";
    const paused = plotter.status === "paused" || plotter.status === "paper";
    $("dockPause").querySelector("span").textContent = paused ? "再開" : "一時停止";
    $("dockPause").querySelector("use").setAttribute("href", paused ? "#i-play" : "#i-pause");
  }
}

function updateFlow() {
  const text = hasText();
  const fresh = isFresh();
  const states = {
    write: text ? "done" : "active",
    render: fresh ? "done" : text ? "active" : "todo",
    plot: plotter.running ? "active" : store.lastOutcome === "done" && fresh ? "done" : fresh ? "active" : "todo",
  };
  if (states.render === "active" && states.write === "done") states.plot = "todo";
  if (fresh && !plotter.running && store.lastOutcome !== "done") states.plot = "active";
  document.querySelectorAll(".flow-step").forEach((b) => {
    b.dataset.state = states[b.dataset.step];
    b.setAttribute("aria-current", String(states[b.dataset.step] === "active"));
  });
}

function bindFlow() {
  document.querySelectorAll(".flow-step").forEach((b) =>
    b.addEventListener("click", () => {
      const step = b.dataset.step;
      if (step === "write") {
        setView("editor");
        editor.textarea.focus();
      } else if (step === "render") {
        if (isFresh()) setView("paper");
        else doRender();
      } else {
        openPlot();
      }
    }),
  );
}

async function openPlot() {
  selectTab("plot");
  if (window.innerWidth <= 1020) setView("inspector");
  if (plotter.status === "unsupported") {
    toast("このブラウザではプロッタに接続できません。パネルの案内を確認してください", "warn", 4200);
    return;
  }
  if (!plotter.connected) await connectPlotter();
  if (plotter.connected) $("startBtn").focus();
}

// ------------------------------------------------------------------ プロッタ

function jobPages() {
  if (store.source === "upload" && store.upload) return store.upload.pages;
  if (!store.render) return [];
  store.render.jobPages ||= store.render.pages.map((page, i) => ({ pageNo: i + 1, lines: page.gcode, view: page }));
  return store.render.jobPages;
}

function bindPlotter() {
  const reason = unsupportedReason();
  if (reason) {
    const port = location.port ? `:${location.port}` : "";
    $("unsupported").hidden = false;
    $("unsupported").innerHTML =
      reason === "browser"
        ? "このブラウザは WebSerial に対応していません。プロッタをつないだ PC の <b>Chrome</b> か <b>Edge</b> で開いてください。G-code の保存はこのままできます。"
        : `WebSerial は安全な接続でのみ使えます。プロッタをつないだ PC で <code>http://localhost${port}</code> を開いてください。`;
    $("connectBtn").disabled = true;
    $("anyPortBtn").hidden = true;
  }
  $("connectBtn").addEventListener("click", connectPlotter);
  $("anyPortBtn").addEventListener("click", () => plotter.connect({ anyPort: true }));
  $("disconnectBtn").addEventListener("click", () => plotter.disconnect());
  $("homeBtn").addEventListener("click", () => plotter.home());
  $("penUpBtn").addEventListener("click", () => plotter.penUp());
  $("penDownBtn").addEventListener("click", () => plotter.penDown());
  $("machinePill").addEventListener("click", openPlot);
  $("startBtn").addEventListener("click", startJob);
  $("pauseBtn").addEventListener("click", () => plotter.pause());
  $("resumeBtn").addEventListener("click", () => plotter.resume());
  $("stopBtn").addEventListener("click", () => plotter.stop());
  $("estopBtn").addEventListener("click", () => plotter.emergencyStop());
  $("fileInput").addEventListener("change", (e) => loadFiles([...e.target.files]));
  $("useRenderBtn").addEventListener("click", () => {
    store.source = "render";
    store.selected = new Set(jobPages().map((_, i) => i));
    refreshAll();
  });
  $("copyLog").addEventListener("click", async () => {
    const text = logEntries.map((e) => `[${e.time}] ${e.message}`).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      toast("ログをコピーしました", "ok", 1800);
    } catch {
      toast("コピーできませんでした", "warn");
    }
  });

  plotter.addEventListener("change", () => {
    if (plotter.port) store.knownPort = plotter.port;
    refreshAll();
  });
  // G-code ファイルは用紙へドロップしても開ける
  const stage = $("stage");
  stage.addEventListener("dragover", (e) => {
    if (![...e.dataTransfer.items].some((i) => i.kind === "file")) return;
    e.preventDefault();
    stage.classList.add("is-dropping");
  });
  stage.addEventListener("dragleave", (e) => {
    if (!stage.contains(e.relatedTarget)) stage.classList.remove("is-dropping");
  });
  stage.addEventListener("drop", (e) => {
    e.preventDefault();
    stage.classList.remove("is-dropping");
    const files = [...e.dataTransfer.files].filter((f) => /\.(gcode|nc|txt)$/i.test(f.name));
    if (!files.length) {
      toast("G-code（.gcode / .nc / .txt）をドロップしてください", "warn");
      return;
    }
    selectTab("plot");
    loadFiles(files);
  });
  // 用紙の進捗は毎回（軽い）、パネル類の更新は間引く（送信ループを止めない）
  let panelTimer = 0;
  plotter.addEventListener("progress", () => {
    const p = plotter.progress;
    const page = store.job?.pages[p.pageIndex];
    if (page && shownInk !== page.view) {
      paper.setInk(page.view);
      shownInk = page.view;
    }
    paper.setPlotProgress(p.line);
    if (panelTimer) return;
    panelTimer = window.setTimeout(() => {
      panelTimer = 0;
      refreshAll();
    }, 150);
  });
  plotter.addEventListener("log", (e) => addLog(e.detail));
  plotter.addEventListener("paper", (e) => openPaperDialog(e.detail));
  plotter.addEventListener("finish", (e) => {
    const { outcome } = e.detail;
    store.lastOutcome = outcome;
    $("paperDialog").close();
    shownInk = null;
    if (outcome === "done") toast("描き終わりました。お疲れさまでした", "ok", 5000);
    refreshAll();
  });

  // 以前に許可したプロッタがあれば、ポート選択なしのワンクリックで再接続できる
  plotter.knownPort().then((port) => {
    store.knownPort = port;
    refreshAll();
  });

  $("paperContinue").addEventListener("click", () => {
    $("paperDialog").close();
    plotter.resume();
  });
  $("paperStop").addEventListener("click", () => {
    $("paperDialog").close();
    plotter.stop();
  });
  $("paperDialog").addEventListener("cancel", (e) => e.preventDefault());
}

async function loadFiles(files) {
  if (!files.length) return;
  files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const pages = [];
  for (const [i, file] of files.entries()) {
    const lines = normalizeGcode(await file.text());
    const { strokes, spans } = parseGcode(lines);
    pages.push({ pageNo: i + 1, lines, name: file.name, view: { strokes, spans, gcode: lines } });
  }
  store.upload = { name: files.length === 1 ? files[0].name : `${files.length} ファイル`, pages };
  store.source = "upload";
  store.page = 0;
  store.selected = new Set(pages.map((_, i) => i));
  $("fileInput").value = "";
  addLog({ message: `G-code を読み込みました: ${store.upload.name}`, level: "info", time: new Date() });
  refreshAll();
}

function connectPlotter() {
  return plotter.connect(store.knownPort ? { port: store.knownPort } : {});
}

function startJob() {
  const pages = jobPages().filter((_, i) => store.selected.has(i));
  if (!pages.length) {
    toast("描くページを選んでください", "warn");
    return;
  }
  const name = store.source === "upload" ? store.upload.name : `清書 No.${store.render.seed}`;
  store.job = { name, pages };
  store.lastOutcome = null;
  shownInk = null;
  plotter.run({ name, pages });
}

function togglePause() {
  if (plotter.status === "streaming") plotter.pause();
  else plotter.resume();
}

function openPaperDialog({ done, next }) {
  $("paperDoneNo").textContent = done;
  $("paperNextNo").textContent = next;
  $("paperDoneText").textContent = `${done} ページ目`;
  $("paperNextText").textContent = `${next} ページ目`;
  $("paperDialog").showModal();
  $("paperContinue").focus();
}

const STATUS_TEXT = {
  unsupported: "このブラウザは非対応",
  disconnected: "未接続",
  connecting: "接続中…",
  idle: "待機中",
  busy: "動作中…",
  streaming: "描画中",
  pausing: "停止準備中",
  paused: "一時停止中",
  paper: "用紙交換待ち",
};

function updatePlotPanel() {
  const s = plotter.status;
  const connected = plotter.connected;
  const running = plotter.running;
  $("machinePill").dataset.status = s;
  $("machineState").textContent = STATUS_TEXT[s];
  $("machineCard").dataset.status = s;
  $("plotBadge").hidden = !running;
  if (connected) {
    $("machineSub").textContent = plotter.needsHoming ? "接続済み — 描く前に自動で原点復帰します" : "接続済み — 原点復帰済み";
  } else if (s === "unsupported") {
    $("machineSub").textContent = "このブラウザからは接続できません";
  } else if (s === "connecting") {
    $("machineSub").textContent = "接続しています…";
  } else {
    $("machineSub").textContent = store.knownPort ? "前回つないだプロッタにすぐ接続できます" : "USB でつないで「接続する」を押してください";
  }
  $("connectBtn").querySelector("span").textContent = store.knownPort ? "再接続する" : "接続する";
  $("connectBtn").hidden = connected;
  $("connectBtn").disabled = s === "unsupported" || s === "connecting";
  $("disconnectBtn").hidden = !connected;
  $("disconnectBtn").disabled = running || s === "busy";
  $("anyPortBtn").hidden = connected || s === "unsupported";
  const machineIdle = s === "idle";
  for (const id of ["homeBtn", "penUpBtn", "penDownBtn"]) $(id).disabled = !machineIdle;
  $("homeBtn").classList.toggle("needs", machineIdle && plotter.needsHoming);
  $("machineControls").hidden = !connected;

  // 描くもの
  const pages = jobPages();
  const fromUpload = store.source === "upload" && store.upload;
  const src = $("jobSource");
  if (fromUpload) {
    src.innerHTML = `<svg class="icon"><use href="#i-file"/></svg><span></span>`;
    src.querySelector("span").innerHTML = `${escapeHtml(store.upload.name)}<small>読み込んだ G-code・${pages.length}ページ</small>`;
  } else if (store.render) {
    const stale = !isFresh();
    src.innerHTML = `<svg class="icon"><use href="#i-sparkle"/></svg><span>清書 No.${store.render.seed}<small>${pages.length}ページ${stale ? "・原稿の変更は未反映" : ""}</small></span>`;
  } else {
    src.innerHTML = `<svg class="icon"><use href="#i-paper"/></svg><span>まだありません<small>原稿を清書するか、G-code を開いてください</small></span>`;
  }
  $("useRenderBtn").hidden = !(fromUpload && store.render);
  renderJobThumbs(pages, running);
  const selected = pages.filter((_, i) => store.selected.has(i));
  const estimate = selected.reduce((n, p) => n + (p._est ??= estimateLineSeconds(p.lines).reduce((a, b) => a + b, 0)), 0);
  $("jobEstimate").textContent = selected.length ? `約${formatDuration(estimate * 1.25)}` : "";

  // 実行
  const p = plotter.progress;
  const pct = running && p ? Math.floor((p.sent / p.total) * 100) : store.lastOutcome === "done" ? 100 : 0;
  $("ringFg").style.strokeDasharray = `${pct} 100`;
  $("ringFg").classList.toggle("has-value", pct > 0);
  $("ringPct").textContent = pct;
  let title;
  let sub;
  if (!connected) {
    title = "プロッタ未接続";
    sub = s === "unsupported" ? "G-code を保存して別の方法で送れます" : "上のボタンで接続してください";
  } else if (running) {
    title =
      s === "paused"
        ? "一時停止中"
        : s === "pausing"
          ? "ペンを上げたら止まります"
          : s === "paper"
            ? "用紙の交換を待っています"
            : `描いています — ${p?.pageNo ?? 1}ページ目`;
    sub = p ? `残り 約${formatDuration(p.remaining)}${p.pageCount > 1 ? ` · ${p.pageIndex + 1} / ${p.pageCount} 枚目` : ""}` : "";
  } else if (store.lastOutcome === "done") {
    title = "描き終わりました";
    sub = "別の原稿もそのまま続けて描けます";
  } else if (!selected.length) {
    title = "描くものがありません";
    sub = "原稿を清書するか、G-code を開いてください";
  } else {
    title = "準備ができました";
    sub = "用紙を左上の角に合わせてセットしてください";
  }
  $("runTitle").textContent = title;
  $("runSub").textContent = sub;
  $("runLines").textContent = running && p ? `${p.sent.toLocaleString()} / ${p.total.toLocaleString()} 行` : "";
  $("startBtn").hidden = running;
  $("startBtn").disabled = !(s === "idle" && selected.length);
  $("pauseBtn").hidden = !(s === "streaming");
  $("resumeBtn").hidden = !(s === "paused" || s === "pausing" || s === "paper");
  $("stopBtn").hidden = !running;
  $("estopBtn").hidden = !connected;
  $("estopBtn").disabled = !running && s !== "busy";
}

let thumbKey = "";

function renderJobThumbs(pages, running) {
  const box = $("jobPages");
  const key = pages.map((p) => p.view).length + (store.source === "upload" ? store.upload?.name : store.render?.seed) + store.source;
  if (key !== thumbKey) {
    thumbKey = key;
    box.replaceChildren(
      ...(pages.length > 1 || store.source === "upload"
        ? pages.map((page, i) => {
            const btn = el("button", { class: "job-page", type: "button", "aria-pressed": "true", title: `${page.pageNo}ページ目を描く / 描かない` }, [
              el("img", { src: PaperView.thumbnail(page.view, store.boot.paper, 52), alt: "" }),
              el("span", { text: page.pageNo }),
            ]);
            btn.addEventListener("click", () => {
              if (plotter.running) return;
              if (store.selected.has(i)) store.selected.delete(i);
              else store.selected.add(i);
              refreshAll();
            });
            return btn;
          })
        : []),
    );
  }
  const p = plotter.progress;
  [...box.children].forEach((btn, i) => {
    btn.setAttribute("aria-pressed", String(store.selected.has(i)));
    const jobIndex = running && store.job ? store.job.pages.indexOf(pages[i]) : -1;
    btn.classList.toggle("is-current", running && p && jobIndex === p.pageIndex);
    btn.classList.toggle("is-done", running && p && jobIndex >= 0 && jobIndex < p.pageIndex);
  });
}

const logEntries = [];

function addLog({ message, level, time }) {
  const stamp = time.toLocaleTimeString("ja-JP", { hour12: false });
  logEntries.push({ message, level, time: stamp });
  if (logEntries.length > 400) logEntries.shift();
  const log = $("log");
  const line = el("div", { class: `log-line ${level}` }, [el("time", { text: stamp }), el("span", { text: message })]);
  log.append(line);
  while (log.children.length > 400) log.firstChild.remove();
  log.scrollTop = log.scrollHeight;
  $("logCount").textContent = logEntries.length;
  if (level === "error") toast(message, "error", 5200);
  if (level === "ok" && /接続しました/.test(message)) toast("プロッタに接続しました", "ok");
}

// ------------------------------------------------------------------ キーボード

function bindKeyboard() {
  document.addEventListener("keydown", (e) => {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key === "Enter") {
      e.preventDefault();
      doRender({ reroll: e.shiftKey });
      return;
    }
    if (mod && e.key.toLowerCase() === "s") {
      e.preventDefault();
      if (isFresh()) downloadGcode();
      else toast(`先に清書してください（${MOD}+Enter）`, "warn");
      return;
    }
    const typing = e.target.closest?.("input, textarea, select, [contenteditable]");
    if (typing || mod || e.altKey || $("paperDialog").open) return;
    if (e.key === "?") {
      $("shortcutsDialog").showModal();
    } else if (e.key === " " && plotter.running) {
      e.preventDefault();
      togglePause();
    } else if (e.key === "Escape" && plotter.status === "streaming") {
      plotter.pause();
    } else if (e.key === "ArrowLeft") {
      goPage(store.page - 1);
    } else if (e.key === "ArrowRight") {
      goPage(store.page + 1);
    } else if (e.key === "+" || e.key === "=") {
      paper.zoomAt(1.25);
    } else if (e.key === "-") {
      paper.zoomAt(0.8);
    } else if (e.key === "0") {
      paper.fit();
    }
  });
  $("shortcutsBtn").addEventListener("click", () => $("shortcutsDialog").showModal());
  $("shortcutsDialog").querySelector("[data-close]").addEventListener("click", () => $("shortcutsDialog").close());
  $("themeBtn").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try {
      localStorage.setItem("pp.theme", root.dataset.theme);
    } catch {
      // 保存できなくても切り替えは効く
    }
    paper.invalidate();
  });
  window.addEventListener("beforeunload", (e) => {
    if (plotter.running) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
}

boot().catch((error) => {
  console.error(error);
  toast(`起動に失敗しました: ${error.message || error}`, "error", 10000);
});
