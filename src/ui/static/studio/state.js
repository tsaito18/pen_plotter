// スタジオの状態と操作。画面（view.js）はここを読んで描くだけ。
// 用紙ビューア・プロッタ・エディタの実体もここが持ち、状態と同期させる。

import { MOD, STORAGE, api, debounce, el, toast } from "../common.js";
import { PaperView } from "../paper.js";
import { Plotter, estimateLineSeconds, normalizeGcode, parseGcode } from "../plotter.js";
import { createStore } from "../store.js";

// 仕上がりの雰囲気（筆跡 3 軸の組み合わせ）
export const PRESETS = [
  { id: "neat", label: "端正", values: { temperature: 0.1, messiness: 0.2, instance_variation: 0.05 }, path: "M4 11.5h36" },
  { id: "natural", label: "自然", values: { temperature: 0.2, messiness: 0.4, instance_variation: 0.1 }, path: "M4 12c6-2 10 1 16-1s10-1 16-1" },
  { id: "casual", label: "くだけた", values: { temperature: 0.5, messiness: 0.9, instance_variation: 0.3 }, path: "M4 13c4-5 8 3 12-1s6-6 10-2 6 3 10-1" },
  { id: "rough", label: "走り書き", values: { temperature: 0.9, messiness: 1.4, instance_variation: 0.55 }, path: "M4 15c3-9 6 7 9-2s4-8 7 0 4 7 7-3 5-5 9 1" },
];

export const COVERAGE_TIERS = [
  { key: "user_strokes", label: "あなたの筆跡", color: "var(--accent)" },
  { key: "composed", label: "部品から組み立て", color: "#7a9e7e" },
  { key: "ml_inference", label: "ML で変形", color: "var(--pencil)" },
  { key: "kanjivg", label: "KanjiVG 字形", color: "var(--text-2)" },
  { key: "geometric", label: "幾何字形", color: "#c2a46b" },
  { key: "missing_glyphs", label: "未収録", color: "var(--danger)" },
];

export const store = createStore({
  boot: null,
  text: "",
  settings: {},
  profile: null,
  japaneseOnly: false,
  seed: 0,
  model: null,
  models: [],
  draft: null, // /api/layout の結果
  render: null, // {pages, coverage, seed, elapsed, key, jobPages}
  rendering: false,
  progress: null, // 清書の進捗 {fraction, message}
  page: 0,
  tab: "style",
  view: "paper", // スマホでの表示（editor / paper / inspector）
  zoom: 100,
  source: "render", // 描くものの元: render | upload
  upload: null,
  selected: [], // 描くページ（添字）
  job: null,
  lastOutcome: null,
  knownPort: null,
  logs: [],
  paperChange: null, // {done, next}
  shortcutsOpen: false,
  animating: false,
  plotterRev: 0, // プロッタの状態が変わったら増える（描き直しの合図）
});

export const plotter = new Plotter();
let paper = null;
let editor = null;

// ------------------------------------------------------------------ 読み出し

export const randomSeed = () => Math.floor(Math.random() * 90000) + 10000;

const inputKey = (s) => JSON.stringify([s.text, s.settings, s.profile, s.japaneseOnly, s.seed, s.model]);
export const hasText = (s) => s.text.trim().length > 0;
export const isFresh = (s) => Boolean(s.render && s.render.key === inputKey(s));
/** プロッタ画面で G-code ファイルを選んでいるときは、それを用紙に出す。 */
export const showingUpload = (s) => s.source === "upload" && Boolean(s.upload) && s.tab === "plot";

export function jobPages(s) {
  if (s.source === "upload" && s.upload) return s.upload.pages;
  if (!s.render) return [];
  s.render.jobPages ||= s.render.pages.map((page, i) => ({ pageNo: i + 1, lines: page.gcode, view: page }));
  return s.render.jobPages;
}

export function displayPages(s) {
  if (plotter.running && s.job) return s.job.pages.map((p) => p.view);
  if (showingUpload(s)) return s.upload.pages.map((p) => p.view);
  if (isFresh(s)) return s.render.pages;
  return hasText(s) && s.draft ? s.draft.pages : [];
}

/** 自分の筆跡で書けなかった字（記号・英数字は除く）＝「書いて教える」候補。 */
export function teachable(coverage) {
  if (!coverage) return [];
  const chars = [...(coverage.ml_inference?.chars || ""), ...(coverage.kanjivg?.chars || ""), ...(coverage.missing_glyphs?.chars || "")];
  return [...new Set(chars)].filter((c) => /[぀-ヿ㐀-鿿]/.test(c));
}

export function modeOf(s) {
  const running = plotter.running && s.job;
  if (running) return ["plot", `描画中 · ${plotter.progress?.pageNo ?? 1}ページ目`];
  if (s.rendering) return ["busy", "清書中…"];
  if (showingUpload(s)) return ["ink", `G-code · ${s.upload.name}`];
  if (isFresh(s)) return ["ink", `清書 · No.${s.render.seed}`];
  if (hasText(s) && s.render) return ["stale", "下書き（清書は古い）"];
  if (hasText(s)) return ["draft", "下書き"];
  return ["empty", "白紙"];
}

export function flowStates(s) {
  const text = hasText(s);
  const fresh = isFresh(s);
  let plot = "todo";
  if (plotter.running) plot = "active";
  else if (fresh) plot = s.lastOutcome === "done" ? "done" : "active";
  return { write: text ? "done" : "active", render: fresh ? "done" : text ? "active" : "todo", plot };
}

export function estimateSeconds(pages) {
  return pages.reduce((n, p) => n + (p._est ??= estimateLineSeconds(p.lines).reduce((a, b) => a + b, 0)), 0) * 1.25;
}

// ------------------------------------------------------------------ 起動

function pick(obj, template) {
  const out = {};
  for (const k of Object.keys(template)) {
    if (obj && obj[k] !== undefined && typeof obj[k] === typeof template[k]) out[k] = obj[k];
  }
  return out;
}

export async function boot() {
  const boot = await api("/api/bootstrap");
  const defaults = boot.settings;
  const ids = boot.profiles.map((p) => p.id);
  const saved = STORAGE.get("profile");
  store.set({
    boot,
    text: STORAGE.get("text", "") || "",
    settings: { ...defaults, ...pick(STORAGE.get("settings", {}), defaults) },
    profile: ids.includes(saved) ? saved : ids[0] || null,
    japaneseOnly: Boolean(STORAGE.get("japaneseOnly", false)),
    seed: Number(STORAGE.get("seed")) || randomSeed(),
  });
  bindPlotter();
  loadModels();
  requestLayout();
}

// ------------------------------------------------------------------ 原稿

const saveText = debounce((t) => STORAGE.set("text", t), 400);

/** エディタ（textarea ＋ ハイライト層）を結び付ける。 */
export function attachEditor(instance) {
  editor = instance;
  editor.value = store.get().text;
  editor.textarea.addEventListener("input", () => setText(editor.value));
}

export function setText(text) {
  if (editor && editor.value !== text) editor.value = text; // 入力イベント経由で戻ってくる
  if (store.get().text === text) return;
  store.set({ text });
  saveText(text);
  requestLayout();
}

export const focusEditor = () => editor?.textarea.focus();
export const insertAtCursor = (text) => editor?.insert(text);

export function insertExample(ex) {
  const text = store.get().text;
  setText(text.trim() ? `${text.replace(/\s+$/, "")}\n\n${ex.text}` : ex.text);
  focusEditor();
  toast(`「${ex.label}」を挿入しました`, "ok", 2200);
}

// ------------------------------------------------------------------ 下書き（組版のみ・入力に追従）

let layoutAbort = null;
export const requestLayout = debounce(async () => {
  layoutAbort?.abort();
  layoutAbort = new AbortController();
  const { text, settings } = store.get();
  try {
    const res = await fetch("/api/layout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, settings }),
      signal: layoutAbort.signal,
    });
    const draft = await res.json();
    store.set((s) => {
      const pages = displayPages({ ...s, draft });
      return { draft, page: Math.min(s.page, Math.max(0, pages.length - 1)) };
    });
  } catch (error) {
    if (error.name !== "AbortError") console.warn(error);
  }
}, 180);

// ------------------------------------------------------------------ 設定

const persistSettings = debounce((v) => STORAGE.set("settings", v), 300);

export function setSettings(patch) {
  const settings = { ...store.get().settings, ...patch };
  store.set({ settings });
  persistSettings(settings);
  requestLayout();
}

export function resetSettings() {
  setSettings({ ...store.get().boot.settings });
  toast("設定を既定値に戻しました", "ok", 2000);
}

export function setProfile(profile) {
  STORAGE.set("profile", profile);
  store.set({ profile });
}

export function setJapaneseOnly(value) {
  STORAGE.set("japaneseOnly", value);
  store.set({ japaneseOnly: value });
}

export function setSeed(seed) {
  STORAGE.set("seed", seed);
  store.set({ seed });
}

export async function loadModels() {
  if (!store.get().boot?.collect) return;
  try {
    const { models, active } = await api("/api/models");
    store.set({ models, model: active });
  } catch {
    store.set({ models: [] });
  }
}

export async function useModel(name) {
  try {
    const { active } = await api("/api/models/use", { method: "POST", body: { name: name || null } });
    store.set({ model: active });
    toast(active ? `清書に「${active}」を使います` : "ML を使わずに清書します", "ok", 2400);
  } catch (error) {
    toast(error.message, "error");
  }
}

/** 清書でまだ自分の筆跡を使えなかった字を、筆跡画面の「依頼」に送る。 */
export async function teachMissing() {
  const s = store.get();
  const chars = teachable(s.render?.coverage).join("");
  if (!chars || !s.profile) return;
  try {
    await api("/api/collect/queue", { method: "POST", body: { profile: s.profile, chars } });
    toast(`${[...chars].length} 字を筆跡画面に送りました。iPad で「筆跡」を開くと書けます`, "ok", 6000, {
      label: "ここで開く",
      run: () => window.open("/collect?queue=1#write", "_blank"),
    });
  } catch (error) {
    toast(error.message, "error");
  }
}

// ------------------------------------------------------------------ 清書

export async function doRender({ reroll = false } = {}) {
  const s0 = store.get();
  if (s0.rendering || !hasText(s0) || plotter.running) return;
  if (s0.draft?.errors?.length) {
    toast("設定に問題があります。右のパネルを確認してください", "warn");
    return;
  }
  if (reroll) setSeed(randomSeed());
  const s = store.get();
  const key = inputKey(s);
  store.set({ rendering: true, progress: { fraction: 0, message: "準備中" } });
  try {
    const res = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: s.text, settings: s.settings, profile: s.profile, japanese_only: s.japaneseOnly, seed: s.seed }),
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
        if (event.type === "progress") store.set({ progress: { fraction: event.fraction, message: event.message } });
        else if (event.type === "error") throw new Error(event.message);
        else if (event.type === "result") result = event;
      }
    }
    if (!result) throw new Error("サーバーから結果が返りませんでした");
    const page = Math.min(store.get().page, result.pages.length - 1);
    store.set({
      render: { ...result, key },
      page,
      source: "render",
      selected: result.pages.map((_, i) => i),
      lastOutcome: null,
      animating: true,
    });
    paper?.setInk(result.pages[page], { animate: true });
    shownInk = result.pages[page];
    const strokes = result.pages.reduce((n, p) => n + p.strokes.length, 0);
    toast(`${result.pages.length}ページを清書しました — ${strokes.toLocaleString()}画・${result.elapsed}秒`, "ok");
  } catch (error) {
    toast(error.message || String(error), "error", 5200);
  } finally {
    store.set({ rendering: false, progress: null });
  }
}

export function downloadGcode() {
  const s = store.get();
  if (!isFresh(s)) {
    toast(`先に清書してください（${MOD}+Enter）`, "warn");
    return;
  }
  const pages = s.render.pages;
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

// ------------------------------------------------------------------ 用紙

let shownInk = null;

/** 用紙ビューアを結び付ける。 */
export function attachPaper(canvas) {
  const { boot } = store.get();
  paper = new PaperView(canvas, boot.paper);
  if (boot.paper.background) paper.setBackground("/api/paper");
  paper.on((type, detail) => {
    if (type === "zoom") store.set({ zoom: detail });
    if (type === "animationend") store.set({ animating: false });
  });
  const insets = () => {
    paper.insets = window.innerWidth <= 640 ? { top: 56, bottom: 72, left: 12, right: 12 } : { top: 64, bottom: 84, left: 28, right: 28 };
    if (paper.fitted) paper.fit();
  };
  insets();
  window.addEventListener("resize", insets);
  shownInk = null;
  syncPaper(store.get());
  return () => window.removeEventListener("resize", insets);
}

export const zoom = {
  in: () => paper?.zoomAt(1.25),
  out: () => paper?.zoomAt(0.8),
  fit: () => paper?.fit(),
  repaint: () => paper?.invalidate(),
};

/** 状態に合わせて用紙ビューアの表示を揃える（毎回呼んでよい。変化が無ければ何もしない）。 */
export function syncPaper(s) {
  if (!paper || !s.boot) return;
  const fresh = isFresh(s);
  const text = hasText(s);
  if (s.draft) paper.setRuled(s.draft.ruled);
  if (plotter.running && s.job) {
    const p = plotter.progress;
    const page = s.job.pages[p ? p.pageIndex : 0];
    if (shownInk !== page.view) {
      paper.setInk(page.view);
      shownInk = page.view;
    }
    paper.showDraft = false;
    paper.setInkAlpha(1);
    paper.setPlotProgress(p ? p.line : 0);
    return;
  }
  paper.setPlotProgress(null);
  const upload = showingUpload(s);
  const inkPage = upload ? s.upload.pages[Math.min(s.page, s.upload.pages.length - 1)].view : s.render?.pages[s.page] || null;
  if (shownInk !== inkPage) {
    if (!(fresh && paper.anim)) paper.setInk(inkPage);
    shownInk = inkPage;
  }
  paper.showDraft = !upload && !fresh && text;
  paper.setInkAlpha(upload || fresh ? 1 : 0.16);
  paper.setDraft(text ? s.draft?.pages[s.page] || null : null, s.settings.line_spacing);
}

export function goPage(i) {
  const s = store.get();
  const pages = displayPages(s);
  if (!pages.length || plotter.running) return;
  store.set({ page: Math.max(0, Math.min(pages.length - 1, i)) });
}

export const skipAnimation = () => paper?.skipAnimation();

// ------------------------------------------------------------------ パネル・表示

export function selectTab(tab) {
  store.set({ tab });
}

export function setView(view) {
  store.set({ view });
}

// ------------------------------------------------------------------ プロッタ

export function connectPlotter(opts = {}) {
  const { knownPort } = store.get();
  return plotter.connect(knownPort && !opts.anyPort ? { port: knownPort } : opts);
}

export async function openPlot() {
  selectTab("plot");
  if (window.innerWidth <= 1020) setView("inspector");
  if (plotter.status === "unsupported") {
    toast("このブラウザではプロッタに接続できません。パネルの案内を確認してください", "warn", 4200);
    return;
  }
  if (!plotter.connected) await connectPlotter();
  if (plotter.connected) requestAnimationFrame(() => document.getElementById("startBtn")?.focus());
}

export function startJob() {
  const s = store.get();
  const pages = jobPages(s).filter((_, i) => s.selected.includes(i));
  if (!pages.length) {
    toast("描くページを選んでください", "warn");
    return;
  }
  const name = s.source === "upload" ? s.upload.name : `清書 No.${s.render.seed}`;
  shownInk = null;
  store.set({ job: { name, pages }, lastOutcome: null });
  plotter.run({ name, pages });
}

export function togglePause() {
  if (plotter.status === "streaming") plotter.pause();
  else plotter.resume();
}

export function togglePage(i) {
  if (plotter.running) return;
  store.set((s) => ({ selected: s.selected.includes(i) ? s.selected.filter((x) => x !== i) : [...s.selected, i].sort() }));
}

export function useRender() {
  store.set((s) => ({ source: "render", selected: jobPages({ ...s, source: "render" }).map((_, i) => i) }));
}

export async function loadFiles(files) {
  if (!files.length) return;
  files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const pages = [];
  for (const [i, file] of files.entries()) {
    const lines = normalizeGcode(await file.text());
    const { strokes, spans } = parseGcode(lines);
    pages.push({ pageNo: i + 1, lines, name: file.name, view: { strokes, spans, gcode: lines } });
  }
  const upload = { name: files.length === 1 ? files[0].name : `${files.length} ファイル`, pages };
  store.set({ upload, source: "upload", page: 0, selected: pages.map((_, i) => i) });
  addLog({ message: `G-code を読み込みました: ${upload.name}`, level: "info", time: new Date() });
}

export function continueAfterPaper(go) {
  store.set({ paperChange: null });
  if (go) plotter.resume();
  else plotter.stop();
}

function addLog({ message, level, time }) {
  const stamp = time.toLocaleTimeString("ja-JP", { hour12: false });
  store.set((s) => ({ logs: [...s.logs.slice(-399), { message, level, time: stamp, id: `${time.getTime()}-${s.logs.length}` }] }));
  if (level === "error") toast(message, "error", 5200);
  if (level === "ok" && /接続しました/.test(message)) toast("プロッタに接続しました", "ok");
}

export async function copyLog() {
  const text = store
    .get()
    .logs.map((e) => `[${e.time}] ${e.message}`)
    .join("\n");
  try {
    await navigator.clipboard.writeText(text);
    toast("ログをコピーしました", "ok", 1800);
  } catch {
    toast("コピーできませんでした", "warn");
  }
}

const bump = () => store.set((s) => ({ plotterRev: s.plotterRev + 1 }));

function bindPlotter() {
  plotter.addEventListener("change", () => {
    if (plotter.port) store.set({ knownPort: plotter.port });
    bump();
  });
  // 用紙の進捗は毎回（軽い）、パネル類の描き直しは間引く（送信ループを止めない）
  let timer = 0;
  plotter.addEventListener("progress", () => {
    syncPaper(store.get());
    if (timer) return;
    timer = window.setTimeout(() => {
      timer = 0;
      bump();
    }, 150);
  });
  plotter.addEventListener("log", (e) => addLog(e.detail));
  plotter.addEventListener("paper", (e) => store.set({ paperChange: e.detail }));
  plotter.addEventListener("finish", (e) => {
    shownInk = null;
    store.set({ lastOutcome: e.detail.outcome, paperChange: null });
    if (e.detail.outcome === "done") toast("描き終わりました。お疲れさまでした", "ok", 5000);
  });
  // 以前に許可したプロッタがあれば、ポート選択なしのワンクリックで再接続できる
  plotter.knownPort().then((port) => store.set({ knownPort: port }));
  window.addEventListener("beforeunload", (e) => {
    if (plotter.running) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
}

export const thumbnail = (view, paperSize) => (view._thumb ||= PaperView.thumbnail(view, paperSize, 52));
