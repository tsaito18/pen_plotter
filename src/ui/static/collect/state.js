// 筆跡画面の状態と操作。画面（view.js）はここを読んで描くだけ。
// 書き込みパッドと書き順アニメーションの実体もここが持つ。

import { STORAGE, api, toast, toastUndo } from "../common.js";
import { createStore } from "../store.js";

export const MODES = ["write", "review", "train"];

export const store = createStore({
  profiles: [],
  profile: null,
  mode: "write",
  current: null, // /api/collect/next の結果（＋mode・next）
  forced: null, // 見直しから「この字を書く」
  queue: [], // スタジオからの依頼
  queueActive: false,
  queueIndex: 0,
  session: 0,
  tally: null, // {completed, total}
  pop: false, // 保存直後、増えた点をはずませる
  padCount: 0,
  ghost: Boolean(STORAGE.get("ghost", false)),
  touch: Boolean(STORAGE.get("touch", false)),
  glyphRev: 0, // お手本を読み込んだら増える
  stats: null,
  issues: null,
  filter: "all",
  query: "",
  drawer: null, // {char, info, samples, preview}
  training: null,
  models: { models: [], active: null },
});

let pad = null;
let guide = null;
const glyphs = new Map();

export const glyph = (char) => glyphs.get(char);

// ------------------------------------------------------------------ 起動

export async function boot() {
  const params = new URLSearchParams(location.search);
  await loadProfiles(params.get("profile"));
  store.set({
    forced: params.get("char") ? params.get("char").slice(0, 1) : null,
    queueActive: Boolean(params.get("queue")),
  });
  const mode = location.hash.slice(1);
  setMode(MODES.includes(mode) ? mode : "write", { push: false });
  window.addEventListener("hashchange", () => {
    const m = location.hash.slice(1);
    if (MODES.includes(m) && m !== store.get().mode) setMode(m, { push: false });
  });
  window.setInterval(pollQueue, 6000);
  window.setInterval(() => store.get().mode === "train" && loadTraining(), 1500);
}

/** 書き込みパッドを結び付ける。 */
export function attachPad(instance) {
  pad = instance;
  pad.allowTouch = store.get().touch;
  pad.addEventListener("change", () => {
    pad.root.classList.toggle("has-ink", pad.count > 0);
    store.set({ padCount: pad.count });
  });
  pad.addEventListener("strokestart", () => pad.root.classList.add("has-ink"));
  pad.addEventListener("gestureundo", () => toast("1 画戻しました", "info", 1200));
  applyGhost();
}

export function attachGuide(instance) {
  guide = instance;
  const char = store.get().current?.char;
  if (char && glyphs.has(char)) guide.set(glyphs.get(char));
}

export const replayGuide = () => guide?.play();
export const redrawGuide = () => guide?.draw(1);
export const undoStroke = () => pad?.undo();
export const clearPad = () => pad?.clear();

// ------------------------------------------------------------------ 書き手

async function loadProfiles(wanted) {
  const boot = await api("/api/bootstrap");
  const saved = wanted || STORAGE.get("profile");
  const ids = boot.profiles.map((p) => p.id);
  store.set({ profiles: boot.profiles, profile: ids.includes(saved) ? saved : ids[0] || null });
}

export async function selectProfile(id) {
  STORAGE.set("profile", id);
  store.set({ profile: id, forced: null, queueActive: false, queue: [], current: null });
  toast(`書き手を「${id}」にしました`, "ok", 1800);
  await refreshMode();
}

export async function createProfile(id) {
  if (!id) return false;
  try {
    await api("/api/profiles", { method: "POST", body: { id } });
    await loadProfiles(id);
    await selectProfile(id);
    return true;
  } catch (error) {
    toast(`追加できません: ${error.message}`, "error");
    return false;
  }
}

function needProfile() {
  if (store.get().profile) return true;
  toast("まず書き手を追加してください", "warn");
  setMode("write");
  window.requestAnimationFrame(() => document.getElementById("welcomeInput")?.focus());
  return false;
}

// ------------------------------------------------------------------ モード

export function setMode(mode, { push = true } = {}) {
  store.set({ mode });
  if (push) history.replaceState(null, "", `${location.pathname}${location.search}#${mode}`);
  refreshMode();
}

async function refreshMode() {
  const { mode } = store.get();
  if (mode === "write") {
    await pollQueue();
    await loadNext();
  } else if (mode === "review") {
    await loadReview();
  } else {
    await Promise.all([loadTraining(), loadModels()]);
  }
}

// ------------------------------------------------------------------ 集める

async function glyphOf(char) {
  if (!glyphs.has(char)) {
    const data = await api("/api/glyph", { params: { char } }).catch(() => ({ strokes: [] }));
    glyphs.set(char, data.strokes || []);
    store.set((s) => ({ glyphRev: s.glyphRev + 1 }));
  }
  return glyphs.get(char);
}

async function loadNext() {
  const s = store.get();
  if (!s.profile) {
    store.set({ current: null });
    return;
  }
  const params = { profile: s.profile };
  const queue = s.queueActive ? s.queue.slice(s.queueIndex) : [];
  let mode = "normal";
  let next = null;
  let info;
  if (s.forced) {
    info = await api("/api/collect/next", { params: { ...params, char: s.forced } });
    mode = "forced";
  } else if (queue.length) {
    info = await api("/api/collect/next", { params: { ...params, char: queue[0] } });
    mode = "queue";
    next = queue[1] || null;
  } else {
    if (s.queueActive) {
      store.set({ queueActive: false });
      toast("依頼の字をすべて書きました", "ok", 3000);
    }
    info = await api("/api/collect/next", { params: { ...params, prefer: s.current?.next || undefined } });
    next = info.next;
    store.set({ tally: { completed: info.completed, total: info.total } });
  }
  const current = { ...info, mode, next };
  store.set({ current, pop: false });
  if (!current.char) {
    guide?.set([]);
    return;
  }
  const strokes = await glyphOf(current.char);
  if (store.get().current !== current) return;
  guide?.set(strokes);
  applyGhost();
}

function applyGhost() {
  const { current, ghost } = store.get();
  pad?.setGhost(ghost && current?.char ? glyphs.get(current.char) : null);
}

export function setGhost(value) {
  STORAGE.set("ghost", value);
  store.set({ ghost: value });
  applyGhost();
}

export function setTouch(value) {
  STORAGE.set("touch", value);
  store.set({ touch: value });
  if (pad) pad.allowTouch = value;
}

let saving = false;

export async function save() {
  const s = store.get();
  if (saving || !pad || pad.empty || !s.current?.char || !needProfile()) return;
  saving = true;
  const info = s.current;
  try {
    const result = await api("/api/collect/samples", {
      method: "POST",
      body: { profile: s.profile, character: info.char, strokes: pad.data() },
    });
    flyAway();
    store.set((st) => ({ current: { ...info, count: result.count }, pop: true, session: st.session + 1 }));
    toast(`「${info.char}」を保存しました`, "ok", 3200, {
      label: "取り消す",
      run: async () => {
        // 取り消すのは「この保存」（後から別の字を保存していても間違えない）
        await api("/api/collect/samples", {
          method: "DELETE",
          params: { profile: s.profile, char: info.char, file: result.filename },
        }).catch(() => null);
        store.set((st) => ({ session: Math.max(0, st.session - 1) }));
        toast(`「${info.char}」を取り消しました`, "info", 1600);
        loadNext();
      },
    });
    if (info.mode === "forced") store.set({ forced: null });
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
  const frame = document.getElementById("fly");
  const img = document.createElement("img");
  img.src = pad.snapshot();
  img.alt = "";
  frame.append(img);
  pad.clear();
  pad.root.classList.add("flash");
  window.setTimeout(() => pad.root.classList.remove("flash"), 600);
  const from = frame.getBoundingClientRect();
  const dots = document.getElementById("subjectDots").getBoundingClientRect();
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

export function skip() {
  const s = store.get();
  if (s.forced) store.set({ forced: null });
  else if (s.queueActive && s.queue.length) store.set({ queueIndex: (s.queueIndex + 1) % s.queue.length });
  pad?.clear();
  loadNext();
}

async function pollQueue() {
  const { profile, queueIndex } = store.get();
  if (!profile) return;
  try {
    const { chars } = await api("/api/collect/queue", { params: { profile } });
    store.set({ queue: chars, queueIndex: queueIndex >= chars.length ? 0 : queueIndex });
  } catch {
    // 一時的な失敗は次の問い合わせで回復する
  }
}

export function startQueue() {
  store.set({ queueActive: true, queueIndex: 0, forced: null });
  loadNext();
}

export function stopQueue() {
  store.set({ queueActive: false });
  loadNext();
}

export async function clearQueue() {
  await api("/api/collect/queue", { method: "POST", body: { profile: store.get().profile, chars: "" } });
  store.set({ queue: [], queueActive: false });
  loadNext();
}

// ------------------------------------------------------------------ 見直す

async function loadReview() {
  const { profile } = store.get();
  if (!profile) return;
  const [stats, issues] = await Promise.all([
    api("/api/collect/stats", { params: { profile } }),
    api("/api/collect/issues", { params: { profile } }),
  ]);
  store.set({ stats, issues });
}

export function issueChars(issues) {
  const set = new Set();
  for (const a of issues?.anomalies || []) set.add(a.character);
  for (const m of issues?.mismatches || []) set.add(m.character);
  return set;
}

export const issueTotal = (issues) => (issues ? issues.anomalies.length + issues.mismatches.length : 0);

export const setFilter = (filter) => store.set({ filter });

export function search(query) {
  const c = query.trim();
  store.set({ query: c });
  if (c) openDrawer(c);
}

export async function trash(char, filename) {
  const { profile } = store.get();
  await api("/api/collect/samples", { method: "DELETE", params: { profile, char, file: filename } });
  toastUndo(`「${char}」のサンプルを消しました`, async () => {
    await api("/api/collect/samples/restore", { method: "POST", body: { profile, char, files: [filename] } });
    refreshAfterEdit(char);
  });
  window.setTimeout(() => refreshAfterEdit(char), 260);
}

/** 「問題ない」「どれも正しい」— 異常検出の対象から外す。 */
export async function dismissIssue(char, filenames, key) {
  const { profile } = store.get();
  for (const file of filenames) {
    await api("/api/collect/samples/metadata", { method: "POST", body: { profile, char, file, key, value: true } });
  }
  window.setTimeout(loadReview, 260);
}

function refreshAfterEdit(char) {
  loadReview();
  if (store.get().drawer?.char === char) openDrawer(char);
}

export async function openDrawer(char) {
  const { profile } = store.get();
  if (!profile) return;
  store.set({ drawer: { char, info: null, samples: null, preview: null } });
  const patch = (p) => store.set((s) => (s.drawer?.char === char ? { drawer: { ...s.drawer, ...p } } : null));
  try {
    const [info, samples] = await Promise.all([
      api("/api/collect/next", { params: { profile, char } }),
      api("/api/collect/samples", { params: { profile, char } }),
    ]);
    patch({ info, samples });
  } catch (error) {
    toast(error.message, "error");
  }
  try {
    patch({ preview: await api("/api/collect/preview", { params: { profile, char, n: 3 } }) });
  } catch {
    patch({ preview: { variants: [], source: null } });
  }
}

export const closeDrawer = () => store.set({ drawer: null, query: "" });

export function writeDrawerChar() {
  const char = store.get().drawer?.char;
  store.set({ forced: char, drawer: null, query: "" });
  setMode("write");
}

export async function deleteAll() {
  const { drawer, profile } = store.get();
  if (!drawer) return;
  const { char } = drawer;
  const { files } = await api("/api/collect/samples", { method: "DELETE", params: { profile, char } });
  toastUndo(`「${char}」のサンプルを ${files.length} 件消しました`, async () => {
    await api("/api/collect/samples/restore", { method: "POST", body: { profile, char, files } });
    refreshAfterEdit(char);
  });
  refreshAfterEdit(char);
}

// ------------------------------------------------------------------ 学習

export async function startTraining(form) {
  if (!needProfile()) return;
  const num = (v) => (String(v ?? "").trim() ? Number(v) : null);
  const body = {
    profile: store.get().profile,
    kind: form.kind,
    dataset: form.dataset,
    epochs: num(form.epochs),
    batch_size: num(form.batch),
    learning_rate: num(form.lr),
    deformer_type: form.deformer,
    device: form.device.trim() || null,
    output: form.output.trim() || null,
    checkpoint: form.kind === "finetune" ? form.base || null : null,
  };
  try {
    setTraining(await api("/api/training/start", { method: "POST", body }));
    toast("学習を始めました", "ok", 2000);
  } catch (error) {
    toast(`始められません: ${error.message}`, "error", 5000);
  }
}

export async function cancelTraining() {
  setTraining(await api("/api/training/cancel", { method: "POST" }));
}

async function loadTraining() {
  try {
    setTraining(await api("/api/training"));
  } catch {
    // サーバー再起動中など
  }
}

function setTraining(training) {
  const prev = store.get().training?.state;
  store.set({ training });
  if (prev === "running" && training.state !== "running") {
    loadModels();
    if (training.state === "succeeded") toast("学習が終わりました。「このモデルを使う」でスタジオに反映できます", "ok", 6000);
    if (training.state === "failed") toast(`学習に失敗しました: ${training.error}`, "error", 6000);
  }
}

async function loadModels() {
  try {
    store.set({ models: await api("/api/models") });
  } catch {
    // 一覧が取れなくても学習は続けられる
  }
}

export async function useModel(name) {
  try {
    await api("/api/models/use", { method: "POST", body: { name } });
    toast(name ? `スタジオの清書に「${name}」を使います` : "ML を使わずに清書します", "ok", 3000);
    await loadModels();
  } catch (error) {
    toast(error.message, "error");
  }
}
