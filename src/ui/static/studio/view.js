// スタジオの画面（Preact + htm）。状態は state.js の store を読むだけで、操作は state.js の関数を呼ぶ。
// 用紙（canvas）とエディタ（textarea）は Preact が作った要素に命令的な部品を結び付ける。

import { Component } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { Brand, Icon, MOD, STORAGE, ThemeButton, formatDuration, html, isMac, toast, toastUndo, useDismiss } from "../common.js";
import { Editor } from "../editor.js";
import { unsupportedReason } from "../plotter.js";
import { useStore } from "../store.js";
import * as A from "./state.js";

const { store, plotter } = A;

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

const Kbd = ({ keys }) => html`<kbd class="kbd-hint">${keys.replace("⌘", isMac ? "⌘" : "Ctrl")}</kbd>`;

// ================================================================== 全体

export function App() {
  const s = useStore(store);
  useKeyboard();
  return html`
    <div class="app" id="app" data-view=${s.view}>
      <header class="topbar">
        <${Brand} current="studio" />
        <${Flow} s=${s} />
        <div class="topbar-end">
          <${MachinePill} />
          <${ThemeButton} onToggle=${A.zoom.repaint} />
          <button class="icon-btn" id="shortcutsBtn" type="button" aria-label="キーボードショートカット" data-tip="ショートカット ?"
            onClick=${() => store.set({ shortcutsOpen: true })}>
            <${Icon} name="i-keyboard" />
          </button>
        </div>
      </header>
      <main class="workspace">
        <${EditorPane} s=${s} />
        <${Stage} s=${s} />
        <${Inspector} s=${s} />
      </main>
      <${MobileNav} view=${s.view} />
    </div>
    <${PaperDialog} change=${s.paperChange} />
    <${ShortcutsDialog} open=${s.shortcutsOpen} />
  `;
}

function useKeyboard() {
  useEffect(() => {
    const onKey = (e) => {
      const s = store.get();
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key === "Enter") {
        e.preventDefault();
        A.doRender({ reroll: e.shiftKey });
        return;
      }
      if (mod && e.key.toLowerCase() === "s") {
        e.preventDefault();
        A.downloadGcode();
        return;
      }
      const typing = e.target.closest?.("input, textarea, select, [contenteditable]");
      if (typing || mod || e.altKey || s.paperChange || s.shortcutsOpen) return;
      if (e.key === "?") store.set({ shortcutsOpen: true });
      else if (e.key === " " && plotter.running) {
        e.preventDefault();
        A.togglePause();
      } else if (e.key === "Escape" && plotter.status === "streaming") plotter.pause();
      else if (e.key === "ArrowLeft") A.goPage(s.page - 1);
      else if (e.key === "ArrowRight") A.goPage(s.page + 1);
      else if (e.key === "+" || e.key === "=") A.zoom.in();
      else if (e.key === "-") A.zoom.out();
      else if (e.key === "0") A.zoom.fit();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);
}

// ================================================================== トップバー

const FLOW = [
  { step: "write", label: "書く" },
  { step: "render", label: "清書" },
  { step: "plot", label: "描く" },
];

function flowAction(step, s) {
  if (step === "write") {
    A.setView("editor");
    A.focusEditor();
  } else if (step === "render") {
    if (A.isFresh(s)) A.setView("paper");
    else A.doRender();
  } else {
    A.openPlot();
  }
}

function Flow({ s }) {
  const states = A.flowStates(s);
  return html`
    <nav class="flow" aria-label="作業の流れ">
      <ol>
        ${FLOW.map(
          (f, i) => html`
            <li key=${f.step}>
              <button class="flow-step" data-step=${f.step} data-state=${states[f.step]} aria-current=${String(states[f.step] === "active")}
                type="button" onClick=${() => flowAction(f.step, store.get())}>
                <span class="flow-num">${i + 1}</span><span class="flow-label">${f.label}</span>
              </button>
            </li>
          `,
        )}
      </ol>
    </nav>
  `;
}

const MachinePill = () => html`
  <button class="machine-pill" id="machinePill" type="button" data-status=${plotter.status} onClick=${A.openPlot}>
    <span class="machine-dot" aria-hidden="true"></span>
    <span class="machine-text"><span class="machine-name">xDraw A4</span><span class="machine-state" id="machineState">${STATUS_TEXT[plotter.status]}</span></span>
  </button>
`;

// ================================================================== 原稿

/** textarea とハイライト層。中身は Editor が持つので Preact は描き直さない。 */
class EditorBox extends Component {
  componentDidMount() {
    A.attachEditor(new Editor(this.base.querySelector("textarea"), this.base.querySelector("pre")));
  }

  shouldComponentUpdate() {
    return false;
  }

  render() {
    return html`
      <div class="editor" id="editorBox">
        <pre class="editor-layer" id="editorLayer" aria-hidden="true"></pre>
        <textarea class="editor-input" id="textInput" spellcheck="false" aria-label="原稿テキスト"
          placeholder=${"ここに原稿を書きます。\n\n# 見出し\n本文の中に $V = IR$ のような数式も。\n\n入力に合わせて右の用紙に下書きが現れます。"}></textarea>
      </div>
    `;
  }
}

function ExamplesMenu({ examples }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useDismiss(open, setOpen, ref);
  return html`
    <div class=${`menu${open ? " is-open" : ""}`} id="examplesMenu" ref=${ref}>
      <button class="tool-btn" type="button" aria-haspopup="true" aria-expanded=${String(open)} onClick=${() => setOpen(!open)}>
        <${Icon} name="i-text" />例文
      </button>
      <div class="menu-pop" role="menu">
        ${examples.map((ex) => {
          const first = ex.text.split("\n").find((l) => l.trim() && !l.startsWith("#")) || ex.text;
          return html`
            <button key=${ex.label} class="menu-item" type="button" role="menuitem"
              onClick=${() => {
                setOpen(false);
                A.insertExample(ex);
              }}>
              <b>${ex.label}</b><span>${first.slice(0, 40)}</span>
            </button>
          `;
        })}
      </div>
    </div>
  `;
}

function EditorPane({ s }) {
  const [syntaxOpen, setSyntaxOpen] = useState(false);
  const chars = s.text.replace(/\s/g, "").length;
  const pages = A.hasText(s) && s.draft ? s.draft.pages.length : 0;
  const clear = () => {
    if (!A.hasText(s)) return;
    const previous = s.text;
    A.setText("");
    toastUndo("原稿を消去しました", () => A.setText(previous));
  };
  return html`
    <section class="pane pane-editor" id="paneEditor" aria-labelledby="editorTitle">
      <div class="pane-head">
        <h2 class="pane-title" id="editorTitle"><span class="pane-kicker">01</span>原稿</h2>
        <div class="pane-tools">
          <${ExamplesMenu} examples=${s.boot.examples} />
          <button class="tool-btn" id="syntaxBtn" type="button" aria-expanded=${String(syntaxOpen)} aria-controls="syntaxPanel"
            onClick=${() => setSyntaxOpen(!syntaxOpen)}>
            <${Icon} name="i-book" />書式
          </button>
        </div>
      </div>

      <div class="syntax-panel" id="syntaxPanel" hidden=${!syntaxOpen}>
        <div class="syntax-grid" id="syntaxGrid">
          ${s.boot.syntax.map(
            (row) => html`
              <div class="syntax-name">${row.name}</div>
              <div class="syntax-ex">
                <code title="クリックで挿入" onClick=${() => !row.example.startsWith("（") && A.insertAtCursor(row.example)}>${row.example}</code>
                <small>${row.note}</small>
              </div>
            `,
          )}
        </div>
      </div>

      <${EditorBox} />

      <div class="pane-foot">
        <div class="stats" aria-live="polite">
          <span><b id="statChars">${chars.toLocaleString()}</b>字</span>
          <span><b id="statPages">${pages}</b>ページ</span>
        </div>
        <button class="ghost-btn" id="clearBtn" type="button" onClick=${clear}><${Icon} name="i-trash" />消去</button>
      </div>
    </section>
  `;
}

// ================================================================== 用紙

function PaperCanvas() {
  const ref = useRef(null);
  useEffect(() => A.attachPaper(ref.current), []);
  return html`<canvas class="stage-canvas" id="paperCanvas" role="img" aria-label="用紙のプレビュー" ref=${ref}></canvas>`;
}

function Stage({ s }) {
  const [dropping, setDropping] = useState(false);
  const running = plotter.running && s.job;
  const text = A.hasText(s);
  const [mode, label] = A.modeOf(s);
  const pages = A.displayPages(s);
  const current = running ? (plotter.progress?.pageIndex ?? 0) : s.page;
  useEffect(() => A.syncPaper(s));

  // G-code ファイルは用紙へドロップしても開ける
  const onDragOver = (e) => {
    if (![...e.dataTransfer.items].some((i) => i.kind === "file")) return;
    e.preventDefault();
    setDropping(true);
  };
  const onDragLeave = (e) => {
    if (!e.currentTarget.contains(e.relatedTarget)) setDropping(false);
  };
  const onDrop = (e) => {
    e.preventDefault();
    setDropping(false);
    const files = [...e.dataTransfer.files].filter((f) => /\.(gcode|nc|txt)$/i.test(f.name));
    if (!files.length) {
      toast("G-code（.gcode / .nc / .txt）をドロップしてください", "warn");
      return;
    }
    A.selectTab("plot");
    A.loadFiles(files);
  };

  return html`
    <section class=${`stage${dropping ? " is-dropping" : ""}`} id="stage" aria-label="用紙プレビュー"
      onDragOver=${onDragOver} onDragLeave=${onDragLeave} onDrop=${onDrop}>
      <${PaperCanvas} />

      <div class="stage-top">
        <div class="mode-chip" id="modeChip" data-mode=${mode}><span class="mode-dot"></span><span id="modeText">${label}</span></div>
        <div class="pages" id="pageTabs" role="tablist" aria-label="ページ">
          ${pages.length > 1 &&
          pages.map(
            (_, i) => html`
              <button key=${i} class="page-tab" type="button" role="tab" aria-selected=${String(i === current)} onClick=${() => A.goPage(i)}>
                ${i + 1}
              </button>
            `,
          )}
        </div>
        <div class="zoom" role="group" aria-label="表示倍率">
          <button class="icon-btn sm" id="zoomOut" type="button" aria-label="縮小" onClick=${A.zoom.out}><${Icon} name="i-minus" /></button>
          <button class="zoom-val" id="zoomFit" type="button" aria-label="用紙全体を表示" onClick=${A.zoom.fit}><span id="zoomText">${s.zoom}%</span></button>
          <button class="icon-btn sm" id="zoomIn" type="button" aria-label="拡大" onClick=${A.zoom.in}><${Icon} name="i-plus" /></button>
        </div>
      </div>

      <div class="stage-empty" id="stageEmpty" hidden=${text || running || A.showingUpload(s)}>
        <p class="stage-empty-title">書き始めると、ここに下書きが現れます</p>
        <p class="stage-empty-sub">
          左の原稿欄に入力するか、<button class="link-btn" id="emptyExample" type="button" onClick=${() => A.insertExample(s.boot.examples[0])}>例文を入れてみる</button>
        </p>
      </div>

      <div class="render-progress" id="renderProgress" hidden=${!s.progress}>
        <div class="rp-track"><div class="rp-fill" id="rpFill" style=${{ width: `${Math.round((s.progress?.fraction || 0) * 100)}%` }}></div></div>
        <div class="rp-text"><span id="rpText">${s.progress?.message || "清書しています"}</span><span id="rpPct">${Math.round((s.progress?.fraction || 0) * 100)}%</span></div>
      </div>

      <${Dock} s=${s} />
    </section>
  `;
}

function Dock({ s }) {
  const running = plotter.running;
  const fresh = A.isFresh(s);
  const text = A.hasText(s);
  const p = plotter.progress;
  const pct = running && p ? (p.sent / p.total) * 100 : 0;
  const paused = plotter.status === "paused" || plotter.status === "paper";
  const runText =
    plotter.status === "paused"
      ? "一時停止中"
      : plotter.status === "paper"
        ? "用紙交換待ち"
        : plotter.status === "pausing"
          ? "止まります…"
          : `描画中 ${Math.floor(pct)}%`;
  const ready = !running && fresh;
  return html`
    <div class="dock" id="dock">
      <button class=${`btn btn-primary btn-lg${text && !s.render && !s.rendering ? " is-emphasis" : ""}`} id="renderBtn" type="button"
        hidden=${running || fresh} disabled=${!text || s.rendering || Boolean(s.draft?.errors?.length)} onClick=${() => A.doRender()}>
        <${Icon} name="i-sparkle" /><span id="renderLabel">${s.rendering ? "清書しています…" : s.render ? "清書し直す" : "清書する"}</span><${Kbd} keys="⌘↵" />
      </button>
      <button class="btn btn-quiet" id="rerollBtn" type="button" hidden=${!ready} data-tip=${`同じ原稿を別の書きぶりで ⇧${MOD}↵`}
        onClick=${() => A.doRender({ reroll: true })}>
        <${Icon} name="i-shuffle" /><span>書き直す</span>
      </button>
      <button class="btn btn-quiet" id="downloadBtn" type="button" hidden=${!ready} data-tip="G-code を保存" onClick=${A.downloadGcode}>
        <${Icon} name="i-download" /><span>G-code</span>
      </button>
      <span class="dock-sep" id="dockSep" hidden=${!ready}></span>
      <button class="btn btn-ink btn-lg" id="plotBtn" type="button" hidden=${!ready} onClick=${A.openPlot}>
        <${Icon} name="i-pen" /><span id="plotLabel">${plotter.connected ? "プロッタで描く" : "接続して描く"}</span><${Icon} name="i-arrow" cls="icon icon-trail" />
      </button>
      <div class="dock-run" id="dockRun" hidden=${!running}>
        <div class="dock-run-meta"><span id="dockRunText">${runText}</span><span class="dock-run-eta" id="dockRunEta">${p ? `残り ${formatDuration(p.remaining)}` : ""}</span></div>
        <div class="dock-run-track"><div class="dock-run-fill" id="dockRunFill" style=${{ width: `${pct}%` }}></div></div>
      </div>
      <button class="btn btn-quiet" id="dockPause" type="button" hidden=${!running} onClick=${A.togglePause}>
        <${Icon} name=${paused ? "i-play" : "i-pause"} /><span>${paused ? "再開" : "一時停止"}</span>
      </button>
      <button class="btn btn-danger" id="dockEstop" type="button" hidden=${!running} data-tip="緊急停止（ソフトリセット）" onClick=${() => plotter.emergencyStop()}>
        <${Icon} name="i-octagon" /><span>緊急停止</span>
      </button>
    </div>
  `;
}

// ================================================================== インスペクタ

function Inspector({ s }) {
  const plot = s.tab === "plot";
  return html`
    <aside class="pane pane-inspector" id="paneInspector" aria-label="設定とプロッタ">
      <div class="seg" role="tablist" aria-label="パネル" data-active=${s.tab}>
        <button class="seg-btn" role="tab" id="tabStyle" aria-selected=${String(!plot)} aria-controls="panelStyle" type="button" onClick=${() => A.selectTab("style")}>
          <${Icon} name="i-sliders" />仕上がり
        </button>
        <button class="seg-btn" role="tab" id="tabPlot" aria-selected=${String(plot)} aria-controls="panelPlot" type="button" onClick=${() => A.selectTab("plot")}>
          <${Icon} name="i-pen" />プロッタ<span class="seg-badge" id="plotBadge" hidden=${!plotter.running}></span>
        </button>
        <span class="seg-glider" aria-hidden="true"></span>
      </div>
      <div class="panel" id="panelStyle" role="tabpanel" aria-labelledby="tabStyle" hidden=${plot}>
        <${StylePanel} s=${s} />
      </div>
      <div class="panel" id="panelPlot" role="tabpanel" aria-labelledby="tabPlot" hidden=${!plot}>
        <${PlotPanel} s=${s} />
      </div>
    </aside>
  `;
}

// ------------------------------------------------------------------ 仕上がり

function Coverage({ s }) {
  const coverage = s.render?.coverage;
  const total = coverage ? A.COVERAGE_TIERS.reduce((n, t) => n + (coverage[t.key]?.count || 0), 0) : 0;
  const teach = A.teachable(coverage);
  const missing = coverage?.missing_glyphs?.chars || "";
  const tiers = A.COVERAGE_TIERS.filter((t) => coverage?.[t.key]?.count);
  return html`
    <div class="card coverage" id="coverageCard" hidden=${!total}>
      <div class="card-head"><h3 class="card-title">字形の出どころ</h3><span class="card-meta" id="coverageMeta">${total ? `${total.toLocaleString()}字` : ""}</span></div>
      <div class="coverage-bar" id="coverageBar">
        ${tiers.map((t) => html`<span key=${t.key} title=${`${t.label}: ${coverage[t.key].chars.slice(0, 80)}`} style=${{ background: t.color, flexGrow: coverage[t.key].count }}></span>`)}
      </div>
      <ul class="coverage-legend" id="coverageLegend">
        ${tiers.map((t) => html`<li key=${t.key}><i style=${{ background: t.color }}></i><span>${t.label}</span><b>${coverage[t.key].count.toLocaleString()}</b></li>`)}
      </ul>
      <p class="coverage-missing" id="coverageMissing" hidden=${!missing}>${missing ? `字形が無く空白になった字: ${missing}` : ""}</p>
      <div class="teach" id="teach" hidden=${!teach.length || !s.boot.collect || !s.profile}>
        <p class="teach-text">まだあなたの筆跡がない字 <b id="teachCount">${teach.length}</b></p>
        <p class="teach-chars" id="teachChars">${teach.length > 40 ? `${teach.slice(0, 40).join("")}…` : teach.join("")}</p>
        <button class="btn btn-quiet small" id="teachBtn" type="button" onClick=${A.teachMissing}><${Icon} name="i-pen" />筆跡画面で書いて教える</button>
      </div>
    </div>
  `;
}

const Select = ({ id, value, onChange, children }) => html`
  <span class="select-wrap">
    <select id=${id} value=${value} onChange=${(e) => onChange(e.currentTarget.value)}>${children}</select><${Icon} name="i-chevron" />
  </span>
`;

function SeedBox({ seed }) {
  const [draft, setDraft] = useState(null);
  const commit = () => {
    const v = parseInt((draft ?? "").replace(/\D/g, ""), 10);
    if (Number.isFinite(v)) A.setSeed(v);
    setDraft(null);
  };
  return html`
    <span class="seed-box">
      <input class="seed-val" id="seedInput" type="text" inputmode="numeric" autocomplete="off" aria-label="書きぶり番号"
        value=${draft ?? String(seed)} onInput=${(e) => setDraft(e.currentTarget.value)} onChange=${commit}
        onKeyDown=${(e) => e.key === "Enter" && e.currentTarget.blur()} />
      <button class="icon-btn sm" id="seedDice" type="button" aria-label="別の番号にする" data-tip="別の番号" onClick=${() => A.setSeed(A.randomSeed())}>
        <${Icon} name="i-shuffle" />
      </button>
    </span>
  `;
}

const near = (a, b) => Math.abs(a - b) < 1e-6;

function WriterGroup({ s }) {
  const profiles = s.boot.profiles;
  return html`
    <div class="field-group">
      <div class="group-head"><h3 class="group-title">書き手</h3></div>
      <label class="select-field" id="profileField" hidden=${!profiles.length}>
        <span class="field-label">筆跡プロファイル</span>
        <${Select} id="profileSelect" value=${s.profile || ""} onChange=${A.setProfile}>
          ${profiles.map((p) => html`<option key=${p.id} value=${p.id}>${p.id}（${p.characters}字・${p.samples}サンプル）</option>`)}
        <//>
      </label>
      <label class="select-field" id="modelField" hidden=${!s.models.length}>
        <span class="field-label">清書のモデル</span>
        <${Select} id="modelSelect" value=${s.model || ""} onChange=${A.useModel}>
          ${s.models.map((m) => html`<option key=${m.name} value=${m.name}>${m.name}</option>`)}
          <option value="">ML を使わない</option>
        <//>
      </label>
      <div class="presets" id="presets" role="radiogroup" aria-label="仕上がりの雰囲気">
        ${A.PRESETS.map(
          (p) => html`
            <button key=${p.id} class="preset" type="button" role="radio" data-id=${p.id}
              aria-checked=${String(Object.entries(p.values).every(([k, v]) => near(s.settings[k], v)))} onClick=${() => A.setSettings(p.values)}>
              <span><svg viewBox="0 0 44 22"><path d=${p.path} /></svg></span><span>${p.label}</span>
            </button>
          `,
        )}
      </div>
      <label class="switch-row">
        <span><span class="field-label">日本語だけ描く</span><span class="field-info">英数字・数式・記号を省く</span></span>
        <input type="checkbox" class="switch" id="japaneseOnly" checked=${s.japaneseOnly} onChange=${(e) => A.setJapaneseOnly(e.currentTarget.checked)} />
      </label>
      <div class="seed-row">
        <label for="seedInput"><span class="field-label">書きぶり番号</span><span class="field-info">同じ番号なら、何度清書しても同じ字形</span></label>
        <${SeedBox} seed=${s.seed} />
      </div>
    </div>
  `;
}

function RangeControl({ c, value, initial }) {
  const id = `ctl-${c.field}`;
  const pct = ((value - c.minimum) / (c.maximum - c.minimum)) * 100;
  const digits = c.step >= 1 ? 0 : c.step >= 0.1 ? 1 : 2;
  return html`
    <div class="range-field">
      <div class="range-top">
        <label class="field-label" for=${id} title="ダブルクリックで既定値" onDblClick=${() => A.setSettings({ [c.field]: initial })}>
          ${c.label}${c.hardware_note && html`<span class="hw-tag">実機は0推奨</span>`}
        </label>
        <output class=${`range-value${Math.abs(value - initial) > 1e-9 ? " is-changed" : ""}`} for=${id}>${value.toFixed(digits)}${c.unit ? ` ${c.unit}` : ""}</output>
      </div>
      <input type="range" id=${id} min=${c.minimum} max=${c.maximum} step=${c.step} value=${value} style=${{ "--fill": `${pct}%` }}
        onInput=${(e) => A.setSettings({ [c.field]: parseFloat(e.currentTarget.value) })} />
      ${c.info && html`<span class="field-info">${c.info}</span>`}
    </div>
  `;
}

const ToggleControl = ({ c, value }) => html`
  <label class="switch-row">
    <span><span class="field-label">${c.label}</span>${c.info && html`<span class="field-info">${c.info}</span>`}</span>
    <input type="checkbox" class="switch" checked=${Boolean(value)} onChange=${(e) => A.setSettings({ [c.field]: e.currentTarget.checked })} />
  </label>
`;

function Section({ section, s, open, onToggle }) {
  const body = html`
    <div class="group-body">
      ${section.controls.map((c) =>
        c.kind === "toggle"
          ? html`<${ToggleControl} key=${c.field} c=${c} value=${s.settings[c.field]} />`
          : html`<${RangeControl} key=${c.field} c=${c} value=${s.settings[c.field]} initial=${s.boot.settings[c.field]} />`,
      )}
    </div>
  `;
  if (!section.collapsed) {
    return html`<div class="field-group"><div class="group-head"><h3 class="group-title">${section.title}</h3></div>${body}</div>`;
  }
  return html`
    <div class=${`field-group is-collapsible${open ? " is-open" : ""}`}>
      <button class="group-head" type="button" aria-expanded=${String(open)} onClick=${onToggle}>
        <h3 class="group-title">${section.title}</h3><span><${Icon} name="i-chevron" /></span>
      </button>
      ${body}
    </div>
  `;
}

function StylePanel({ s }) {
  const [open, setOpen] = useState(() => STORAGE.get("sectionsOpen", {}));
  const toggle = (id) => {
    const next = { ...open, [id]: !open[id] };
    setOpen(next);
    STORAGE.set("sectionsOpen", next);
  };
  const errors = s.draft?.errors || [];
  return html`
    <${Coverage} s=${s} />
    <${WriterGroup} s=${s} />
    <div id="sections">
      ${s.boot.sections.map((sec) => html`<${Section} key=${sec.id} section=${sec} s=${s} open=${Boolean(open[sec.id])} onToggle=${() => toggle(sec.id)} />`)}
    </div>
    <div class="validation" id="validation" role="alert" hidden=${!errors.length}>
      ${errors.length > 0 && html`<ul>${errors.map((e) => html`<li key=${e}>${e}</li>`)}</ul>`}
    </div>
    <button class="ghost-btn reset-btn" id="resetBtn" type="button" onClick=${A.resetSettings}>既定値に戻す</button>
  `;
}

// ------------------------------------------------------------------ プロッタ

const UNSUPPORTED = unsupportedReason();

function UnsupportedNote() {
  if (!UNSUPPORTED) return null;
  const port = location.port ? `:${location.port}` : "";
  return html`
    <div class="unsupported" id="unsupported">
      ${UNSUPPORTED === "browser"
        ? html`このブラウザは WebSerial に対応していません。プロッタをつないだ PC の <b>Chrome</b> か <b>Edge</b> で開いてください。G-code の保存はこのままできます。`
        : html`WebSerial は安全な接続でのみ使えます。プロッタをつないだ PC で <code>http://localhost${port}</code> を開いてください。`}
    </div>
  `;
}

function machineSub(s) {
  const status = plotter.status;
  if (plotter.connected) return plotter.needsHoming ? "接続済み — 描く前に自動で原点復帰します" : "接続済み — 原点復帰済み";
  if (status === "unsupported") return "このブラウザからは接続できません";
  if (status === "connecting") return "接続しています…";
  return s.knownPort ? "前回つないだプロッタにすぐ接続できます" : "USB でつないで「接続する」を押してください";
}

function MachineCard({ s }) {
  const status = plotter.status;
  const connected = plotter.connected;
  const idle = status === "idle";
  return html`
    <div class="card machine-card" id="machineCard" data-status=${status}>
      <div class="machine-hero">
        <div class="machine-figure" aria-hidden="true">
          <svg viewBox="0 0 120 80">
            <rect class="mf-bed" x="10" y="16" width="100" height="56" rx="4" />
            <rect class="mf-paper" x="34" y="22" width="52" height="46" rx="1" />
            <path class="mf-rail" d="M14 30h92" />
            <g class="mf-carriage"><rect x="52" y="24" width="16" height="12" rx="2" /><path d="M60 36v8" /></g>
            <path class="mf-ink" d="M44 52c6-4 10 2 16-2s8-6 14-2" />
          </svg>
        </div>
        <div class="machine-info">
          <div class="machine-title">xDraw A4</div>
          <div class="machine-sub" id="machineSub">${machineSub(s)}</div>
        </div>
      </div>
      <${UnsupportedNote} />
      <div class="btn-row">
        <button class="btn btn-primary grow" id="connectBtn" type="button" hidden=${connected}
          disabled=${status === "unsupported" || status === "connecting"} onClick=${() => A.connectPlotter()}>
          <${Icon} name="i-plug" /><span>${s.knownPort ? "再接続する" : "接続する"}</span>
        </button>
        <button class="btn btn-quiet grow" id="disconnectBtn" type="button" hidden=${!connected}
          disabled=${plotter.running || status === "busy"} onClick=${() => plotter.disconnect()}>
          <${Icon} name="i-unplug" /><span>切断</span>
        </button>
      </div>
      <button class="link-btn small" id="anyPortBtn" type="button" hidden=${connected || status === "unsupported"}
        onClick=${() => A.connectPlotter({ anyPort: true })}>
        一覧に出ないときは、すべてのポートから選ぶ
      </button>
      <div class="machine-controls" id="machineControls" hidden=${!connected}>
        <button class=${`mc-btn${idle && plotter.needsHoming ? " needs" : ""}`} id="homeBtn" type="button" disabled=${!idle} onClick=${() => plotter.home()}>
          <${Icon} name="i-home" /><span>原点復帰</span>
        </button>
        <button class="mc-btn" id="penUpBtn" type="button" disabled=${!idle} onClick=${() => plotter.penUp()}><${Icon} name="i-up" /><span>ペン上</span></button>
        <button class="mc-btn" id="penDownBtn" type="button" disabled=${!idle} onClick=${() => plotter.penDown()}><${Icon} name="i-down" /><span>ペン下</span></button>
      </div>
    </div>
  `;
}

function JobSource({ s, pages }) {
  if (s.source === "upload" && s.upload) {
    return html`<${Icon} name="i-file" /><span>${s.upload.name}<small>読み込んだ G-code・${pages.length}ページ</small></span>`;
  }
  if (s.render) {
    const stale = !A.isFresh(s);
    return html`<${Icon} name="i-sparkle" /><span>清書 No.${s.render.seed}<small>${pages.length}ページ${stale ? "・原稿の変更は未反映" : ""}</small></span>`;
  }
  return html`<${Icon} name="i-paper" /><span>まだありません<small>原稿を清書するか、G-code を開いてください</small></span>`;
}

function JobCard({ s, pages }) {
  const running = plotter.running;
  const p = plotter.progress;
  const selected = pages.filter((_, i) => s.selected.includes(i));
  const fromUpload = s.source === "upload" && s.upload;
  const showThumbs = pages.length > 1 || fromUpload;
  return html`
    <div class="card job-card" id="jobCard">
      <div class="card-head">
        <h3 class="card-title">描くもの</h3>
        <span class="card-meta" id="jobEstimate">${selected.length ? `約${formatDuration(A.estimateSeconds(selected))}` : ""}</span>
      </div>
      <div class="job-source" id="jobSource"><${JobSource} s=${s} pages=${pages} /></div>
      <div class="job-pages" id="jobPages" role="group" aria-label="描くページ">
        ${showThumbs &&
        pages.map((page, i) => {
          const jobIndex = running && s.job ? s.job.pages.indexOf(page) : -1;
          const cls = ["job-page", running && p && jobIndex === p.pageIndex && "is-current", running && p && jobIndex >= 0 && jobIndex < p.pageIndex && "is-done"];
          return html`
            <button key=${`${s.source}-${i}`} class=${cls.filter(Boolean).join(" ")} type="button" aria-pressed=${String(s.selected.includes(i))}
              title=${`${page.pageNo}ページ目を描く / 描かない`} onClick=${() => A.togglePage(i)}>
              <img src=${A.thumbnail(page.view, s.boot.paper)} alt="" /><span>${page.pageNo}</span>
            </button>
          `;
        })}
      </div>
      <div class="btn-row">
        <label class="btn btn-quiet small file-btn">
          <${Icon} name="i-upload" /><span>G-code を開く</span>
          <input type="file" id="fileInput" accept=".gcode,.nc,.txt" multiple hidden
            onChange=${(e) => {
              A.loadFiles([...e.currentTarget.files]);
              e.currentTarget.value = "";
            }} />
        </label>
        <button class="link-btn small" id="useRenderBtn" type="button" hidden=${!(fromUpload && s.render)} onClick=${A.useRender}>清書した原稿に戻す</button>
      </div>
    </div>
  `;
}

function runTitles(s, selected) {
  const status = plotter.status;
  const p = plotter.progress;
  if (!plotter.connected) {
    return ["プロッタ未接続", status === "unsupported" ? "G-code を保存して別の方法で送れます" : "上のボタンで接続してください"];
  }
  if (plotter.running) {
    const title =
      status === "paused"
        ? "一時停止中"
        : status === "pausing"
          ? "ペンを上げたら止まります"
          : status === "paper"
            ? "用紙の交換を待っています"
            : `描いています — ${p?.pageNo ?? 1}ページ目`;
    const sub = p ? `残り 約${formatDuration(p.remaining)}${p.pageCount > 1 ? ` · ${p.pageIndex + 1} / ${p.pageCount} 枚目` : ""}` : "";
    return [title, sub];
  }
  if (s.lastOutcome === "done") return ["描き終わりました", "別の原稿もそのまま続けて描けます"];
  if (!selected.length) return ["描くものがありません", "原稿を清書するか、G-code を開いてください"];
  return ["準備ができました", "用紙を左上の角に合わせてセットしてください"];
}

function RunCard({ s, pages }) {
  const status = plotter.status;
  const running = plotter.running;
  const p = plotter.progress;
  const selected = pages.filter((_, i) => s.selected.includes(i));
  const pct = running && p ? Math.floor((p.sent / p.total) * 100) : s.lastOutcome === "done" ? 100 : 0;
  const [title, sub] = runTitles(s, selected);
  return html`
    <div class="card run-card" id="runCard">
      <div class="run-meter">
        <svg class="ring" viewBox="0 0 100 100" aria-hidden="true">
          <circle class="ring-bg" cx="50" cy="50" r="44" />
          <circle class=${`ring-fg${pct > 0 ? " has-value" : ""}`} id="ringFg" cx="50" cy="50" r="44" pathLength="100" style=${{ strokeDasharray: `${pct} 100` }} />
        </svg>
        <div class="ring-label"><span class="ring-pct" id="ringPct">${pct}</span><span class="ring-unit">%</span></div>
      </div>
      <div class="run-info">
        <div class="run-title" id="runTitle">${title}</div>
        <div class="run-sub" id="runSub">${sub}</div>
        <div class="run-sub mono" id="runLines">${running && p ? `${p.sent.toLocaleString()} / ${p.total.toLocaleString()} 行` : ""}</div>
      </div>
      <div class="run-actions">
        <button class="btn btn-ink btn-lg grow" id="startBtn" type="button" hidden=${running} disabled=${!(status === "idle" && selected.length)} onClick=${A.startJob}>
          <${Icon} name="i-play" /><span>描画を開始</span>
        </button>
        <button class="btn btn-quiet grow" id="pauseBtn" type="button" hidden=${status !== "streaming"} onClick=${() => plotter.pause()}>
          <${Icon} name="i-pause" /><span>一時停止</span>
        </button>
        <button class="btn btn-quiet grow" id="resumeBtn" type="button" hidden=${!["paused", "pausing", "paper"].includes(status)} onClick=${() => plotter.resume()}>
          <${Icon} name="i-play" /><span>再開</span>
        </button>
        <button class="btn btn-quiet" id="stopBtn" type="button" hidden=${!running} onClick=${() => plotter.stop()}>
          <${Icon} name="i-stop" /><span>停止</span>
        </button>
      </div>
      <button class="btn btn-danger wide" id="estopBtn" type="button" hidden=${!plotter.connected} disabled=${!running && status !== "busy"}
        onClick=${() => plotter.emergencyStop()}>
        <${Icon} name="i-octagon" /><span>緊急停止</span>
      </button>
    </div>
  `;
}

function LogCard({ logs }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logs.length]);
  return html`
    <details class="card log-card" id="logCard">
      <summary><${Icon} name="i-terminal" /><span>通信ログ</span><span class="log-count" id="logCount">${logs.length}</span><${Icon} name="i-chevron" cls="icon chev" /></summary>
      <div class="log" id="log" role="log" aria-live="polite" ref=${ref}>
        ${logs.map((e) => html`<div key=${e.id} class=${`log-line ${e.level}`}><time>${e.time}</time><span>${e.message}</span></div>`)}
      </div>
      <button class="ghost-btn small" id="copyLog" type="button" onClick=${A.copyLog}><${Icon} name="i-copy" />コピー</button>
    </details>
  `;
}

function PlotPanel({ s }) {
  const pages = A.jobPages(s);
  return html`
    <${MachineCard} s=${s} />
    <${JobCard} s=${s} pages=${pages} />
    <${RunCard} s=${s} pages=${pages} />
    <${LogCard} logs=${s.logs} />
  `;
}

// ================================================================== スマホの切り替え・ダイアログ

const VIEWS = [
  { view: "editor", icon: "i-text", label: "原稿" },
  { view: "paper", icon: "i-paper", label: "用紙" },
  { view: "inspector", icon: "i-sliders", label: "設定" },
];

const MobileNav = ({ view }) => html`
  <nav class="mobile-nav" aria-label="表示切替">
    ${VIEWS.map(
      (v) => html`
        <button key=${v.view} type="button" data-view=${v.view} aria-current=${String(view === v.view)} onClick=${() => A.setView(v.view)}>
          <${Icon} name=${v.icon} />${v.label}
        </button>
      `,
    )}
    <a href="/collect"><${Icon} name="i-pen" />筆跡</a>
  </nav>
`;

/** <dialog> を open に合わせて開閉する。 */
function useModal(open, focus) {
  const ref = useRef(null);
  useEffect(() => {
    const dialog = ref.current;
    if (open && !dialog.open) {
      dialog.showModal();
      if (focus) dialog.querySelector(focus)?.focus();
    } else if (!open && dialog.open) dialog.close();
  }, [open]);
  return ref;
}

function PaperDialog({ change }) {
  const ref = useModal(Boolean(change), "#paperContinue");
  const done = change?.done ?? 1;
  const next = change?.next ?? 2;
  return html`
    <dialog class="modal" id="paperDialog" aria-labelledby="paperDialogTitle" ref=${ref} onCancel=${(e) => e.preventDefault()}>
      <div class="modal-body">
        <div class="paper-swap" aria-hidden="true">
          <div class="sheet sheet-out"><span class="sheet-lines"></span><span class="sheet-check"><${Icon} name="i-check" /></span><b id="paperDoneNo">${done}</b></div>
          <${Icon} name="i-arrow" cls="icon swap-arrow" />
          <div class="sheet sheet-in"><b id="paperNextNo">${next}</b></div>
        </div>
        <h2 class="modal-title" id="paperDialogTitle">用紙を交換してください</h2>
        <ol class="steps">
          <li>描き終えた <b id="paperDoneText">${done} ページ目</b> を外す</li>
          <li>新しい用紙を、前と同じ位置（左上の角）に合わせて置く</li>
          <li>「続ける」で <b id="paperNextText">${next} ページ目</b> を描き始めます</li>
        </ol>
        <div class="modal-actions">
          <button class="btn btn-quiet" id="paperStop" type="button" onClick=${() => A.continueAfterPaper(false)}>ここで終える</button>
          <button class="btn btn-ink btn-lg" id="paperContinue" type="button" autofocus onClick=${() => A.continueAfterPaper(true)}>
            <${Icon} name="i-play" />続ける<kbd class="kbd-hint">↵</kbd>
          </button>
        </div>
      </div>
    </dialog>
  `;
}

const SHORTCUTS = [
  [["⌘", "↵"], "清書する"],
  [["⇧", "⌘", "↵"], "別の書きぶりで書き直す"],
  [["⌘", "S"], "G-code を保存"],
  [["Space"], "描画の一時停止 / 再開"],
  [["←", "→"], "ページを移動"],
  [["+", "−", "0"], "拡大・縮小・全体表示"],
];

function ShortcutsDialog({ open }) {
  const ref = useModal(open);
  const close = () => store.set({ shortcutsOpen: false });
  const key = (k) => (k === "⌘" ? MOD : k);
  return html`
    <dialog class="modal" id="shortcutsDialog" aria-labelledby="shortcutsTitle" ref=${ref} onClose=${close}>
      <div class="modal-body">
        <h2 class="modal-title" id="shortcutsTitle">キーボードショートカット</h2>
        <dl class="shortcuts">
          ${SHORTCUTS.map(([keys, label]) => html`<div key=${label}><dt>${keys.map((k) => html`<kbd>${key(k)}</kbd>`)}</dt><dd>${label}</dd></div>`)}
          <div><dt>ドラッグ / <kbd>${MOD}</kbd>+ホイール</dt><dd>用紙を動かす / 拡大縮小</dd></div>
          <div><dt><kbd>?</kbd></dt><dd>この一覧</dd></div>
        </dl>
        ${isMac && html`<p class="modal-note">Windows では <kbd>⌘</kbd> の代わりに <kbd>Ctrl</kbd>。</p>`}
        <div class="modal-actions"><button class="btn btn-quiet" type="button" onClick=${close}>閉じる</button></div>
      </div>
    </dialog>
  `;
}
