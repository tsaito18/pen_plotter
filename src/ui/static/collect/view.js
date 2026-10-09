// 筆跡の画面（Preact + htm）。状態は state.js の store を読むだけで、操作は state.js の関数を呼ぶ。
// 書き込みパッド・書き順・サムネイルの canvas は Preact が作った要素に命令的な部品を結び付ける。

import { Component } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { Brand, Icon, ThemeButton, formatDuration, html, useDismiss } from "../common.js";
import { StrokeOrder, drawStrokes, unflatten } from "../glyph.js";
import { LOGICAL, WritingPad } from "../pad.js";
import { useStore } from "../store.js";
import * as A from "./state.js";

const { store } = A;

// ================================================================== 全体

export function App() {
  const s = useStore(store);
  useKeyboard();
  const issues = A.issueTotal(s.issues);
  return html`
    <div class="app collect" id="app" data-mode=${s.mode}>
      <header class="topbar">
        <${Brand} current="collect" />
        <div class="seg modes" role="tablist" aria-label="筆跡の作業" data-active=${s.mode}>
          ${[
            ["write", "i-pen", "集める"],
            ["review", "i-grid", "見直す"],
            ["train", "i-brain", "学習"],
          ].map(
            ([mode, name, label]) => html`
              <button key=${mode} class="seg-btn" role="tab" data-mode=${mode} aria-selected=${String(s.mode === mode)} type="button" onClick=${() => A.setMode(mode)}>
                <${Icon} name=${name} />${label}${mode === "review" && html`<span class="seg-count" id="issueCount" hidden=${!issues}>${issues}</span>`}
              </button>
            `,
          )}
          <span class="seg-glider" aria-hidden="true"></span>
        </div>
        <div class="topbar-end">
          <${ProfileMenu} profiles=${s.profiles} profile=${s.profile} />
          <${ThemeButton} onToggle=${A.redrawGuide} />
        </div>
      </header>
      <main class="views">
        <${WriteView} s=${s} />
        <${ReviewView} s=${s} />
        <${TrainView} s=${s} />
      </main>
    </div>
  `;
}

function useKeyboard() {
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest?.("input, textarea, select")) return;
      const s = store.get();
      if (e.key === "Escape" && s.drawer) {
        A.closeDrawer();
        return;
      }
      if (s.mode !== "write") return;
      const mod = e.metaKey || e.ctrlKey;
      if (e.key === "Enter") {
        e.preventDefault();
        A.save();
      } else if ((mod && e.key.toLowerCase() === "z") || e.key === "Backspace") {
        e.preventDefault();
        A.undoStroke();
      } else if (e.key === "Escape") {
        A.clearPad();
      } else if (e.key === "ArrowRight") {
        A.skip();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);
}

function ProfileMenu({ profiles, profile }) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const ref = useRef(null);
  useDismiss(open, setOpen, ref);
  const toggle = () => {
    setOpen(!open);
    if (!open) window.requestAnimationFrame(() => document.getElementById("profileInput")?.focus({ preventScroll: true }));
  };
  const submit = async (e) => {
    e.preventDefault();
    if (await A.createProfile(draft.trim())) {
      setDraft("");
      setOpen(false);
    }
  };
  return html`
    <div class=${`menu profile-menu${open ? " is-open" : ""}`} id="profileMenu" ref=${ref}>
      <button class="profile-pill" type="button" aria-haspopup="true" aria-expanded=${String(open)} onClick=${toggle}>
        <span class="avatar" id="profileAvatar">${profile ? profile[0].toUpperCase() : "+"}</span>
        <span class="profile-text"><span class="profile-label">書き手</span><span class="profile-name" id="profileName">${profile || "未設定"}</span></span>
        <${Icon} name="i-chevron" />
      </button>
      <div class="menu-pop" role="menu">
        <div class="menu-head">書き手を選ぶ</div>
        <div id="profileList">
          ${profiles.map(
            (p) => html`
              <button key=${p.id} class="profile-item" type="button" role="menuitemradio" aria-checked=${String(p.id === profile)}
                onClick=${() => {
                  setOpen(false);
                  A.selectProfile(p.id);
                }}>
                <span class="avatar">${p.id[0].toUpperCase()}</span><b>${p.id}</b><span>${p.characters}字</span>
              </button>
            `,
          )}
          ${!profiles.length && html`<p class="menu-head">まだ誰もいません。下で追加してください</p>`}
        </div>
        <form class="profile-new" id="profileForm" autocomplete="off" onSubmit=${submit}>
          <input id="profileInput" placeholder="新しい人の ID（英数字）" pattern="[A-Za-z0-9_\\-]+" aria-label="新しいプロファイル ID"
            value=${draft} onInput=${(e) => setDraft(e.currentTarget.value)} />
          <button class="btn btn-quiet small" type="submit"><${Icon} name="i-plus" />追加</button>
        </form>
      </div>
    </div>
  `;
}

// ================================================================== 集める

function Request({ s }) {
  const active = s.queueActive;
  return html`
    <div class=${`request${active ? " is-active" : ""}`} id="request" hidden=${!s.queue.length}>
      <div class="request-text">
        <b id="requestCount">${s.queue.length}</b> <span id="requestLead">${active ? "字、スタジオからの依頼を書いています" : "字、スタジオから「書いて教えて」"}</span>
      </div>
      <div class="request-chars" id="requestChars">${s.queue.join("")}</div>
      <div class="request-actions">
        <button class="btn btn-primary small" id="requestStart" type="button" hidden=${active} onClick=${A.startQueue}>この字から書く</button>
        <button class="btn btn-quiet small" id="requestStop" type="button" hidden=${!active} onClick=${A.stopQueue}>ふだんの順に戻る</button>
        <button class="link-btn small" id="requestClear" type="button" onClick=${A.clearQueue}>取り下げる</button>
      </div>
    </div>
  `;
}

/** 書き順アニメーションの canvas（中身は StrokeOrder が描く）。 */
class GuideCanvas extends Component {
  componentDidMount() {
    A.attachGuide(new StrokeOrder(this.base));
  }

  shouldComponentUpdate() {
    return false;
  }

  render() {
    return html`<canvas id="guideCanvas"></canvas>`;
  }
}

function Tier({ s }) {
  const info = s.current;
  if (info?.mode === "queue") return html`<span class="chip queue" id="subjectTier">依頼 ${s.queueIndex + 1} / ${s.queue.length}</span>`;
  if (info?.mode === "forced") return html`<span class="chip queue" id="subjectTier">書き直し</span>`;
  if (info?.char) return html`<span class="chip" id="subjectTier" data-tier=${info.tier}>${info.tier_label}</span>`;
  if (info) return html`<span class="chip" id="subjectTier">全部そろいました</span>`;
  return html`<span class="chip" id="subjectTier">${s.profile ? "—" : "書き手を選んでください"}</span>`;
}

function Subject({ s }) {
  const info = s.current;
  const char = info?.char;
  const strokes = char ? A.glyph(char) : null;
  const count = info?.count ?? 0;
  const target = info?.target ?? 0;
  return html`
    <div class="subject" id="subject">
      <div class="subject-head">
        <${Tier} s=${s} />
        <span class="dots" id="subjectDots" aria-label=${char ? `${count} / ${target} サンプル` : "サンプル数"}>
          ${char && Array.from({ length: Math.max(target, count) }, (_, i) => html`<i class=${[i < count && "on", s.pop && i === count - 1 && "pop"].filter(Boolean).join(" ")}></i>`)}
        </span>
      </div>
      <div class="subject-body">
        <div class="subject-char swap" id="subjectChar" aria-live="polite" key=${char || "none"}>${char || (info ? "完" : s.profile ? "　" : "？")}</div>
        <button class=${`subject-guide${strokes && !strokes.length ? " empty" : ""}`} id="guideBtn" type="button" aria-label="書き順をもう一度見る" onClick=${A.replayGuide}>
          <${GuideCanvas} />
          <span class="guide-replay"><${Icon} name="i-play" />書き順</span>
        </button>
      </div>
      <div class="subject-foot">
        <span id="subjectInfo">${!char ? "" : strokes ? (strokes.length ? `お手本 ${strokes.length} 画` : "お手本なし") : "お手本 — 画"}</span>
        <span class="subject-next">次 <b id="nextChar">${info?.next || "—"}</b></span>
      </div>
    </div>
  `;
}

/** 書き込みパッド（中の canvas は WritingPad が作る）。 */
class PadBox extends Component {
  componentDidMount() {
    A.attachPad(new WritingPad(this.base));
  }

  shouldComponentUpdate() {
    return false;
  }

  render() {
    return html`
      <div class="pad" id="pad">
        <div class="pad-grid" aria-hidden="true"></div>
        <div class="pad-hint" id="padHint"><${Icon} name="i-pen" /><span>Apple Pencil で、ふだんどおりに書いてください</span></div>
      </div>
    `;
  }
}

function Welcome({ hidden }) {
  const [id, setId] = useState("");
  return html`
    <form class="welcome" id="welcome" hidden=${hidden} autocomplete="off"
      onSubmit=${(e) => {
        e.preventDefault();
        A.createProfile(id.trim());
      }}>
      <p class="welcome-kicker">はじめに</p>
      <h2 class="welcome-title">あなたの字を、<br />教えてください。</h2>
      <p class="welcome-text">ここで書いた字が、スタジオの清書とプロッタの線になります。まず書き手の名前（英数字）を決めましょう。</p>
      <div class="welcome-row">
        <input id="welcomeInput" placeholder="例: taiga" pattern="[A-Za-z0-9_\\-]+" aria-label="書き手の ID" required value=${id} onInput=${(e) => setId(e.currentTarget.value)} />
        <button class="btn btn-primary" type="submit">始める<${Icon} name="i-arrow" /></button>
      </div>
    </form>
  `;
}

function strokeChip(n, expected) {
  if (n === 0) return ["idle", expected ? `お手本は ${expected} 画` : "0 画"];
  if (!expected) return ["idle", `${n} 画`];
  if (n === expected) return ["match", `${n} 画 — お手本と同じ`];
  return ["differ", `${n} 画（お手本は ${expected} 画）`];
}

function WriteView({ s }) {
  const n = s.padCount;
  const char = s.current?.char;
  const expected = char ? (A.glyph(char) || []).length : 0;
  const [chipState, chipText] = strokeChip(n, expected);
  const tally = s.tally || { completed: 0, total: 381 };
  return html`
    <section class="view write" id="viewWrite" aria-label="筆跡を集める" hidden=${s.mode !== "write"}>
      <aside class="rail">
        <${Request} s=${s} />
        <${Subject} s=${s} />
        <div class="actions">
          <button class="btn btn-primary btn-xl" id="saveBtn" type="button" disabled=${n === 0 || !char} onClick=${A.save}>
            <${Icon} name="i-check" /><span>保存して次へ</span><kbd class="kbd-hint">↵</kbd>
          </button>
          <div class="actions-row">
            <button class="btn btn-quiet" id="undoBtn" type="button" disabled=${n === 0} onClick=${A.undoStroke}><${Icon} name="i-undo" />1画戻す</button>
            <button class="btn btn-quiet" id="clearBtn" type="button" disabled=${n === 0} onClick=${A.clearPad}><${Icon} name="i-x" />書き直す</button>
            <button class="btn btn-quiet" id="skipBtn" type="button" onClick=${A.skip}><${Icon} name="i-right" />飛ばす</button>
          </div>
        </div>
        <ul class="tips" aria-label="操作のこつ">
          <li><kbd>↵</kbd>保存して次へ</li>
          <li><span class="gesture">２本指タップ</span>1画戻す</li>
          <li><kbd>→</kbd>飛ばす</li>
        </ul>
        <div class="tally">
          <div class="tally-bar"><span id="tallyFill" style=${{ width: `${(tally.completed / Math.max(tally.total, 1)) * 100}%` }}></span></div>
          <div class="tally-text">
            <span><b id="tallyDone">${tally.completed}</b> / <span id="tallyTotal">${tally.total}</span> 字そろった</span>
            <span>今回 <b id="tallySession">${s.session}</b> 字</span>
          </div>
        </div>
      </aside>

      <div class="stage-pad">
        <div class="pad-frame">
          <${PadBox} />
          <div class="fly" id="fly" aria-hidden="true"></div>
          <${Welcome} hidden=${Boolean(s.profile)} />
        </div>
        <div class="pad-bar">
          <span class="stroke-chip" id="strokeChip" data-state=${chipState}><span id="strokeText">${chipText}</span></span>
          <label class="mini-switch"><input type="checkbox" class="switch" id="ghostToggle" checked=${s.ghost} onChange=${(e) => A.setGhost(e.currentTarget.checked)} /><span>お手本を敷く</span></label>
          <label class="mini-switch" id="touchToggleWrap"><input type="checkbox" class="switch" id="touchToggle" checked=${s.touch} onChange=${(e) => A.setTouch(e.currentTarget.checked)} /><span>指でも書く</span></label>
        </div>
      </div>
    </section>
  `;
}

// ================================================================== 見直す

const sampleSeconds = (strokes) => {
  const first = strokes[0]?.[0]?.timestamp;
  const lastStroke = strokes[strokes.length - 1];
  const last = lastStroke?.[lastStroke.length - 1]?.timestamp;
  return first !== undefined && last !== undefined ? Math.max(0, (last - first) / 1000) : 0;
};

/** 筆跡（Y-DOWN・論理座標）または清書（Y-UP）を canvas に描く。 */
function StrokesCanvas({ strokes, opts }) {
  const ref = useRef(null);
  useEffect(() => {
    const raf = requestAnimationFrame(() => drawStrokes(ref.current, strokes, opts));
    return () => cancelAnimationFrame(raf);
  }, [strokes]);
  return html`<canvas ref=${ref}></canvas>`;
}

// 古いサンプルは書き込み欄の大きさが違うので、外接矩形に合わせる（小さな字は小さいまま）
const SAMPLE_OPTS = { minSpan: LOGICAL * 0.55, margin: 0.14 };

function Thumb({ sample, onDelete, outlier = false }) {
  const [gone, setGone] = useState(false);
  const secs = sampleSeconds(sample.strokes);
  const del = async () => {
    await onDelete();
    setGone(true);
  };
  return html`
    <div class=${`thumb${outlier ? " outlier" : ""}${gone ? " is-gone" : ""}`}>
      <${StrokesCanvas} strokes=${sample.strokes} opts=${SAMPLE_OPTS} />
      <div class="thumb-meta">${sample.stroke_count}画${secs ? ` · ${secs.toFixed(1)}秒` : ""}</div>
      ${onDelete && html`<button class="thumb-del" type="button" aria-label="このサンプルを消す" onClick=${del}><${Icon} name="i-trash" /></button>`}
    </div>
  `;
}

const FILTERS = [
  ["all", "すべて"],
  ["todo", "まだ"],
  ["done", "そろった"],
  ["issues", "要確認"],
];
const TIER_NAMES = ["レポート頻出", "基本", "標準"];

function Cells({ s }) {
  const { stats, filter, query } = s;
  const flagged = A.issueChars(s.issues);
  const target = stats.target;
  const entries = Object.entries(stats.char_counts).filter(([c, n]) => {
    if (query && c !== query) return false;
    if (filter === "todo") return n < target;
    if (filter === "done") return n >= target;
    return true;
  });
  if (!entries.length) return html`<div class="cells-empty">${query ? `「${query}」はまだありません` : "該当する字はありません"}</div>`;
  return entries.map(
    ([c, n], i) => html`
      <button key=${c} class=${`cell${flagged.has(c) ? " has-issue" : ""}`} type="button" data-level=${Math.min(n, 3)} title=${`${c}: ${n} サンプル`}
        style=${{ "--i": Math.min(i, 200) }} onClick=${() => A.openDrawer(c)}>
        ${c}<span class="cell-bar"><i></i><i></i><i></i></span>
      </button>
    `,
  );
}

function IssueCard({ a }) {
  const [gone, setGone] = useState(false);
  return html`
    <div class=${`issue${gone ? " is-gone" : ""}`}>
      <div class="issue-top">
        <span class="issue-char">${a.character}</span>
        <div class="reasons">${a.reasons.map((r) => html`<span class="reason">${r}</span>`)}</div>
      </div>
      <${Thumb} sample=${a} />
      <div class="issue-actions">
        <button class="btn btn-quiet small" type="button"
          onClick=${async () => {
            await A.dismissIssue(a.character, [a.filename], "ignore_anomaly");
            setGone(true);
          }}>
          <${Icon} name="i-check" />問題ない
        </button>
        <button class="btn btn-danger small" type="button"
          onClick=${async () => {
            await A.trash(a.character, a.filename);
            setGone(true);
          }}>
          <${Icon} name="i-trash" />消す
        </button>
      </div>
    </div>
  `;
}

function MismatchCard({ m }) {
  const [gone, setGone] = useState(false);
  const outliers = m.samples.filter((x) => x.is_outlier).map((x) => x.filename);
  return html`
    <div class=${`mismatch issue${gone ? " is-gone" : ""}`}>
      <div class="issue-top">
        <span class="issue-char">${m.character}</span>
        <span class="reason mild">多いのは ${m.mode_count} 画</span>
      </div>
      <div class="mismatch-row">
        ${m.samples.map(
          (x) => html`<${Thumb} key=${x.filename} sample=${x} outlier=${x.is_outlier} onDelete=${x.is_outlier ? () => A.trash(m.character, x.filename) : null} />`,
        )}
      </div>
      <div class="issue-actions">
        <button class="btn btn-quiet small" type="button"
          onClick=${async () => {
            await A.dismissIssue(m.character, outliers, "ignore_stroke_mismatch");
            setGone(true);
          }}>
          <${Icon} name="i-check" />どれも正しい
        </button>
      </div>
    </div>
  `;
}

function Issues({ issues }) {
  const { anomalies, mismatches } = issues;
  if (!anomalies.length && !mismatches.length) {
    return html`<div class="cells-empty"><${Icon} name="i-check" /><br />確認が必要なサンプルはありません</div>`;
  }
  return html`
    ${anomalies.length > 0 &&
    html`
      <h4 class="issues-title">形があやしい（${anomalies.length}）</h4>
      <div class="issue-grid">${anomalies.map((a) => html`<${IssueCard} key=${a.filename} a=${a} />`)}</div>
    `}
    ${mismatches.length > 0 &&
    html`
      <h4 class="issues-title">画数がそろっていない（${mismatches.length}）</h4>
      ${mismatches.map((m) => html`<${MismatchCard} key=${m.character} m=${m} />`)}
    `}
  `;
}

const PREVIEW_LABEL = {
  user_strokes: "あなたの筆跡から",
  ml_inference: "ML があなた風に変形",
  kanjivg: "お手本の字形のまま（筆跡を集めると変わります）",
  geometric: "幾何字形",
  missing_glyphs: "字形がありません",
};
const PREVIEW_OPTS = { yUp: true, minSpan: 6, margin: 0.16 };
const skeletons = html`<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>`;

function Drawer({ drawer }) {
  const d = drawer || {};
  const { info, samples, preview } = d;
  return html`
    <aside class="drawer" id="drawer" aria-label="字の詳細" hidden=${!drawer}>
      <div class="drawer-head">
        <div class="drawer-char" id="drawerChar">${d.char || "　"}</div>
        <div class="drawer-meta">
          <span class="chip" id="drawerTier" data-tier=${info?.tier}>${info?.tier_label || ""}</span>
          <span id="drawerCount">${info && samples ? `${samples.length} / ${info.target} サンプル` : ""}</span>
        </div>
        <button class="icon-btn" id="drawerClose" type="button" aria-label="閉じる" onClick=${A.closeDrawer}><${Icon} name="i-x" /></button>
      </div>
      <div class="drawer-actions">
        <button class="btn btn-primary grow" id="drawerWrite" type="button" onClick=${A.writeDrawerChar}><${Icon} name="i-pen" />この字を書く</button>
        <button class="btn btn-danger" id="drawerDeleteAll" type="button" disabled=${!samples?.length} onClick=${A.deleteAll}><${Icon} name="i-trash" />全部消す</button>
      </div>
      <h4 class="drawer-title">清書では</h4>
      <div class="preview" id="drawerPreview">
        ${!drawer
          ? null
          : !preview
            ? skeletons
            : html`
                ${preview.variants.map((v, i) => html`<div class="thumb" key=${`${d.char}-${i}`}><${StrokesCanvas} strokes=${v.map(unflatten)} opts=${PREVIEW_OPTS} /></div>`)}
                <p class="preview-note">${preview.source ? PREVIEW_LABEL[preview.source] : "試し書きできませんでした"}</p>
              `}
      </div>
      <h4 class="drawer-title">あなたのサンプル</h4>
      <div class="samples" id="drawerSamples">
        ${!drawer
          ? null
          : !samples
            ? skeletons
            : samples.length
              ? samples.map((x) => html`<${Thumb} key=${x.filename} sample=${x} onDelete=${() => A.trash(d.char, x.filename)} />`)
              : html`<p class="preview-note">まだ書いていません。「この字を書く」から書けます</p>`}
      </div>
    </aside>
  `;
}

function ReviewView({ s }) {
  const { stats } = s;
  const tiers = stats?.tiers || {};
  const done = Object.values(tiers).reduce((n, t) => n + t.completed, 0);
  const total = Object.values(tiers).reduce((n, t) => n + t.total, 0) || 381;
  const issues = A.issueTotal(s.issues);
  const showIssues = s.filter === "issues";
  return html`
    <section class="view review" id="viewReview" aria-label="筆跡を見直す" hidden=${s.mode !== "review"}>
      <div class="review-head">
        <div class="figures">
          <div class="figure"><b id="figDone">${done}</b><span>/ <span id="figTotal">${total}</span> 字そろった</span></div>
          <div class="figure"><b id="figSamples">${(stats?.total_samples || 0).toLocaleString()}</b><span>サンプル</span></div>
          <div class=${`figure warn${issues ? "" : " zero"}`}><b id="figIssues">${issues}</b><span>要確認</span></div>
        </div>
        <div class="tiers" id="tiers">
          ${stats &&
          ["tier1", "tier2", "tier3"].map((key, i) => {
            const t = tiers[key];
            return html`
              <div class="tier" data-tier=${i} key=${key}>
                <span>${TIER_NAMES[i]}</span>
                <div class="tier-track"><span style=${{ width: `${(t.completed / Math.max(t.total, 1)) * 100}%` }}></span></div>
                <b>${t.completed}/${t.total}</b>
              </div>
            `;
          })}
        </div>
      </div>
      <div class="review-tools">
        <div class="chips" role="radiogroup" aria-label="絞り込み" id="filters">
          ${FILTERS.map(
            ([f, label]) => html`<button key=${f} type="button" role="radio" data-filter=${f} aria-checked=${String(s.filter === f)} onClick=${() => A.setFilter(f)}>${label}</button>`,
          )}
        </div>
        <label class="search">
          <${Icon} name="i-search" />
          <input id="searchInput" placeholder="字をさがす" maxlength="1" aria-label="字をさがす" value=${s.query} onInput=${(e) => A.search(e.currentTarget.value)} />
        </label>
        <div class="legend" aria-hidden="true"><i class="l0"></i>まだ<i class="l1"></i>途中<i class="l3"></i>そろった</div>
      </div>
      <div class="review-body" id="reviewBody">
        <div class="cells" id="cells" hidden=${showIssues}>${stats && !showIssues && html`<${Cells} s=${s} />`}</div>
        <div class="issues" id="issues" hidden=${!showIssues}>${s.issues && showIssues && html`<${Issues} issues=${s.issues} />`}</div>
      </div>
      <${Drawer} drawer=${s.drawer} />
      <div class="scrim" id="scrim" hidden=${!s.drawer} onClick=${A.closeDrawer}></div>
    </section>
  `;
}

// ================================================================== 学習

const TITLES = { idle: "待機中", running: "学習しています", succeeded: "学習が終わりました", failed: "学習に失敗しました", cancelled: "止めました" };

function trainSub(t) {
  if (!t) return "学習を始めると、ここに進み具合が出ます";
  if (t.state === "running") {
    if (t.epoch === 0) return "準備しています（筆跡とお手本を読み込み中）";
    const elapsed = t.started_at ? Date.now() / 1000 - t.started_at : 0;
    const remain = (elapsed / t.epoch) * (t.total_epochs - t.epoch);
    return `エポック ${t.epoch} / ${t.total_epochs}${Number.isFinite(remain) ? ` · 残り 約${formatDuration(remain)}` : ""}`;
  }
  if (t.state === "succeeded") return `保存しました: ${t.checkpoint_name || t.checkpoint_path}`;
  if (t.state === "failed") return t.error || "";
  return "学習を始めると、ここに進み具合が出ます";
}

function sparkPaths(values) {
  if (values.length < 2) return ["", ""];
  const lo = Math.min(...values);
  const span = Math.max(...values) - lo || 1;
  const line = values
    .map((v, i) => `${i ? "L" : "M"}${((i / (values.length - 1)) * 300).toFixed(1)},${(58 - ((v - lo) / span) * 50).toFixed(1)}`)
    .join("");
  return [line, `${line}L300,64L0,64Z`];
}

function MiniSeg({ id, value, options, onChange }) {
  return html`
    <div class="seg mini" id=${id} data-active=${value}>
      ${options.map(
        ([v, label]) => html`<button key=${v} class="seg-btn" type="button" data-value=${v} aria-selected=${String(v === value)} onClick=${() => onChange(v)}>${label}</button>`,
      )}
      <span class="seg-glider" aria-hidden="true"></span>
    </div>
  `;
}

function TrainCard({ s }) {
  const models = s.models.models;
  const [form, setForm] = useState({ dataset: "current", kind: "user_train", epochs: "", batch: "", lr: "", deformer: "twostage", device: "", output: "", base: "" });
  const set = (k) => (e) => setForm({ ...form, [k]: e.currentTarget.value });
  const running = s.training?.state === "running";
  const fallbackBase = models.some((m) => m.name === "pretrain_checkpoint.pt") ? "pretrain_checkpoint.pt" : models[0]?.name || "";
  const base = models.some((m) => m.name === form.base) ? form.base : fallbackBase;
  return html`
    <div class="card train-card">
      <div class="card-head"><h3 class="card-title">学習させる</h3><span class="card-meta" id="trainProfile">${s.profile ? `書き手: ${s.profile}` : ""}</span></div>
      <p class="train-lead">集めた筆跡から、あなたの書き癖をモデルに覚えさせます。終わったら「このモデルを使う」でスタジオの清書に反映されます。</p>
      <div class="field-row">
        <span class="field-label">データ</span>
        <${MiniSeg} id="trainDataset" value=${form.dataset} options=${[["current", "この人だけ"], ["all", "全員"]]} onChange=${(v) => setForm({ ...form, dataset: v })} />
      </div>
      <div class="field-row">
        <span class="field-label">方法</span>
        <${MiniSeg} id="trainKind" value=${form.kind} options=${[["user_train", "はじめから"], ["finetune", "微調整"]]} onChange=${(v) => setForm({ ...form, kind: v })} />
      </div>
      <details class="advanced">
        <summary>詳細設定<${Icon} name="i-chevron" /></summary>
        <div class="adv-grid">
          <label><span>エポック数</span><input type="number" id="trainEpochs" min="1" placeholder="80" value=${form.epochs} onInput=${set("epochs")} /></label>
          <label><span>バッチサイズ</span><input type="number" id="trainBatch" min="1" placeholder="256" value=${form.batch} onInput=${set("batch")} /></label>
          <label><span>学習率</span><input type="number" id="trainLr" step="0.0001" min="0" placeholder="0.001" value=${form.lr} onInput=${set("lr")} /></label>
          <label>
            <span>変形モデル</span>
            <select id="trainDeformer" value=${form.deformer} onChange=${set("deformer")}>
              <option value="twostage">twostage（推奨）</option><option value="transformer">transformer</option><option value="offset">offset</option>
            </select>
          </label>
          <label><span>デバイス</span><input id="trainDevice" placeholder="自動（cuda / xpu / cpu）" value=${form.device} onInput=${set("device")} /></label>
          <label><span>保存名</span><input id="trainOutput" placeholder="自動（日時）" value=${form.output} onInput=${set("output")} /></label>
          <label class="wide" id="trainBaseWrap" hidden=${form.kind !== "finetune"}>
            <span>微調整の元モデル</span>
            <select id="trainBase" value=${base} onChange=${set("base")}>${models.map((m) => html`<option key=${m.name} value=${m.name}>${m.name}</option>`)}</select>
          </label>
        </div>
      </details>
      <div class="btn-row">
        <button class="btn btn-ink btn-lg grow" id="trainStart" type="button" hidden=${running} onClick=${() => A.startTraining({ ...form, base })}>
          <${Icon} name="i-play" />学習を始める
        </button>
        <button class="btn btn-quiet" id="trainCancel" type="button" hidden=${!running} onClick=${A.cancelTraining}><${Icon} name="i-stop" />止める</button>
      </div>
    </div>
  `;
}

function StatusCard({ s }) {
  const t = s.training;
  const state = t?.state || "idle";
  const pct = t?.total_epochs ? Math.floor((t.epoch / t.total_epochs) * 100) : 0;
  const shown = state === "succeeded" ? 100 : pct;
  const logs = t?.logs || [];
  const [line, area] = sparkPaths(logs.map((l) => /loss=([\d.eE+-]+)/.exec(l)).filter(Boolean).map((m) => Number(m[1])));
  const fresh = state === "succeeded" && t.checkpoint_name && t.checkpoint_name !== s.models.active;
  return html`
    <div class="card status-card" id="statusCard" data-state=${state}>
      <div class="status-top">
        <div class="run-meter">
          <svg class="ring" viewBox="0 0 100 100" aria-hidden="true">
            <circle class="ring-bg" cx="50" cy="50" r="44" />
            <circle class=${`ring-fg${shown > 0 ? " has-value" : ""}`} id="trainRing" cx="50" cy="50" r="44" pathLength="100" style=${{ strokeDasharray: `${shown} 100` }} />
          </svg>
          <div class="ring-label"><span class="ring-pct" id="trainPct">${shown}</span><span class="ring-unit">%</span></div>
        </div>
        <div>
          <div class="run-title" id="trainTitle">${TITLES[state] || state}</div>
          <div class="run-sub" id="trainSub">${trainSub(t)}</div>
        </div>
      </div>
      <div class="loss">
        <div class="loss-head"><span>損失（下がるほど覚えた）</span><b id="trainLoss">${t?.loss === null || t?.loss === undefined ? "—" : Number(t.loss).toFixed(4)}</b></div>
        <svg class="spark" id="spark" viewBox="0 0 300 64" preserveAspectRatio="none" aria-hidden="true"><path class="spark-area" id="sparkArea" d=${area} /><path class="spark-line" id="sparkLine" d=${line} /></svg>
      </div>
      <div class="btn-row" id="useNewWrap" hidden=${!fresh}>
        <button class="btn btn-primary grow" id="useNew" type="button" onClick=${() => A.useModel(t.checkpoint_name)}><${Icon} name="i-sparkle" />このモデルを使う</button>
      </div>
      <details class="log-card inline">
        <summary><${Icon} name="i-terminal" /><span>ログ</span><${Icon} name="i-chevron" cls="icon chev" /></summary>
        <pre class="log" id="trainLog">${logs.join("\n")}</pre>
      </details>
    </div>
  `;
}

function ModelsCard({ s }) {
  const { models, active } = s.models;
  const newest = s.training?.state === "succeeded" ? s.training.checkpoint_name : null;
  return html`
    <div class="card models-card">
      <div class="card-head"><h3 class="card-title">モデル</h3><span class="card-meta" id="modelsMeta">${active ? `使用中: ${active}` : "ML を使っていません"}</span></div>
      <ul class="models" id="models">
        ${models.map((m) => {
          const isActive = m.name === active;
          const date = new Date(m.modified * 1000).toLocaleString("ja-JP", { dateStyle: "short", timeStyle: "short" });
          return html`
            <li key=${m.name} class=${`model${isActive ? " is-active" : ""}${m.name === newest ? " is-new" : ""}`}>
              <div><div class="model-name">${m.name}</div><div class="model-meta">${date} · ${(m.size / 1e6).toFixed(1)} MB</div></div>
              ${isActive
                ? html`<span class="badge"><${Icon} name="i-check" />使用中</span>`
                : html`<button class="btn btn-quiet small" type="button" onClick=${() => A.useModel(m.name)}>使う</button>`}
            </li>
          `;
        })}
        ${!models.length && html`<li class="models-empty">まだモデルがありません。左で学習させると、ここに並びます</li>`}
        ${active &&
        html`
          <li class="model">
            <div class="model-meta">ML を使わずに清書する</div>
            <button class="btn btn-quiet small" type="button" onClick=${() => A.useModel(null)}>使わない</button>
          </li>
        `}
      </ul>
    </div>
  `;
}

const TrainView = ({ s }) => html`
  <section class="view train" id="viewTrain" aria-label="学習" hidden=${s.mode !== "train"}>
    <div class="train-grid">
      <${TrainCard} s=${s} />
      <${StatusCard} s=${s} />
      <${ModelsCard} s=${s} />
    </div>
  </section>
`;
