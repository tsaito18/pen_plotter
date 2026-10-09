// スタジオと筆跡で共有する小さな道具（DOM・保存・トースト・テーマ・アイコン・htm）。

import { h } from "preact";
import { useEffect } from "preact/hooks";
import htm from "htm";

/** JSX の代わりのテンプレート（ビルド不要）。 */
export const html = htm.bind(h);

export const isMac = /Mac|iPhone|iPad/.test(navigator.platform);
export const MOD = isMac ? "⌘" : "Ctrl";
export const ICONS = "/static/icons.svg";

/** localStorage（失敗しても動き続ける）。キーは "pp." 付きで両画面が共有する。 */
export const STORAGE = {
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

export const icon = (name, cls = "icon") => `<svg class="${cls}"><use href="${ICONS}#${name}"/></svg>`;

/** アイコン（icons.svg のシンボル）。 */
export const Icon = ({ name, cls = "icon" }) => html`<svg class=${cls}><use href=${`${ICONS}#${name}`} /></svg>`;

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of [].concat(children)) if (c !== null && c !== undefined && c !== false) node.append(c);
  return node;
}

export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), ms);
  };
}

export function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "";
  if (seconds < 60) return "1分未満";
  const m = Math.round(seconds / 60);
  if (m < 60) return `${m}分`;
  return `${Math.floor(m / 60)}時間${m % 60 ? `${m % 60}分` : ""}`;
}

function dismiss(node, after) {
  window.setTimeout(() => {
    node.classList.add("is-leaving");
    node.addEventListener("animationend", () => node.remove(), { once: true });
  }, after);
}

/** 上部に一時的な通知を出す。action={label, run} で「元に戻す」などのボタンを付ける。 */
export function toast(message, kind = "info", timeout = 3400, action = null) {
  const icons = { ok: "i-check", warn: "i-alert", error: "i-alert", info: "i-sparkle", undo: "i-trash" };
  const node = el("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" });
  node.innerHTML = `${icon(icons[kind] || icons.info)}<span></span>`;
  node.querySelector("span").textContent = message;
  if (action) {
    const btn = el("button", { class: "toast-action", type: "button", text: action.label });
    btn.addEventListener("click", () => {
      action.run();
      node.remove();
    });
    node.append(btn);
  }
  document.getElementById("toasts").append(node);
  dismiss(node, action ? Math.max(timeout, 6000) : timeout);
  return node;
}

export const toastUndo = (message, undo) => toast(message, "undo", 6000, { label: "元に戻す", run: undo });

/** テーマを切り替える（選択は両画面で共有）。 */
export function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try {
    localStorage.setItem("pp.theme", root.dataset.theme);
  } catch {
    // 保存できなくても切り替えは効く
  }
}

/** テーマ切り替えボタン（両画面のトップバー）。 */
export const ThemeButton = ({ onToggle = () => {} }) => html`
  <button class="icon-btn" id="themeBtn" type="button" aria-label="テーマを切り替え" data-tip="テーマ"
    onClick=${() => {
      toggleTheme();
      onToggle();
    }}>
    <${Icon} name="i-sun" cls="icon icon-sun" /><${Icon} name="i-moon" cls="icon icon-moon" />
  </button>
`;

/** ロゴと画面の切り替え（スタジオ / 筆跡）。 */
export const Brand = ({ current }) => html`
  <div class="topbar-start">
    <a class="brand" href="/" aria-label="Pen Plotter スタジオへ">
      <span class="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 32 32"><path class="brand-stroke" d="M7 23c4-1 6-5 8-9s4-6 9-6" /></svg>
      </span>
      <span class="brand-name">Pen Plotter</span>
    </a>
    <nav class="apps" aria-label="画面">
      <a href="/" aria-current=${current === "studio" ? "page" : null}>スタジオ</a>
      <a href="/collect" aria-current=${current === "collect" ? "page" : null}>筆跡</a>
    </nav>
  </div>
`;

/** 入れ子の閉じ方（外側のクリック・Esc）を持つドロップダウン。 */
export function useDismiss(open, setOpen, ref) {
  useEffect(() => {
    if (!open) return undefined;
    const onClick = (e) => {
      if (!ref.current?.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("click", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("click", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
}

/** JSON を返す API 呼び出し。失敗時は detail をメッセージにした Error を投げる。 */
export async function api(path, { method = "GET", body, params } = {}) {
  const url = new URL(path, location.origin);
  for (const [k, v] of Object.entries(params || {})) if (v !== undefined && v !== null) url.searchParams.set(k, v);
  const res = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join(" / ") : data.detail;
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return data;
}
