// 原稿エディタ: textarea の下に同じ寸法のハイライト層を重ね、書式（見出し・数式・表）を色分けする。

const escapeHtml = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);

function highlightInline(line) {
  // $...$ を数式として色付け（$$ はブロック側で処理）
  let out = "";
  let rest = line;
  for (;;) {
    const m = rest.match(/\$(?!\$)([^$]+)\$/);
    if (!m) break;
    out += escapeHtml(rest.slice(0, m.index));
    out += `<span class="tok-math"><span class="tok-mark">$</span>${escapeHtml(m[1])}<span class="tok-mark">$</span></span>`;
    rest = rest.slice(m.index + m[0].length);
  }
  out += escapeHtml(rest);
  return out.replace(/\\noindent/g, '<span class="tok-cmd">\\noindent</span>');
}

export function highlight(text) {
  const lines = text.split("\n");
  let inBlock = false;
  const html = lines.map((line) => {
    const trimmed = line.trim();
    if (inBlock || trimmed.startsWith("$$")) {
      const opens = !inBlock;
      const closes = (opens && trimmed.length > 2 && trimmed.endsWith("$$")) || (!opens && trimmed.endsWith("$$"));
      inBlock = !closes;
      return `<span class="tok-block">${escapeHtml(line)}</span>`;
    }
    const heading = line.match(/^(#{1,3})(.*)$/);
    if (heading) {
      return `<span class="tok-h tok-h${heading[1].length}"><span class="tok-mark">${heading[1]}</span>${highlightInline(heading[2] || "")}</span>`;
    }
    if (/^\s*\|.*\|\s*$/.test(line)) {
      if (/^\s*\|[\s:|-]+\|\s*$/.test(line)) return `<span class="tok-rule">${escapeHtml(line)}</span>`;
      return `<span class="tok-table">${highlightInline(line).replace(/\|/g, '<span class="tok-mark">|</span>')}</span>`;
    }
    if (/^:\s/.test(line)) {
      return `<span class="tok-caption"><span class="tok-mark">:</span>${highlightInline(line.slice(1))}</span>`;
    }
    return highlightInline(line);
  });
  // 末尾の改行でも高さが揃うように 1 文字足す
  return `${html.join("\n")}\n `;
}

export class Editor {
  /**
   * @param {HTMLTextAreaElement} textarea
   * @param {HTMLElement} layer ハイライト層（textarea と同じ箱）
   */
  constructor(textarea, layer) {
    this.textarea = textarea;
    this.layer = layer;
    const sync = () => {
      this.layer.innerHTML = highlight(this.textarea.value);
      this._syncScroll();
    };
    textarea.addEventListener("input", sync);
    textarea.addEventListener("scroll", () => this._syncScroll());
    textarea.addEventListener("keydown", (e) => this._onKey(e));
    this.refresh = sync;
    sync();
  }

  get value() {
    return this.textarea.value;
  }

  set value(text) {
    this.textarea.value = text;
    this.refresh();
    this.textarea.dispatchEvent(new Event("input", { bubbles: true }));
  }

  _syncScroll() {
    this.layer.scrollTop = this.textarea.scrollTop;
    this.layer.scrollLeft = this.textarea.scrollLeft;
  }

  /** 選択範囲を置き換え（undo 履歴に残す）。 */
  insert(text) {
    this.textarea.focus();
    if (!document.execCommand("insertText", false, text)) {
      const { selectionStart: a, selectionEnd: b, value } = this.textarea;
      this.textarea.value = value.slice(0, a) + text + value.slice(b);
      this.textarea.selectionStart = this.textarea.selectionEnd = a + text.length;
      this.textarea.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }

  _onKey(e) {
    if (e.isComposing) return;
    const ta = this.textarea;
    // $ で選択範囲を数式に囲む
    if (e.key === "$" && ta.selectionStart !== ta.selectionEnd) {
      e.preventDefault();
      const sel = ta.value.slice(ta.selectionStart, ta.selectionEnd);
      this.insert(`$${sel}$`);
      return;
    }
    // 表の行で Enter → 次の行の先頭に "| " を補う
    if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
      const before = ta.value.slice(0, ta.selectionStart);
      const line = before.slice(before.lastIndexOf("\n") + 1);
      if (/^\s*\|.*\|\s*$/.test(line) && !/^\s*\|[\s:|-]+\|\s*$/.test(line)) {
        e.preventDefault();
        this.insert("\n| ");
      }
    }
  }
}
