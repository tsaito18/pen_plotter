// 用紙ビューア（Canvas）。紙座標は mm・Y-UP（左下原点）＝G-code と同じ。
//
// レイヤー（下から）: 用紙（スキャン画像 or 白地＋罫線）→ 下書き（組版ドラフト）→ インク（清書）
// → プロット進捗（未描画は淡く）→ ペン先マーカー。
// ズームはホイール（Ctrl/⌘ 併用またはピンチ）、パンはドラッグ / ホイール。

// 完全接触の線幅はサーバ（PlotterConfig.pen_width_mm＝実物のスキャン実測）から受け取る
let INK_MAX_MM = 0.35;
const INK_MIN_RATIO = 0.17; // 払いの抜けの最小幅（ペン幅比、preview.WIDTH_MIN_RATIO と同じ）
let INK_MIN_MM = INK_MAX_MM * INK_MIN_RATIO;
const CHUNK = 96; // 進捗描画用にストロークをまとめる単位
const WIDTH_BUCKETS = 8;

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const widthOf = (contact) => INK_MIN_MM + (INK_MAX_MM - INK_MIN_MM) * contact;

/** 1 ページ分のストロークを、描画を速くする Path2D 群へ前処理する。 */
function preparePage(page) {
  const strokes = page.strokes.map((s) => {
    const pts = s.points;
    const uniform = typeof s.contact === "number";
    const full = new Path2D();
    const tapers = []; // [x0,y0,x1,y1,width]
    let open = false;
    for (let i = 0; i + 3 < pts.length; i += 2) {
      const c = uniform ? s.contact : s.contact[i / 2];
      if (c >= 0.999) {
        if (!open) full.moveTo(pts[i], pts[i + 1]);
        full.lineTo(pts[i + 2], pts[i + 3]);
        open = true;
      } else {
        tapers.push([pts[i], pts[i + 1], pts[i + 2], pts[i + 3], widthOf(c)]);
        open = false;
      }
    }
    // 1 点だけの画（点）も見えるようにごく短い線にする
    if (pts.length === 2) {
      full.moveTo(pts[0], pts[1]);
      full.lineTo(pts[0] + 0.05, pts[1]);
    }
    const length = (() => {
      let l = 0;
      for (let i = 0; i + 3 < pts.length; i += 2) l += Math.hypot(pts[i + 2] - pts[i], pts[i + 3] - pts[i + 1]);
      return l;
    })();
    return { pts, contact: s.contact, uniform, full, tapers, length };
  });
  const chunks = [];
  for (let start = 0; start < strokes.length; start += CHUNK) {
    chunks.push(buildBatch(strokes.slice(start, start + CHUNK)));
  }
  const ends = (page.spans || []).map((span) => span[1]);
  return { strokes, chunks, all: buildBatch(strokes), spans: page.spans || [], ends };
}

function buildBatch(strokes) {
  const full = new Path2D();
  const buckets = Array.from({ length: WIDTH_BUCKETS }, () => ({ path: new Path2D(), width: 0, used: false }));
  for (const s of strokes) {
    full.addPath(s.full);
    for (const [x0, y0, x1, y1, w] of s.tapers) {
      const k = Math.min(WIDTH_BUCKETS - 1, Math.floor(((w - INK_MIN_MM) / (INK_MAX_MM - INK_MIN_MM)) * WIDTH_BUCKETS));
      const b = buckets[k];
      b.path.moveTo(x0, y0);
      b.path.lineTo(x1, y1);
      b.width = INK_MIN_MM + ((k + 0.5) / WIDTH_BUCKETS) * (INK_MAX_MM - INK_MIN_MM);
      b.used = true;
    }
  }
  return { full, buckets: buckets.filter((b) => b.used) };
}

export class PaperView {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{width:number, height:number}} paper 用紙寸法 (mm)
   */
  constructor(canvas, paper) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.paper = paper;
    if (paper.pen_width_mm) {
      INK_MAX_MM = paper.pen_width_mm;
      INK_MIN_MM = INK_MAX_MM * INK_MIN_RATIO;
    }
    this.scale = 1; // CSS px / mm
    this.ox = 0;
    this.oy = 0;
    this.fitted = true;
    this.insets = { top: 72, bottom: 104, left: 32, right: 32 };
    this.background = null;
    this.ruled = [];
    this.draft = null; // {chars, rules, math}
    this.lineSpacing = 7.14;
    this.ink = null; // preparePage の結果
    this.inkAlpha = 1;
    this.anim = null; // {start, duration, total}
    this.plot = null; // {line} 進捗（spans の行番号）
    this._showDraft = true;
    this.dirty = true; // 合成し直す
    this.baseDirty = true; // 用紙・下書き・静止インクの層を作り直す
    // 層: base（用紙＋下書き＋静止インク）と inc（書き進むインク。増えた画だけ足す）
    this.base = document.createElement("canvas");
    this.inc = document.createElement("canvas");
    this.incCount = 0;
    this.incInk = null;
    this.listeners = new Set();
    this._pointers = new Map();
    this._bind();
    this._resize();
    const loop = (t) => {
      if (this.anim) this._stepAnim(t);
      if (this.dirty || this.baseDirty) {
        this.dirty = false;
        this._draw();
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  // ---------------------------------------------------------------- 入力データ

  setBackground(url) {
    const img = new Image();
    img.onload = () => {
      this.background = img;
      this.invalidate();
    };
    img.src = url;
  }

  setRuled(lines) {
    this.ruled = lines || [];
    this.invalidate();
  }

  setDraft(page, lineSpacing) {
    if (page === this.draft && (!lineSpacing || lineSpacing === this.lineSpacing)) return;
    this.draft = page;
    if (lineSpacing) this.lineSpacing = lineSpacing;
    this.invalidate();
  }

  get showDraft() {
    return this._showDraft;
  }

  set showDraft(value) {
    if (value === this._showDraft) return;
    this._showDraft = value;
    this.invalidate();
  }

  /** 清書ページを表示する。animate=true ならペンが書き順どおりに走る。 */
  setInk(page, { animate = false } = {}) {
    this.ink = page ? (page._prepared ||= preparePage(page)) : null;
    this.anim = null;
    if (this.ink && animate && !prefersReducedMotion() && this.ink.strokes.length > 0) {
      const n = this.ink.strokes.length;
      this.anim = { start: null, duration: Math.min(3200, 1100 + n * 0.35), total: n, progress: 0 };
    }
    this.invalidate();
  }

  skipAnimation() {
    if (!this.anim) return false;
    this.anim = null;
    this.invalidate();
    this._emit("animationend");
    return true;
  }

  setInkAlpha(alpha) {
    if (alpha === this.inkAlpha) return;
    this.inkAlpha = alpha;
    this.invalidate();
  }

  /** プロット進捗（送信済み行数）。null で通常表示。 */
  setPlotProgress(line) {
    const next = line === null || line === undefined ? null : { line };
    if (Boolean(next) !== Boolean(this.plot)) this.invalidate();
    else if (next && next.line === this.plot.line) return;
    this.plot = next;
    this.dirty = true;
  }

  /** 用紙・下書き・インクの見た目が変わった（層を作り直す）。 */
  invalidate() {
    this.baseDirty = true;
    this.dirty = true;
  }

  on(handler) {
    this.listeners.add(handler);
  }

  _emit(type, detail) {
    for (const h of this.listeners) h(type, detail);
  }

  // ---------------------------------------------------------------- ビュー

  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.cssW = Math.max(rect.width, 1);
    this.cssH = Math.max(rect.height, 1);
    this.dpr = Math.min(window.devicePixelRatio || 1, 3);
    this.canvas.width = Math.round(this.cssW * this.dpr);
    this.canvas.height = Math.round(this.cssH * this.dpr);
    for (const layer of [this.base, this.inc]) {
      layer.width = this.canvas.width;
      layer.height = this.canvas.height;
    }
    if (this.fitted) this.fit(false);
    this.invalidate();
  }

  fit(notify = true) {
    const { top, bottom, left, right } = this.insets;
    const w = Math.max(this.cssW - left - right, 40);
    const h = Math.max(this.cssH - top - bottom, 40);
    this.scale = Math.min(w / this.paper.width, h / this.paper.height);
    this.ox = left + (w - this.paper.width * this.scale) / 2;
    this.oy = top + (h - this.paper.height * this.scale) / 2;
    this.fitted = true;
    this.invalidate();
    if (notify) this._emit("zoom", this.zoomPercent);
  }

  get fitScale() {
    const { top, bottom, left, right } = this.insets;
    return Math.min(
      Math.max(this.cssW - left - right, 40) / this.paper.width,
      Math.max(this.cssH - top - bottom, 40) / this.paper.height,
    );
  }

  get zoomPercent() {
    return Math.round((this.scale / this.fitScale) * 100);
  }

  zoomAt(factor, cx = this.cssW / 2, cy = this.cssH / 2) {
    const min = this.fitScale * 0.5;
    const max = this.fitScale * 12;
    const next = Math.min(max, Math.max(min, this.scale * factor));
    const k = next / this.scale;
    this.ox = cx - (cx - this.ox) * k;
    this.oy = cy - (cy - this.oy) * k;
    this.scale = next;
    this.fitted = false;
    this.invalidate();
    this._emit("zoom", this.zoomPercent);
  }

  pan(dx, dy) {
    this.ox += dx;
    this.oy += dy;
    this.fitted = false;
    this.invalidate();
  }

  toScreen(x, y) {
    return [this.ox + x * this.scale, this.oy + (this.paper.height - y) * this.scale];
  }

  _bind() {
    new ResizeObserver(() => this._resize()).observe(this.canvas);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.invalidate());
    const c = this.canvas;
    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        if (e.ctrlKey || e.metaKey) {
          this.zoomAt(Math.exp(-e.deltaY * 0.0025), e.offsetX, e.offsetY);
        } else {
          this.pan(-e.deltaX, -e.deltaY);
        }
      },
      { passive: false },
    );
    c.addEventListener("pointerdown", (e) => {
      if (this.skipAnimation()) return;
      c.setPointerCapture(e.pointerId);
      this._pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
      this._pinch = null;
      c.classList.add("is-grabbing");
    });
    c.addEventListener("pointermove", (e) => {
      const prev = this._pointers.get(e.pointerId);
      if (!prev) return;
      const cur = { x: e.offsetX, y: e.offsetY };
      if (this._pointers.size === 2) {
        const [a, b] = [...this._pointers.values()];
        const other = a === prev ? b : a;
        const dist = Math.hypot(cur.x - other.x, cur.y - other.y);
        if (this._pinch) {
          this.zoomAt(dist / this._pinch, (cur.x + other.x) / 2, (cur.y + other.y) / 2);
        }
        this._pinch = dist;
      } else {
        this.pan(cur.x - prev.x, cur.y - prev.y);
      }
      this._pointers.set(e.pointerId, cur);
    });
    const end = (e) => {
      this._pointers.delete(e.pointerId);
      this._pinch = null;
      if (this._pointers.size === 0) c.classList.remove("is-grabbing");
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("dblclick", (e) => {
      if (this.zoomPercent > 130) this.fit();
      else this.zoomAt(2.5 / (this.scale / this.fitScale), e.offsetX, e.offsetY);
    });
  }

  // ---------------------------------------------------------------- 描画

  _stepAnim(t) {
    const a = this.anim;
    if (a.start === null) a.start = t;
    const x = Math.min((t - a.start) / a.duration, 1);
    // 書き始めは速く、終盤はゆっくり（ease-out）
    a.progress = 1 - (1 - x) ** 2.2;
    this.dirty = true;
    if (x >= 1) {
      this.anim = null;
      this.invalidate();
      this._emit("animationend");
    }
  }

  _paperTransform(ctx) {
    const s = this.scale * this.dpr;
    ctx.setTransform(s, 0, 0, -s, this.ox * this.dpr, (this.oy + this.paper.height * this.scale) * this.dpr);
  }

  get _minW() {
    return 0.55 / this.scale; // 縮小表示でも線が消えない最小幅（CSS px 相当）
  }

  /** 書き進むインク（アニメーション・プロット）の進み具合。 */
  _progressive() {
    if (!this.ink) return null;
    if (this.plot) return this._plotSplit(this.plot.line);
    if (this.anim) {
      const exact = this.anim.progress * this.ink.strokes.length;
      const done = Math.floor(exact);
      return { done, partial: exact - done };
    }
    return null;
  }

  _draw() {
    const ink = css("--ink") || "#26252a";
    if (this.baseDirty) {
      this.baseDirty = false;
      this._drawBase();
      this.incCount = 0;
      this.incInk = null;
    }
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.drawImage(this.base, 0, 0);

    const prog = this._progressive();
    if (prog) {
      // 完了した画は inc 層へ追記だけする（毎フレーム全画を描き直さない）
      const ictx = this.inc.getContext("2d");
      if (this.incInk !== this.ink || prog.done < this.incCount) {
        ictx.setTransform(1, 0, 0, 1, 0, 0);
        ictx.clearRect(0, 0, this.inc.width, this.inc.height);
        this.incCount = 0;
        this.incInk = this.ink;
      }
      if (prog.done > this.incCount) {
        this._paperTransform(ictx);
        ictx.lineCap = "round";
        ictx.lineJoin = "round";
        this._drawRange(ictx, this.incCount, prog.done, ink, this._minW);
        this.incCount = prog.done;
      }
      ctx.drawImage(this.inc, 0, 0);
      if (prog.done < this.ink.strokes.length && prog.partial > 0) {
        this._paperTransform(ctx);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = ink;
        this._drawStroke(ctx, this.ink.strokes[prog.done], prog.partial, this._minW);
      }
      this._drawTip(ctx, prog);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  _drawBase() {
    const ctx = this.base.getContext("2d");
    const { width: W, height: H } = this.paper;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.base.width, this.base.height);

    // 用紙（影つき）
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const [px, py] = [this.ox, this.oy];
    const [pw, ph] = [W * this.scale, H * this.scale];
    ctx.save();
    ctx.shadowColor = css("--paper-shadow") || "rgba(0,0,0,.18)";
    ctx.shadowBlur = 32;
    ctx.shadowOffsetY = 10;
    ctx.fillStyle = "#fbfaf6";
    ctx.fillRect(px, py, pw, ph);
    ctx.restore();
    if (this.background) {
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(this.background, px, py, pw, ph);
    }

    this._paperTransform(ctx);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (!this.background && this.ruled.length) {
      ctx.strokeStyle = "rgba(90,140,200,.35)";
      ctx.lineWidth = Math.max(0.12, 0.6 / this.scale);
      ctx.beginPath();
      for (const [x0, y0, x1, y1] of this.ruled) {
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
      }
      ctx.stroke();
    }

    if (this.ink && !this.anim) {
      if (this.plot) {
        // これから描く線は鉛筆の薄い下書きのように
        this._drawBatch(ctx, this.ink.all, "rgba(70,66,58,.17)", this._minW);
      } else {
        ctx.globalAlpha = this.inkAlpha;
        this._drawBatch(ctx, this.ink.all, css("--ink") || "#26252a", this._minW);
        ctx.globalAlpha = 1;
      }
    }
    if (this.draft && this.showDraft) this._drawDraft(ctx);
  }

  _drawTip(ctx, prog) {
    const tip = this._tipAt(prog);
    if (!tip) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const [sx, sy] = this.toScreen(tip[0], tip[1]);
    const accent = css("--accent") || "#e0442a";
    ctx.strokeStyle = accent;
    ctx.fillStyle = accent;
    ctx.globalAlpha = 0.18;
    ctx.beginPath();
    ctx.arc(sx, sy, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(sx, sy, 4, 0, Math.PI * 2);
    ctx.moveTo(sx - 14, sy);
    ctx.lineTo(sx - 7, sy);
    ctx.moveTo(sx + 7, sy);
    ctx.lineTo(sx + 14, sy);
    ctx.moveTo(sx, sy - 14);
    ctx.lineTo(sx, sy - 7);
    ctx.moveTo(sx, sy + 7);
    ctx.lineTo(sx, sy + 14);
    ctx.stroke();
  }

  _drawDraft(ctx) {
    const d = this.draft;
    const pencil = css("--pencil") || "#4f7fd0";
    ctx.save();
    ctx.globalAlpha = this.ink ? 0.9 : 0.75;
    ctx.strokeStyle = pencil;
    ctx.lineWidth = Math.max(0.1, 0.8 / this.scale);
    ctx.beginPath();
    for (const [x0, y0, x1, y1] of d.rules) {
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
    }
    ctx.stroke();
    ctx.setLineDash([0.8, 0.6]);
    for (const [x, y, w, h] of d.math) {
      ctx.strokeRect(x, y, w, h);
    }
    ctx.setLineDash([]);
    // 文字はスクリーン座標で描く（Y 反転を戻す）
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = pencil;
    ctx.textBaseline = "alphabetic";
    const family = css("--font-draft") || "sans-serif";
    let lastSize = 0;
    for (const [ch, x, y, size, slant] of d.chars) {
      const px = size * this.scale;
      if (px < 1.2) continue;
      if (size !== lastSize) {
        ctx.font = `400 ${px.toFixed(2)}px ${family}`;
        lastSize = size;
      }
      const base = y + (this.lineSpacing - size) / 2 + size * 0.12;
      const [sx, sy] = this.toScreen(x, base);
      if (slant) {
        ctx.save();
        ctx.translate(sx + px / 2, sy - px / 2);
        ctx.rotate(-slant);
        ctx.fillText(ch, -px / 2, px / 2);
        ctx.restore();
      } else {
        ctx.fillText(ch, sx, sy);
      }
    }
    ctx.font = `500 ${Math.max(9, 2.2 * this.scale).toFixed(1)}px ${css("--font-mono") || "monospace"}`;
    for (const [x, y, w, h, , source] of d.math) {
      const [sx, sy] = this.toScreen(x, y + h);
      const label = source.length > 40 ? `${source.slice(0, 39)}…` : source;
      ctx.save();
      ctx.beginPath();
      ctx.rect(sx, sy, w * this.scale, h * this.scale);
      ctx.clip();
      ctx.fillText(label, sx + 2, sy + Math.min(h * this.scale - 2, 2.6 * this.scale + 2));
      ctx.restore();
    }
    ctx.restore();
    this._paperTransform(ctx);
  }

  _drawBatch(ctx, batch, color, minW) {
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(INK_MAX_MM, minW);
    ctx.stroke(batch.full);
    for (const b of batch.buckets) {
      ctx.lineWidth = Math.max(b.width, minW * 0.8);
      ctx.stroke(b.path);
    }
  }

  _drawStroke(ctx, s, upto, minW) {
    // upto: 0..1 の描画済み割合（弧長基準）
    if (upto >= 1) {
      ctx.lineWidth = Math.max(INK_MAX_MM, minW);
      ctx.stroke(s.full);
      for (const [x0, y0, x1, y1, w] of s.tapers) {
        ctx.lineWidth = Math.max(w, minW * 0.8);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }
      return null;
    }
    const pts = s.pts;
    const target = s.length * upto;
    let acc = 0;
    for (let i = 0; i + 3 < pts.length; i += 2) {
      const seg = Math.hypot(pts[i + 2] - pts[i], pts[i + 3] - pts[i + 1]);
      if (acc >= target) break;
      const k = seg > 0 ? Math.min(1, (target - acc) / seg) : 1;
      const c = s.uniform ? s.contact : s.contact[i / 2];
      ctx.lineWidth = Math.max(widthOf(c), minW);
      ctx.beginPath();
      ctx.moveTo(pts[i], pts[i + 1]);
      ctx.lineTo(pts[i] + (pts[i + 2] - pts[i]) * k, pts[i + 1] + (pts[i + 3] - pts[i + 1]) * k);
      ctx.stroke();
      acc += seg;
    }
    return null;
  }

  _pointAt(s, upto) {
    const pts = s.pts;
    const target = s.length * upto;
    let acc = 0;
    for (let i = 0; i + 3 < pts.length; i += 2) {
      const seg = Math.hypot(pts[i + 2] - pts[i], pts[i + 3] - pts[i + 1]);
      if (acc + seg >= target) {
        const k = seg > 0 ? (target - acc) / seg : 0;
        return [pts[i] + (pts[i + 2] - pts[i]) * k, pts[i + 1] + (pts[i + 3] - pts[i + 1]) * k];
      }
      acc += seg;
    }
    return [pts[pts.length - 2], pts[pts.length - 1]];
  }

  _drawRange(ctx, from, to, color, minW) {
    ctx.strokeStyle = color;
    let i = from;
    while (i < to) {
      if (i % CHUNK === 0 && i + CHUNK <= to) {
        this._drawBatch(ctx, this.ink.chunks[i / CHUNK], color, minW);
        i += CHUNK;
      } else {
        this._drawStroke(ctx, this.ink.strokes[i], 1, minW);
        i += 1;
      }
    }
  }

  _plotSplit(line) {
    // 送信済み行数 → 完了ストローク数＋現在画の進み具合
    const { spans, ends } = this.ink;
    let lo = 0;
    let hi = ends.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (ends[mid] <= line) lo = mid + 1;
      else hi = mid;
    }
    let partial = 0;
    if (lo < spans.length) {
      const [s, e] = spans[lo];
      if (line > s && e > s) partial = (line - s) / (e - s);
    }
    return { done: lo, partial };
  }

  _tipAt({ done, partial }) {
    if (done === 0 && partial === 0) return null;
    const strokes = this.ink.strokes;
    if (done < strokes.length && partial > 0) return this._pointAt(strokes[done], partial);
    const last = strokes[Math.min(done, strokes.length) - 1];
    return last ? [last.pts[last.pts.length - 2], last.pts[last.pts.length - 1]] : null;
  }

  /** ページの縮小サムネイル（data URL）。 */
  static thumbnail(page, paper, width = 84) {
    const prepared = (page._prepared ||= preparePage(page));
    const h = Math.round((width * paper.height) / paper.width);
    const canvas = document.createElement("canvas");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fbfaf6";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const s = (width / paper.width) * dpr;
    ctx.setTransform(s, 0, 0, -s, 0, paper.height * s);
    ctx.strokeStyle = "#2a2926";
    ctx.lineCap = "round";
    ctx.lineWidth = 0.5;
    ctx.stroke(prepared.all.full);
    for (const b of prepared.all.buckets) ctx.stroke(b.path);
    return canvas.toDataURL();
  }
}
