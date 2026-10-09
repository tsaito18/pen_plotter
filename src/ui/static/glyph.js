// 字形の描画（収集サンプルの縮小表示・お手本の書き順アニメーション）。

const css = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

/** CSS サイズに合わせて解像度を整えた 2D コンテキスト（CSS px 座標）。 */
export function fitCanvas(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const w = Math.max(1, Math.round(r.width));
  const h = Math.max(1, Math.round(r.height));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  return { ctx, w, h };
}

const toXY = (p) => (Array.isArray(p) ? p : [p.x, p.y]);

/**
 * 画の列を枠に収めて描く。
 * @param {Array} strokes 点は {x,y} か [x,y]。
 * @param {{box?: number[], minSpan?: number, yUp?: boolean, color?: string, width?: number, margin?: number}} opts
 *   box=[x0,y0,x1,y1] を渡すと書いた位置・大きさのまま（無ければ外接矩形に合わせる）。
 *   minSpan は外接矩形に合わせるときの最小の幅（座標単位）。
 */
export function drawStrokes(canvas, strokes, opts = {}) {
  const { ctx, w, h } = fitCanvas(canvas);
  const pts = strokes.flat().map(toXY);
  if (!pts.length) return;
  let [x0, y0, x1, y1] = opts.box || [
    Math.min(...pts.map((p) => p[0])),
    Math.min(...pts.map((p) => p[1])),
    Math.max(...pts.map((p) => p[0])),
    Math.max(...pts.map((p) => p[1])),
  ];
  // 「ー」「、」のような小さな字が枠いっぱいに膨らまないよう、最小の幅を取る
  if (opts.minSpan) {
    const grow = (lo, hi) => {
      const pad = Math.max(0, opts.minSpan - (hi - lo)) / 2;
      return [lo - pad, hi + pad];
    };
    [x0, x1] = grow(x0, x1);
    [y0, y1] = grow(y0, y1);
  }
  const span = Math.max(x1 - x0, y1 - y0, 1e-6);
  const m = (opts.margin ?? 0.12) * Math.min(w, h);
  const k = (Math.min(w, h) - m * 2) / span;
  const ox = (w - (x1 - x0) * k) / 2;
  const oy = (h - (y1 - y0) * k) / 2;
  const map = ([x, y]) => [ox + (x - x0) * k, opts.yUp ? oy + (y1 - y) * k : oy + (y - y0) * k];
  ctx.strokeStyle = opts.color || css("--paper-ink", "#1d1c19");
  ctx.lineWidth = opts.width || Math.max(1.2, Math.min(w, h) / 48);
  for (const s of strokes) {
    const p = s.map(toXY).map(map);
    ctx.beginPath();
    p.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (p.length === 1) ctx.lineTo(p[0][0] + 0.1, p[0][1]);
    ctx.stroke();
  }
}

/** 平らな座標列 [x0,y0,x1,y1,...] の画（スタジオの清書・Y-UP）を [[x,y],...] に。 */
export const unflatten = (flat) => {
  const out = [];
  for (let i = 0; i + 1 < flat.length; i += 2) out.push([flat[i], flat[i + 1]]);
  return out;
};

/**
 * お手本（KanjiVG・Y-UP 0..10）の書き順アニメーション。
 * 1 画ずつ朱で描いて墨に落ち着かせ、始筆に番号を添える。
 */
export class StrokeOrder {
  constructor(canvas) {
    this.canvas = canvas;
    this.strokes = [];
    this.raf = 0;
    new ResizeObserver(() => this.draw(1)).observe(canvas);
  }

  set(strokes) {
    this.strokes = strokes || [];
    this.play();
  }

  play() {
    cancelAnimationFrame(this.raf);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || !this.strokes.length) {
      this.draw(1);
      return;
    }
    const per = 520;
    const total = this.strokes.length * per;
    let start = null;
    const step = (t) => {
      start ??= t;
      const x = Math.min((t - start) / total, 1);
      this.draw(x);
      if (x < 1) this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  draw(progress) {
    const { ctx, w, h } = fitCanvas(this.canvas);
    const n = this.strokes.length;
    if (!n) return;
    const size = Math.min(w, h);
    const m = size * 0.1;
    const k = (size - m * 2) / 10;
    const ox = (w - size) / 2 + m;
    const oy = (h - size) / 2 + m;
    const map = (p) => [ox + p.x * k, oy + (10 - p.y) * k];
    const ink = css("--paper-ink", "#1d1c19");
    const accent = css("--accent", "#e0442a");
    const exact = progress * n;
    ctx.lineWidth = Math.max(2, size / 34);
    // まだの画は薄く
    ctx.strokeStyle = css("--line-strong", "rgba(0,0,0,.15)");
    for (const s of this.strokes) this._path(ctx, s.map(map), 1);
    this.strokes.forEach((s, i) => {
      const part = Math.min(Math.max(exact - i, 0), 1);
      if (part <= 0) return;
      ctx.strokeStyle = part < 1 ? accent : ink;
      this._path(ctx, s.map(map), part);
    });
    // 書き順の番号
    const r = Math.max(6, (size / 22) * (n > 8 ? 0.78 : 1));
    ctx.font = `600 ${Math.round(r * 1.15)}px ${css("--font-mono", "monospace")}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    this.strokes.forEach((s, i) => {
      if (!s.length || exact < i) return;
      const [x, y] = map(s[0]);
      ctx.fillStyle = i < Math.floor(exact) || progress >= 1 ? ink : accent;
      ctx.beginPath();
      ctx.arc(x - r * 0.9, y - r * 0.9, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = css("--surface-2", "#fff");
      ctx.fillText(String(i + 1), x - r * 0.9, y - r * 0.9 + 0.5);
    });
  }

  _path(ctx, pts, part) {
    if (pts.length < 2) return;
    const lens = [0];
    for (let i = 1; i < pts.length; i += 1) {
      lens.push(lens[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    }
    const target = lens[lens.length - 1] * part;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i += 1) {
      if (lens[i] <= target) {
        ctx.lineTo(pts[i][0], pts[i][1]);
      } else {
        const seg = lens[i] - lens[i - 1] || 1;
        const t = (target - lens[i - 1]) / seg;
        ctx.lineTo(pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t);
        break;
      }
    }
    ctx.stroke();
  }
}
