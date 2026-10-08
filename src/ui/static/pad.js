// 書き込みパッド（Apple Pencil 前提）。
//
// - 座標は 512×512 の論理座標（Y-DOWN）で記録する（既存サンプル・異常検出の基準と同じ）。
// - 記録する点は pointermove の粒度（既存の学習データと揃える）。画面の線だけは
//   getCoalescedEvents の中間点で滑らかにし、getPredictedEvents で遅れを隠す。
// - ペンを一度でも検出したら指・手のひらは無視（パームリジェクション）。二本指タップで 1 画戻す。

export const LOGICAL = 512;
const INK = "#1d1c19";

const width = (pressure) => 1.4 + Math.min(Math.max(pressure, 0), 1) * 4.2;

export class WritingPad extends EventTarget {
  /**
   * @param {HTMLElement} root .pad 要素（中に canvas を 2 枚作る）
   */
  constructor(root) {
    super();
    this.root = root;
    this.ink = document.createElement("canvas");
    this.live = document.createElement("canvas");
    this.ink.className = "pad-ink";
    this.live.className = "pad-live";
    root.append(this.ink, this.live);
    this.strokes = [];
    this.current = null;
    this.penSeen = false;
    this.allowTouch = false;
    this.ghost = null;
    this._touches = new Map();
    this._bind();
    new ResizeObserver(() => this._resize()).observe(root);
    this._resize();
  }

  get count() {
    return this.strokes.length;
  }

  get empty() {
    return this.strokes.length === 0;
  }

  /** 保存用の画（論理座標）。 */
  data() {
    return this.strokes.map((s) => s.points.map((p) => ({ ...p })));
  }

  undo() {
    if (!this.strokes.length) return;
    this.strokes.pop();
    this._redraw();
    this._changed();
  }

  clear() {
    this.strokes = [];
    this.current = null;
    this._redraw();
    this._changed();
  }

  /** お手本を薄く敷く（KanjiVG・Y-UP 0..10）。null で消す。 */
  setGhost(strokes) {
    this.ghost = strokes && strokes.length ? strokes : null;
    this._redraw();
  }

  snapshot() {
    return this.ink.toDataURL();
  }

  // ---------------------------------------------------------------- 入力

  _accept(e) {
    if (e.pointerType === "pen") {
      this.penSeen = true;
      return true;
    }
    if (e.pointerType === "mouse") return !this.penSeen;
    return this.allowTouch && !this.penSeen;
  }

  _point(e) {
    const r = this.ink.getBoundingClientRect();
    const k = LOGICAL / r.width;
    return {
      x: Math.round((e.clientX - r.left) * k * 10) / 10,
      y: Math.round((e.clientY - r.top) * k * 10) / 10,
      pressure: e.pointerType === "pen" ? e.pressure || 0.5 : 0.5,
      timestamp: Math.round(e.timeStamp),
    };
  }

  _bind() {
    const el = this.live;
    el.addEventListener("pointerdown", (e) => {
      if (e.pointerType === "touch") this._touchDown(e);
      if (!this._accept(e) || this.current) return;
      e.preventDefault();
      try {
        el.setPointerCapture(e.pointerId);
      } catch {
        // 合成イベントなど capture できないポインタでも書ける
      }
      this.current = { id: e.pointerId, points: [this._point(e)], smooth: [this._point(e)] };
      this.dispatchEvent(new Event("strokestart"));
      this._drawLive([]);
    });
    el.addEventListener("pointermove", (e) => {
      if (e.pointerType === "touch") this._touchMove(e);
      if (!this.current || e.pointerId !== this.current.id) return;
      e.preventDefault();
      this.current.points.push(this._point(e));
      // 中間点が取れない環境（合成イベント・一部ブラウザ）ではイベント自体を使う
      const coalesced = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
      for (const c of coalesced.length ? coalesced : [e]) this.current.smooth.push(this._point(c));
      const predicted = e.getPredictedEvents ? e.getPredictedEvents().map((p) => this._point(p)) : [];
      this._drawLive(predicted);
    });
    const end = (e) => {
      if (e.pointerType === "touch") this._touchUp(e);
      if (!this.current || e.pointerId !== this.current.id) return;
      const stroke = this.current;
      this.current = null;
      if (stroke.points.length) {
        this.strokes.push(stroke);
        this._paintStroke(this.ink.getContext("2d"), stroke.smooth);
      }
      this._clearLive();
      this._changed();
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    el.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  // 二本指タップ → 1 画戻す（指は描画に使わない前提）
  _touchDown(e) {
    this._touches.set(e.pointerId, { x: e.clientX, y: e.clientY, t: e.timeStamp, moved: false });
    if (this._touches.size === 2) this._twoFinger = e.timeStamp;
  }

  _touchMove(e) {
    const t = this._touches.get(e.pointerId);
    if (t && Math.hypot(e.clientX - t.x, e.clientY - t.y) > 12) t.moved = true;
  }

  _touchUp(e) {
    const t = this._touches.get(e.pointerId);
    this._touches.delete(e.pointerId);
    if (this._twoFinger && t && !t.moved && e.timeStamp - this._twoFinger < 350) {
      this._twoFinger = null;
      if (this.penSeen || !this.allowTouch) {
        this.undo();
        this.dispatchEvent(new Event("gestureundo"));
      }
    }
    if (this._touches.size === 0) this._twoFinger = null;
  }

  _changed() {
    this.dispatchEvent(new Event("change"));
  }

  // ---------------------------------------------------------------- 描画

  _resize() {
    const size = this.root.getBoundingClientRect().width;
    if (!size) return;
    this.dpr = Math.min(window.devicePixelRatio || 1, 3);
    for (const c of [this.ink, this.live]) {
      c.width = Math.round(size * this.dpr);
      c.height = Math.round(size * this.dpr);
    }
    this.scale = (size * this.dpr) / LOGICAL;
    this._redraw();
  }

  _ctx(canvas) {
    const ctx = canvas.getContext("2d");
    ctx.setTransform(this.scale, 0, 0, this.scale, 0, 0);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    return ctx;
  }

  _redraw() {
    if (!this.scale) return;
    const ctx = this._ctx(this.ink);
    ctx.clearRect(0, 0, LOGICAL, LOGICAL);
    if (this.ghost) this._paintGhost(ctx);
    for (const s of this.strokes) this._paintStroke(ctx, s.smooth);
  }

  _paintGhost(ctx) {
    // KanjiVG（Y-UP 0..10）をマスの 84% に収める
    const pad = LOGICAL * 0.08;
    const k = (LOGICAL - pad * 2) / 10;
    ctx.save();
    ctx.strokeStyle = "rgba(74,127,212,.22)";
    ctx.lineWidth = 9;
    for (const s of this.ghost) {
      ctx.beginPath();
      s.forEach((p, i) => {
        const x = pad + p.x * k;
        const y = pad + (10 - p.y) * k;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    ctx.restore();
  }

  /** 筆圧で太さが変わる線。中点を結ぶ 2 次曲線で滑らかにする。 */
  _paintStroke(ctx, pts) {
    ctx.strokeStyle = INK;
    ctx.fillStyle = INK;
    if (pts.length === 1) {
      ctx.beginPath();
      ctx.arc(pts[0].x, pts[0].y, width(pts[0].pressure) / 2, 0, Math.PI * 2);
      ctx.fill();
      return;
    }
    for (let i = 1; i < pts.length; i += 1) {
      const a = pts[i - 1];
      const b = pts[i];
      const prev = pts[i - 2] || a;
      ctx.lineWidth = width((a.pressure + b.pressure) / 2);
      ctx.beginPath();
      ctx.moveTo((prev.x + a.x) / 2, (prev.y + a.y) / 2);
      ctx.quadraticCurveTo(a.x, a.y, (a.x + b.x) / 2, (a.y + b.y) / 2);
      ctx.stroke();
    }
    const last = pts[pts.length - 1];
    const before = pts[pts.length - 2];
    ctx.lineWidth = width(last.pressure);
    ctx.beginPath();
    ctx.moveTo((before.x + last.x) / 2, (before.y + last.y) / 2);
    ctx.lineTo(last.x, last.y);
    ctx.stroke();
  }

  _drawLive(predicted) {
    const ctx = this._ctx(this.live);
    ctx.clearRect(0, 0, LOGICAL, LOGICAL);
    if (!this.current) return;
    this._paintStroke(ctx, this.current.smooth.concat(predicted));
  }

  _clearLive() {
    const ctx = this._ctx(this.live);
    ctx.clearRect(0, 0, LOGICAL, LOGICAL);
  }
}
