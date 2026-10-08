// WebSerial で xDraw A4（GRBL 互換 DrawCore）へ G-code を流す。
//
// - 送信は 1 行送って ok を待つ単純なストップ＆ウェイト（GRBL の推奨最小プロトコル）。
// - 一時停止は「次にペンを上げた直後」で止める（紙にペンを押し付けたまま止めない）。
// - 停止はペンを上げてから終わる。緊急停止は feed hold + soft reset（位置を失うので要ホーミング）。
// - 複数ページのジョブはページ間で用紙交換を待つ。

const BAUD_RATE = 115200;
const DEFAULT_TIMEOUT_MS = 30000;
const HOME_TIMEOUT_MS = 120000;
const BOOT_WAIT_MS = 2500;
const CH340 = { usbVendorId: 0x1a86 };

// src/gcode/config.py の PlotterConfig と同じ値
export const PEN_UP = "G1G90 Z0.5 F5000";
export const PEN_DOWN = "G1G90 Z5.0 F5000";
export const HOMING = ["$H", "G4 P1", "G92 X0 Y297 Z0", "G90"];
// Z がこの値以上なら紙に接触（終端リフト 2.6・連綿も描画扱い、ペンアップ 0.5 は移動）
export const PEN_CONTACT_Z = 1.5;

export const isSupported = () => "serial" in navigator && window.isSecureContext;

export function unsupportedReason() {
  if (!("serial" in navigator)) return "browser";
  if (!window.isSecureContext) return "insecure";
  return null;
}

/** コメント・空行を除いた送信行。 */
export function normalizeGcode(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => line.replace(/;.*$/, "").replace(/\([^)]*\)/g, "").trim())
    .filter((line) => line.length > 0);
}

const num = (line, axis) => {
  const m = line.match(new RegExp(`${axis}(-?\\d+\\.?\\d*)`));
  return m ? parseFloat(m[1]) : null;
};

/**
 * G-code 行から描画ストロークと各ストロークの行範囲を復元する（アップロード G-code 用）。
 * @returns {{strokes: {points:number[], contact:number}[], spans: [number, number][]}}
 */
export function parseGcode(lines) {
  const strokes = [];
  const spans = [];
  let x = 0;
  let y = 0;
  let z = 0.5;
  let current = null;
  const close = () => {
    if (current && current.points.length >= 4) {
      strokes.push({ points: current.points, contact: 1, finish: "none" });
      spans.push([current.start, current.end]);
    }
    current = null;
  };
  lines.forEach((line, i) => {
    if (!/^G[01](?!\d)/.test(line)) return;
    const nz = num(line, "Z");
    if (nz !== null) z = nz;
    const nx = num(line, "X");
    const ny = num(line, "Y");
    if (nx === null && ny === null) {
      if (z < PEN_CONTACT_Z) close();
      return;
    }
    const tx = nx ?? x;
    const ty = ny ?? y;
    if (/^G1/.test(line) && z >= PEN_CONTACT_Z) {
      if (!current) current = { points: [x, y], start: i, end: i };
      current.points.push(tx, ty);
      current.end = i + 1;
    } else {
      close();
    }
    x = tx;
    y = ty;
  });
  close();
  return { strokes, spans };
}

/** 各行の所要時間の見積もり（秒）。加減速は無視するので実測で補正して使う。 */
export function estimateLineSeconds(lines) {
  let x = 0;
  let y = 0;
  let z = 0.5;
  let feed = 6000;
  return lines.map((line) => {
    if (line.startsWith("$H")) return 12;
    const g4 = line.match(/^G4\s*P(\d+\.?\d*)/);
    if (g4) return parseFloat(g4[1]);
    if (!/^G[01](?!\d)/.test(line)) return 0.01;
    const f = num(line, "F");
    if (f) feed = f;
    const nx = num(line, "X") ?? x;
    const ny = num(line, "Y") ?? y;
    const nz = num(line, "Z") ?? z;
    const dist = Math.hypot(nx - x, ny - y, nz - z);
    const rate = /^G0(?!\d)/.test(line) ? Math.max(feed, 12000) : feed;
    x = nx;
    y = ny;
    z = nz;
    return 0.012 + (dist / rate) * 60;
  });
}

const sum = (arr, from = 0, to = arr.length) => {
  let s = 0;
  for (let i = from; i < to; i += 1) s += arr[i];
  return s;
};

/**
 * プロッタ接続と送信ジョブ。状態変化は "change"、ログは "log"、進捗は "progress" イベント。
 *
 * status: unsupported | disconnected | connecting | idle | busy | streaming | pausing | paused
 *         | paper（用紙交換待ち）
 */
export class Plotter extends EventTarget {
  constructor() {
    super();
    this.port = null;
    this.reader = null;
    this.writer = null;
    this.decoder = new TextDecoder();
    this.encoder = new TextEncoder();
    this.buffer = "";
    this.lineWaiters = [];
    this.status = isSupported() ? "disconnected" : "unsupported";
    this.job = null;
    this.progress = null;
    this.needsHoming = false;
    this.lastLine = "";
    this._resume = null;
    this._cancel = false;
    this._pauseRequested = false;
    if (isSupported()) {
      navigator.serial.addEventListener("disconnect", (event) => {
        if (event.target === this.port) {
          this.log("USB ケーブルが外れました", "error");
          this._teardown();
        }
      });
    }
  }

  // ---------------------------------------------------------------- 状態

  _set(status) {
    this.status = status;
    this.dispatchEvent(new Event("change"));
  }

  log(message, level = "info") {
    this.dispatchEvent(new CustomEvent("log", { detail: { message, level, time: new Date() } }));
  }

  get connected() {
    return !["unsupported", "disconnected", "connecting"].includes(this.status);
  }

  get running() {
    return ["streaming", "pausing", "paused", "paper"].includes(this.status);
  }

  // ---------------------------------------------------------------- 接続

  /** 以前に許可したポートがあれば、その情報（自動再接続の候補）。 */
  async knownPort() {
    if (!isSupported()) return null;
    const ports = await navigator.serial.getPorts();
    return ports[0] || null;
  }

  async connect({ anyPort = false, port = null } = {}) {
    if (!isSupported() || this.connected) return false;
    this._set("connecting");
    try {
      const target =
        port || (await navigator.serial.requestPort(anyPort ? {} : { filters: [CH340] }));
      await target.open({ baudRate: BAUD_RATE });
      this.port = target;
      this.reader = target.readable.getReader();
      this.writer = target.writable.getWriter();
      this.buffer = "";
      this._readLoop();
      const banner = await this._waitBanner();
      this.log(banner ? `接続しました — ${banner}` : `接続しました（${BAUD_RATE}bps）`, "ok");
      this.needsHoming = true;
      this._set("idle");
      return true;
    } catch (error) {
      if (error && error.name === "NotFoundError") {
        this.log("ポートの選択がキャンセルされました");
      } else {
        this.log(`接続できませんでした: ${error.message || error}`, "error");
      }
      await this._teardown();
      return false;
    }
  }

  async disconnect() {
    if (this.running) return;
    await this._teardown();
    this.log("切断しました");
  }

  async _teardown() {
    this._cancel = true;
    this._settleResume(false);
    for (const waiter of this.lineWaiters.splice(0)) waiter.reject(new Error("切断されました"));
    try {
      await this.reader?.cancel();
    } catch {
      // 既に閉じている
    }
    try {
      this.reader?.releaseLock();
      this.writer?.releaseLock();
    } catch {
      // ロック解放済み
    }
    try {
      await this.port?.close();
    } catch {
      // 既に閉じている
    }
    this.port = null;
    this.reader = null;
    this.writer = null;
    this.job = null;
    this.progress = null;
    this._set(isSupported() ? "disconnected" : "unsupported");
  }

  async _readLoop() {
    const reader = this.reader;
    try {
      while (reader === this.reader) {
        const { value, done } = await reader.read();
        if (done) break;
        this.buffer += this.decoder.decode(value, { stream: true });
        let idx;
        while ((idx = this.buffer.search(/\r?\n/)) >= 0) {
          const line = this.buffer.slice(0, idx).trim();
          this.buffer = this.buffer.slice(idx + 1);
          if (line) this._onLine(line);
        }
      }
    } catch (error) {
      if (reader === this.reader) this.log(`受信エラー: ${error.message || error}`, "error");
    }
  }

  _onLine(line) {
    this.lastLine = line;
    const waiter = this.lineWaiters.shift();
    if (waiter) waiter.resolve(line);
    else if (!/^ok$/i.test(line)) this.log(`GRBL: ${line}`);
  }

  _nextLine(timeoutMs) {
    return new Promise((resolve, reject) => {
      const waiter = { resolve: null, reject };
      const timer = window.setTimeout(() => {
        const i = this.lineWaiters.indexOf(waiter);
        if (i >= 0) this.lineWaiters.splice(i, 1);
        reject(new Error(`応答がありません（${Math.round(timeoutMs / 1000)}秒）`));
      }, timeoutMs);
      waiter.resolve = (line) => {
        window.clearTimeout(timer);
        resolve(line);
      };
      waiter.reject = (error) => {
        window.clearTimeout(timer);
        reject(error);
      };
      this.lineWaiters.push(waiter);
    });
  }

  async _waitBanner() {
    // ポートを開くとコントローラが再起動し、起動メッセージ（Grbl ...）を出す
    const deadline = Date.now() + BOOT_WAIT_MS;
    while (Date.now() < deadline) {
      try {
        const line = await this._nextLine(deadline - Date.now());
        if (/grbl|drawcore/i.test(line)) return line;
      } catch {
        return null;
      }
    }
    return null;
  }

  async send(line, timeoutMs = DEFAULT_TIMEOUT_MS) {
    if (!this.writer) throw new Error("プロッタが接続されていません");
    const response = this._waitOk(line.startsWith("$H") ? HOME_TIMEOUT_MS : timeoutMs);
    await this.writer.write(this.encoder.encode(`${line}\n`));
    return response;
  }

  async _waitOk(timeoutMs) {
    for (;;) {
      const response = await this._nextLine(timeoutMs);
      if (/^ok$/i.test(response)) return response;
      if (/^(error|alarm):/i.test(response)) throw new Error(response);
      if (!response.startsWith("<")) this.log(`GRBL: ${response}`);
    }
  }

  // ---------------------------------------------------------------- 単発操作

  async _machine(label, lines) {
    if (this.status !== "idle") return false;
    this._set("busy");
    try {
      for (const line of lines) await this.send(line);
      this.log(label, "ok");
      return true;
    } catch (error) {
      this.log(`${label}に失敗: ${error.message || error}`, "error");
      return false;
    } finally {
      if (this.status === "busy") this._set("idle");
    }
  }

  async home() {
    const ok = await this._machine("原点復帰しました", HOMING);
    if (ok) this.needsHoming = false;
    this.dispatchEvent(new Event("change"));
    return ok;
  }

  penUp() {
    return this._machine("ペンを上げました", [PEN_UP]);
  }

  penDown() {
    return this._machine("ペンを下ろしました", [PEN_DOWN]);
  }

  // ---------------------------------------------------------------- ジョブ

  /**
   * @param {{name: string, pages: {pageNo: number, lines: string[]}[]}} job
   */
  async run(job) {
    if (this.status !== "idle") return;
    const pages = job.pages.filter((p) => p.lines.length > 0);
    if (pages.length === 0) {
      this.log("送信できる G-code がありません", "warn");
      return;
    }
    const estimates = pages.map((p) => estimateLineSeconds(p.lines));
    const totalLines = pages.reduce((n, p) => n + p.lines.length, 0);
    const totalEstimate = estimates.reduce((n, e) => n + sum(e), 0);
    this.job = { ...job, pages };
    this._cancel = false;
    this._pauseRequested = false;
    let sentTotal = 0;
    let doneEstimate = 0;
    let activeMs = 0;
    let lastTick = performance.now();
    const tick = () => {
      const now = performance.now();
      if (this.status === "streaming" || this.status === "pausing") activeMs += now - lastTick;
      lastTick = now;
    };
    const report = (pageIndex, line) => {
      tick();
      // 見積もりと実測の比で残り時間を補正する（序盤は見積もりのまま）
      const ratio = doneEstimate > 20 ? Math.min(Math.max(activeMs / 1000 / doneEstimate, 0.5), 4) : 1.25;
      this.progress = {
        pageIndex,
        pageNo: pages[pageIndex].pageNo,
        pageCount: pages.length,
        line,
        pageLines: pages[pageIndex].lines.length,
        sent: sentTotal,
        total: totalLines,
        remaining: Math.max(totalEstimate - doneEstimate, 0) * ratio,
      };
      this.dispatchEvent(new Event("progress"));
    };

    this._set("streaming");
    this.log(`描画を開始: ${job.name}（${pages.length}ページ・${totalLines.toLocaleString()}行）`);
    let outcome = "done";
    try {
      for (let p = 0; p < pages.length; p += 1) {
        const { lines, pageNo } = pages[p];
        report(p, 0);
        for (let i = 0; i < lines.length; i += 1) {
          if (this._cancel) throw new CancelledError();
          await this.send(lines[i]);
          sentTotal += 1;
          doneEstimate += estimates[p][i];
          report(p, i + 1);
          if (this._pauseRequested && this._isPenUp(lines[i])) {
            tick();
            this._set("paused");
            this.log("一時停止中（ペンは上がっています）");
            const resumed = await this._waitResume();
            lastTick = performance.now();
            if (!resumed) throw new CancelledError();
            this._set("streaming");
            this.log("再開しました");
          }
        }
        if (p < pages.length - 1) {
          tick();
          this._set("paper");
          this.log(`ページ ${pageNo} を描き終えました。用紙を交換してください`, "ok");
          this.dispatchEvent(new CustomEvent("paper", { detail: { done: pageNo, next: pages[p + 1].pageNo } }));
          const resumed = await this._waitResume();
          lastTick = performance.now();
          if (!resumed) throw new CancelledError();
          this._set("streaming");
        }
      }
      this.log("描画が完了しました", "ok");
    } catch (error) {
      if (error instanceof CancelledError) {
        outcome = "stopped";
        this.log(`停止しました（${sentTotal.toLocaleString()} / ${totalLines.toLocaleString()} 行）`, "warn");
        if (this.writer && !this._emergency) {
          try {
            await this.send(PEN_UP);
          } catch {
            // 応答が無くても停止は完了扱い
          }
        }
      } else {
        outcome = "error";
        this.log(`送信エラー: ${error.message || error}`, "error");
      }
    } finally {
      this._emergency = false;
      this._pauseRequested = false;
      this._resume = null;
      const finished = this.job;
      this.job = null;
      if (this.port) this._set("idle");
      this.dispatchEvent(new CustomEvent("finish", { detail: { outcome, job: finished } }));
    }
  }

  _isPenUp(line) {
    const z = num(line, "Z");
    return z !== null && z < PEN_CONTACT_Z && num(line, "X") === null && num(line, "Y") === null;
  }

  _waitResume() {
    return new Promise((resolve) => {
      this._resume = resolve;
    });
  }

  _settleResume(value) {
    const resolve = this._resume;
    this._resume = null;
    if (resolve) resolve(value);
  }

  pause() {
    if (this.status !== "streaming") return;
    this._pauseRequested = true;
    this._set("pausing");
    this.log("ペンを上げたところで一時停止します");
  }

  resume() {
    if (this.status === "pausing") {
      this._pauseRequested = false;
      this._set("streaming");
      return;
    }
    if (this.status === "paused" || this.status === "paper") {
      this._pauseRequested = false;
      this._settleResume(true);
    }
  }

  stop() {
    if (!this.running) return;
    this._cancel = true;
    this._settleResume(false);
  }

  async emergencyStop() {
    if (!this.writer) return;
    this._cancel = true;
    this._emergency = true;
    this._settleResume(false);
    try {
      // feed hold → soft reset（GRBL のリアルタイムコマンド）
      await this.writer.write(new Uint8Array([0x21, 0x18]));
      this.log("緊急停止しました。再開する前に原点復帰してください", "error");
    } catch (error) {
      this.log(`緊急停止に失敗: ${error.message || error}`, "error");
    }
    for (const waiter of this.lineWaiters.splice(0)) waiter.reject(new CancelledError());
    this.needsHoming = true;
    this.dispatchEvent(new Event("change"));
  }
}

class CancelledError extends Error {
  constructor() {
    super("cancelled");
  }
}
