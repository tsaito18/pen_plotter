(function () {
  "use strict";

  const BAUD_RATE = 115200;
  const DEFAULT_TIMEOUT_MS = 30000;
  const HOME_TIMEOUT_MS = 120000;
  const MAX_LOG_ENTRIES = 200;

  const state = {
    port: null,
    reader: null,
    writer: null,
    decoder: new TextDecoder(),
    encoder: new TextEncoder(),
    readBuffer: "",
    connected: false,
    streaming: false,
    cancelRequested: false,
    logEntries: [],
    currentLine: "",
    sent: 0,
    total: 0,
    lastResponse: "",
    paused: false,
    pausedResolve: null,
    preview: { segments: [], total: 0, pageNo: 1, name: "" },
    simTimer: null,
    bgImage: null,
    bgReady: false,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function setHtml(id, html) {
    const node = byId(id);
    if (node) {
      node.innerHTML = html;
    }
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function setButtonDisabled(elemId, disabled) {
    const root = byId(elemId);
    if (!root) {
      return;
    }
    const button = root.querySelector("button") || root;
    button.disabled = Boolean(disabled);
    button.setAttribute("aria-disabled", disabled ? "true" : "false");
  }

  function refreshButtons() {
    const unsupported = !("serial" in navigator) || !window.isSecureContext;
    const busy = state.streaming;
    setButtonDisabled("webserial-connect-btn", unsupported || state.connected || busy);
    setButtonDisabled("webserial-disconnect-btn", !state.connected || busy);
    setButtonDisabled("webserial-start-btn", unsupported || !state.connected || busy);
    setButtonDisabled("webserial-stop-btn", !busy);
    setButtonDisabled("webserial-emergency-btn", !state.connected);
    const resumeRoot = byId("webserial-resume-btn");
    if (resumeRoot) {
      // 続行ボタンは用紙交換ポーズ中のみ表示する
      resumeRoot.style.display = state.paused ? "" : "none";
    }
    setButtonDisabled("webserial-resume-btn", !state.paused);
  }

  function statusText() {
    if (!("serial" in navigator)) {
      return "非対応ブラウザ: Chrome / Edge のローカル表示で使用";
    }
    if (!window.isSecureContext) {
      return "Secure Context ではないため WebSerial 使用不可。localhost で開く";
    }
    if (state.paused) {
      return `用紙交換待ち ${state.sent} / ${state.total}`;
    }
    if (state.streaming) {
      return `送信中 ${state.sent} / ${state.total}`;
    }
    return state.connected ? "接続中" : "未接続";
  }

  function badgeState() {
    if (!("serial" in navigator) || !window.isSecureContext) {
      return { variant: "unsupported", label: "非対応" };
    }
    if (state.paused) {
      return { variant: "streaming", label: "用紙交換待ち" };
    }
    if (state.streaming) {
      return { variant: "streaming", label: "送信中" };
    }
    if (state.connected) {
      return { variant: "connected", label: "接続中" };
    }
    return { variant: "idle", label: "未接続" };
  }

  function refreshBadge() {
    const node = byId("webserial-status-badge");
    if (!node) {
      return;
    }
    const { variant, label } = badgeState();
    node.className = `pp-badge pp-badge--${variant}`;
    node.textContent = label;
  }

  function refreshStatus() {
    refreshBadge();
    setHtml(
      "webserial-status-value",
      escapeHtml(
        `${statusText()} | 現在行: ${state.currentLine || "-"} | 最終応答: ${
          state.lastResponse || "-"
        }`,
      ),
    );
    const pct = state.total > 0 ? Math.round((state.sent / state.total) * 100) : 0;
    const bar = byId("webserial-progress-bar");
    if (bar) {
      bar.style.width = `${pct}%`;
    }
    setHtml("webserial-progress-text", escapeHtml(`${state.sent} / ${state.total} 行 (${pct}%)`));
    const currentLine = byId("webserial-current-line");
    if (currentLine) {
      currentLine.textContent = `現在行: ${state.currentLine || "-"}`;
    }
    refreshButtons();
  }

  function log(message, level = "info") {
    const ts = new Date().toLocaleTimeString();
    state.logEntries.push({ ts, message, level });
    if (state.logEntries.length > MAX_LOG_ENTRIES) {
      state.logEntries = state.logEntries.slice(-MAX_LOG_ENTRIES);
    }
    const html = state.logEntries
      .map((entry) => {
        const color = entry.level === "error" ? "#a8071a" : entry.level === "warn" ? "#874d00" : "#222";
        return `<div style="color:${color};"><span style="color:#666;">[${escapeHtml(
          entry.ts,
        )}]</span> ${escapeHtml(entry.message)}</div>`;
      })
      .join("");
    setHtml("webserial-log-entries", html);
    refreshStatus();
  }

  function ensureSupported() {
    if (!("serial" in navigator)) {
      throw new Error("WebSerial 非対応。Windows Chrome / Edge で開く");
    }
    if (!window.isSecureContext) {
      throw new Error("WebSerial は Secure Context 必須。http://localhost で開く");
    }
  }

  function ensureConnected() {
    if (!state.connected || !state.port || !state.writer || !state.reader) {
      throw new Error("未接続");
    }
  }

  async function disconnectReaderWriter() {
    if (state.reader) {
      try {
        await state.reader.cancel();
      } catch (error) {
        // reader が reset 済みの場合は無視。
      }
      try {
        state.reader.releaseLock();
      } catch (error) {
        // lock 解放済みの場合は無視。
      }
      state.reader = null;
    }
    if (state.writer) {
      try {
        state.writer.releaseLock();
      } catch (error) {
        // lock 解放済みの場合は無視。
      }
      state.writer = null;
    }
  }

  async function connect() {
    try {
      ensureSupported();
      if (state.connected) {
        log("すでに接続中", "warn");
        return;
      }
      const port = await navigator.serial.requestPort();
      await port.open({ baudRate: BAUD_RATE });
      state.port = port;
      state.reader = port.readable.getReader();
      state.writer = port.writable.getWriter();
      state.readBuffer = "";
      state.connected = true;
      state.lastResponse = "";
      log(`接続完了 baud=${BAUD_RATE}`);
    } catch (error) {
      log(`接続失敗: ${error.message || error}`, "error");
    } finally {
      refreshStatus();
    }
  }

  async function disconnect() {
    try {
      if (!state.port) {
        log("未接続", "warn");
        return;
      }
      await disconnectReaderWriter();
      await state.port.close();
      log("切断完了");
    } catch (error) {
      log(`切断失敗: ${error.message || error}`, "error");
    } finally {
      state.port = null;
      state.connected = false;
      state.streaming = false;
      state.cancelRequested = false;
      state.currentLine = "";
      refreshStatus();
    }
  }

  function responseTimeout(ms) {
    return new Promise((_, reject) => {
      window.setTimeout(() => reject(new Error(`GRBL 応答タイムアウト ${ms}ms`)), ms);
    });
  }

  async function readLine(timeoutMs) {
    while (true) {
      const newline = state.readBuffer.search(/\r?\n/);
      if (newline >= 0) {
        const line = state.readBuffer.slice(0, newline).trim();
        state.readBuffer = state.readBuffer.slice(newline + 1);
        if (line) {
          return line;
        }
        continue;
      }
      const result = await Promise.race([state.reader.read(), responseTimeout(timeoutMs)]);
      if (result.done) {
        throw new Error("Serial reader closed");
      }
      state.readBuffer += state.decoder.decode(result.value, { stream: true });
    }
  }

  async function waitForGrblResponse(timeoutMs) {
    while (true) {
      const response = await readLine(timeoutMs);
      state.lastResponse = response;
      if (response.toLowerCase() === "ok") {
        return response;
      }
      if (/^error:/i.test(response) || /^ALARM:/i.test(response)) {
        throw new Error(response);
      }
      log(`GRBL: ${response}`);
    }
  }

  async function writeBytes(bytes) {
    ensureConnected();
    await state.writer.write(bytes);
  }

  async function sendCommand(line, options = {}) {
    ensureConnected();
    state.currentLine = line;
    refreshStatus();
    await state.writer.write(state.encoder.encode(`${line}\n`));
    return waitForGrblResponse(options.timeoutMs || DEFAULT_TIMEOUT_MS);
  }

  function normalizeGcode(text) {
    return String(text || "")
      .split(/\r?\n/)
      .map((line) => line.replace(/;.*$/, "").replace(/\([^)]*\)/g, "").trim())
      .filter((line) => line.length > 0);
  }

  // A4 用紙寸法(mm)。Y-UP・原点左下。G-code 座標も同系のため bounds 計算は不要。
  const PAPER_W_MM = 210;
  const PAPER_H_MM = 297;
  // penZ がこの値以上で紙接触(描画)。終端リフト(2.6)・連綿(つなぎ画)も描画扱い、
  // ペンアップ(0.5)は移動扱いになる閾値。
  const PEN_CONTACT_Z = 1.5;

  function parseGcodeToSegments(lines) {
    // normalizeGcode 済みの行配列を前提。lineIndex は配列 index に一致し送信進捗に対応する。
    const segments = [];
    let penX = 0;
    let penY = 0;
    let penZ = 0.5;
    let hasPrev = false;
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      // $H / G4 / G92 / G90 / G91 は位置もペン状態も更新しない。
      if (/^(\$|G4|G92|G90|G91)/.test(line)) {
        continue;
      }
      const isMove = /^G0/.test(line);
      const isDraw = /^G1/.test(line);
      if (!isMove && !isDraw) {
        continue;
      }
      const mx = line.match(/X(-?\d+\.?\d*)/);
      const my = line.match(/Y(-?\d+\.?\d*)/);
      const mz = line.match(/Z(-?\d+\.?\d*)/);
      if (mz) {
        penZ = parseFloat(mz[1]);
      }
      const hasXY = mx || my;
      if (!hasXY) {
        // X/Y を持たない行(例 `G1G90 Z5.0`)は penZ 更新のみで位置据置。
        continue;
      }
      const nextX = mx ? parseFloat(mx[1]) : penX;
      const nextY = my ? parseFloat(my[1]) : penY;
      if (isDraw && penZ >= PEN_CONTACT_Z && hasPrev) {
        segments.push({ x0: penX, y0: penY, x1: nextX, y1: nextY, lineIndex: i });
      }
      penX = nextX;
      penY = nextY;
      hasPrev = true;
    }
    return segments;
  }

  function ensureBgImage() {
    // レポート用紙背景は head 注入の data URI を一度だけ Image 化する。
    // 非同期ロード完了時に現在の進捗で再描画して背景を反映させる。
    if (state.bgImage || !window.__ppReportPaper) {
      return;
    }
    const img = new Image();
    img.onload = () => {
      state.bgReady = true;
      drawPreview(state.preview._lastSent || 0);
    };
    state.bgImage = img;
    img.src = window.__ppReportPaper;
  }

  function drawPreview(sentInPage) {
    const canvas = byId("webserial-preview-canvas");
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const cssW = rect.width;
    const cssH = rect.height;
    if (cssW <= 0 || cssH <= 0) {
      return;
    }
    // 高 DPI で線がにじまないよう実バッファを dpr 倍にし、描画は CSS px 基準に正規化する。
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    // 背景の遅延ロード完了後に同じ進捗で再描画できるよう保持する。
    state.preview._lastSent = sentInPage;

    // 線が背景の上に来るよう、レポート用紙背景を線描画ループの前に全面へ敷く。
    ensureBgImage();
    if (state.bgReady) {
      ctx.drawImage(state.bgImage, 0, 0, cssW, cssH);
    }

    const toCx = (x) => (x / PAPER_W_MM) * cssW;
    const toCy = (y) => ((PAPER_H_MM - y) / PAPER_H_MM) * cssH;
    const segments = state.preview.segments;

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const seg of segments) {
      const done = seg.lineIndex < sentInPage;
      ctx.strokeStyle = done ? "#1a1a1a" : "#d4d4d4";
      ctx.lineWidth = done ? 1.2 : 1.0;
      ctx.beginPath();
      ctx.moveTo(toCx(seg.x0), toCy(seg.y0));
      ctx.lineTo(toCx(seg.x1), toCy(seg.y1));
      ctx.stroke();
    }

    // ペン位置: 送信済みの最後のセグメント終点に赤丸。未送信(sentInPage=0)なら描かない。
    let penSeg = null;
    for (const seg of segments) {
      if (seg.lineIndex < sentInPage) {
        penSeg = seg;
      }
    }
    if (penSeg) {
      ctx.fillStyle = "#e74c3c";
      ctx.beginPath();
      ctx.arc(toCx(penSeg.x1), toCy(penSeg.y1), 3, 0, Math.PI * 2);
      ctx.fill();
    }

    const info = byId("webserial-preview-info");
    if (info) {
      const p = state.preview;
      if (segments.length === 0 && p.total === 0) {
        info.textContent = "対象なし";
      } else if (sentInPage > 0) {
        info.textContent = `ページ ${p.pageNo}・${sentInPage}/${p.total} 行`;
      } else {
        info.textContent = `対象: ${p.name || "-"} / 全${p.total}行`;
      }
    }
  }

  async function preview(source, generatedFiles, uploadedFiles, pages) {
    // 送信中は start() 側が逐次描画するため、change イベント由来の再描画は無視する。
    if (state.streaming) {
      return;
    }
    try {
      const files = selectedFiles(source, generatedFiles, uploadedFiles, pages);
      if (files.length === 0) {
        state.preview = { segments: [], total: 0, pageNo: 1, name: "" };
        drawPreview(0);
        return;
      }
      const text = await fetchFileText(files[0].file);
      const lines = normalizeGcode(text);
      state.preview = {
        segments: parseGcodeToSegments(lines),
        total: lines.length,
        pageNo: files[0].pageNo,
        name: fileName(files[0].file, 0),
      };
      drawPreview(0);
    } catch (error) {
      log(`[preview] ${error.message || error}`, "warn");
    }
  }

  function fileName(fileData, index) {
    return fileData?.orig_name || fileData?.name || fileData?.path?.split("/").pop() || `file-${index + 1}`;
  }

  async function fetchFileText(fileData) {
    const url = fileData?.url || fileData?.path || (typeof fileData === "string" ? fileData : null);
    if (!url) {
      throw new Error("G-code ファイル URL 取得失敗");
    }
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`G-code 読み込み失敗: ${response.status}`);
    }
    return response.text();
  }

  function selectedFiles(source, generatedFiles, uploadedFiles, pages) {
    // pageNo は1始まりの実ページ番号。一部ページのみ選択時に進捗表示が元ページ番号と
    // ずれないよう、選択サブセット内の連番ではなく元の番号を保持する。
    const raw = source === "uploaded" ? uploadedFiles : generatedFiles;
    if (!raw) {
      return [];
    }
    const all = Array.isArray(raw) ? raw.filter(Boolean) : [raw];
    // 生成済みは「送信ページ」(1始まり)で絞り込む。未指定なら全ページ。
    // アップロードはページ概念がないため常に全件（連番を割り当て）。
    if (source === "uploaded" || !Array.isArray(pages) || pages.length === 0) {
      return all.map((file, i) => ({ file, pageNo: i + 1 }));
    }
    return pages
      .map((p) => Number(p))
      .filter((p) => Number.isInteger(p) && p >= 1 && p <= all.length)
      .sort((a, b) => a - b)
      .map((p) => ({ file: all[p - 1], pageNo: p }));
  }

  async function loadPageLines(items) {
    // 用紙交換ポーズのためページ(ファイル)単位で行配列を保持し、結合しない。
    // items は selectedFiles() の {file, pageNo} 配列。
    const pages = [];
    for (let i = 0; i < items.length; i += 1) {
      const text = await fetchFileText(items[i].file);
      pages.push({
        name: fileName(items[i].file, i),
        lines: normalizeGcode(text),
        pageNo: items[i].pageNo,
      });
    }
    return pages;
  }

  function setPaperChange(visible, message) {
    const node = byId("webserial-paper-change");
    if (!node) {
      return;
    }
    node.style.display = visible ? "" : "none";
    if (visible && message) {
      node.textContent = message;
    }
  }

  function waitForResume() {
    // 次ページ送信前に停止。resume()=続行 / stop()=中断 で解決する。
    return new Promise((resolve) => {
      state.paused = true;
      state.pausedResolve = resolve;
      refreshStatus();
    });
  }

  function settleResume(value) {
    if (!state.pausedResolve) {
      return;
    }
    const resolve = state.pausedResolve;
    state.pausedResolve = null;
    state.paused = false;
    setPaperChange(false);
    resolve(value);
  }

  function resume() {
    if (!state.paused) {
      log("用紙交換待ちではない", "warn");
      return;
    }
    log("[resume] 次ページ送信");
    settleResume(true);
    refreshStatus();
  }

  async function start(source, generatedFiles, uploadedFiles, pages) {
    if (state.streaming) {
      log("すでに送信中", "warn");
      return;
    }
    try {
      ensureConnected();
      const files = selectedFiles(source, generatedFiles, uploadedFiles, pages);
      if (files.length === 0) {
        throw new Error(source === "uploaded" ? "アップロード G-code 未選択" : "送信ページ未選択");
      }
      const pageList = await loadPageLines(files);
      const total = pageList.reduce((sum, page) => sum + page.lines.length, 0);
      if (total === 0) {
        throw new Error("送信可能な G-code 行なし");
      }
      state.streaming = true;
      state.cancelRequested = false;
      state.paused = false;
      state.sent = 0;
      state.total = total;
      state.lastResponse = "";
      log(`[start] ${pageList.length} ページ / ${total} 行`);
      for (let p = 0; p < pageList.length; p += 1) {
        const { name, lines, pageNo } = pageList[p];
        log(`[page ${p + 1}/${pageList.length}] ${name} (${lines.length} 行)`);
        // 送信中ページに追従して薄灰の対象線画を描き直す（resume 後の次ページも自動更新）。
        state.preview = {
          segments: parseGcodeToSegments(lines),
          total: lines.length,
          pageNo,
          name,
        };
        drawPreview(0);
        for (let i = 0; i < lines.length; i += 1) {
          if (state.cancelRequested) {
            log(`[cancelled] ${state.sent} / ${state.total}`, "warn");
            return;
          }
          try {
            const response = await sendCommand(lines[i], {
              timeoutMs: lines[i] === "$H" ? HOME_TIMEOUT_MS : DEFAULT_TIMEOUT_MS,
            });
            state.sent += 1;
            state.lastResponse = response;
            drawPreview(i + 1);
            refreshStatus();
          } catch (error) {
            throw new Error(`page ${p + 1} line ${i + 1}: ${lines[i]} -> ${error.message || error}`);
          }
        }
        // 用紙交換ポーズは「現ページが非空 かつ 以降に非空ページが残っている」ときのみ。
        // 空ページ(正規化後0行)や最後の非空ページ以降で余計に停止しないようにする。
        const hasMoreContent = pageList.slice(p + 1).some((pg) => pg.lines.length > 0);
        if (lines.length > 0 && hasMoreContent) {
          setPaperChange(
            true,
            `ページ ${pageNo} 完了。用紙を交換し「続行」を押してください。`,
          );
          log(`[pause] ページ ${pageNo} 完了。用紙交換待ち`, "warn");
          const resumed = await waitForResume();
          if (!resumed || state.cancelRequested) {
            log(`[cancelled] 用紙交換待ち中に停止 ${state.sent} / ${state.total}`, "warn");
            return;
          }
        }
      }
      log("[done] stream");
    } catch (error) {
      log(`[error] stream: ${error.message || error}`, "error");
    } finally {
      state.streaming = false;
      state.cancelRequested = false;
      state.paused = false;
      state.pausedResolve = null;
      state.currentLine = "";
      setPaperChange(false);
      refreshStatus();
    }
  }

  function stop() {
    if (!state.streaming) {
      log("送信中ではない", "warn");
      return;
    }
    state.cancelRequested = true;
    settleResume(false);
    log("通常停止要求: 次行送信前に停止", "warn");
    refreshStatus();
  }

  async function emergencyStop() {
    try {
      ensureConnected();
      state.cancelRequested = true;
      settleResume(false);
      await writeBytes(new Uint8Array([0x21, 0x18]));
      log("緊急停止送信: ! + Ctrl-X", "error");
    } catch (error) {
      log(`緊急停止失敗: ${error.message || error}`, "error");
    } finally {
      state.streaming = false;
      state.currentLine = "";
      state.lastResponse = "soft reset";
      refreshStatus();
    }
  }

  function _stopSimulate() {
    if (state.simTimer !== null) {
      window.clearInterval(state.simTimer);
      state.simTimer = null;
    }
  }

  async function _simulate(source, generatedFiles, uploadedFiles, pages) {
    // 実機シリアル無しで進捗塗りを目視確認するためのデバッグ用。送信は一切行わない。
    _stopSimulate();
    const files = selectedFiles(source, generatedFiles, uploadedFiles, pages);
    if (files.length === 0) {
      log("[simulate] 対象なし", "warn");
      return;
    }
    const text = await fetchFileText(files[0].file);
    const lines = normalizeGcode(text);
    state.preview = {
      segments: parseGcodeToSegments(lines),
      total: lines.length,
      pageNo: files[0].pageNo,
      name: fileName(files[0].file, 0),
    };
    let sent = 0;
    drawPreview(sent);
    state.simTimer = window.setInterval(() => {
      sent += 1;
      drawPreview(sent);
      if (sent >= state.preview.total) {
        _stopSimulate();
      }
    }, 30);
  }

  function init() {
    if (!("serial" in navigator)) {
      log("WebSerial 非対応。Windows Chrome / Edge で http://localhost から開く", "warn");
    } else if (!window.isSecureContext) {
      log("Secure Context ではないため WebSerial 使用不可。http://localhost で開く", "warn");
    } else {
      log("WebSerial 使用可能");
    }
    refreshStatus();
  }

  window.penPlotterWebSerial = {
    connect,
    disconnect,
    start,
    resume,
    stop,
    emergencyStop,
    normalizeGcode,
    preview,
    _simulate,
    _stopSimulate,
    _constants: {
      BAUD_RATE,
    },
  };

  window.setTimeout(init, 0);
})();
