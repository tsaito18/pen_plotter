(function () {
  "use strict";

  const HOME_SEQUENCE = ["$H", "G4 P1", "G92 X0 Y297 Z0", "G90"];
  const PEN_UP_COMMAND = "G1G90 Z0.5 F5000";
  const PEN_DOWN_COMMAND = "G1G90 Z5 F5000";
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
    setButtonDisabled("webserial-home-btn", !state.connected || busy);
    setButtonDisabled("webserial-pen-up-btn", !state.connected || busy);
    setButtonDisabled("webserial-pen-down-btn", !state.connected || busy);
    setButtonDisabled("webserial-start-btn", unsupported || !state.connected || busy);
    setButtonDisabled("webserial-stop-btn", !busy);
    setButtonDisabled("webserial-emergency-btn", !state.connected);
  }

  function statusText() {
    if (!("serial" in navigator)) {
      return "非対応ブラウザ: Chrome / Edge のローカル表示で使用";
    }
    if (!window.isSecureContext) {
      return "Secure Context ではないため WebSerial 使用不可。localhost で開く";
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

  async function runCommandSequence(name, lines) {
    if (state.streaming) {
      log("送信中は操作不可", "warn");
      return;
    }
    try {
      ensureConnected();
      state.streaming = true;
      state.cancelRequested = false;
      state.sent = 0;
      state.total = lines.length;
      log(`[start] ${name}`);
      for (const line of lines) {
        const timeoutMs = line === "$H" ? HOME_TIMEOUT_MS : DEFAULT_TIMEOUT_MS;
        const response = await sendCommand(line, { timeoutMs });
        state.sent += 1;
        state.lastResponse = response;
        refreshStatus();
      }
      log(`[done] ${name}`);
    } catch (error) {
      log(`[error] ${name}: ${error.message || error}`, "error");
    } finally {
      state.streaming = false;
      state.currentLine = "";
      refreshStatus();
    }
  }

  function normalizeGcode(text) {
    return String(text || "")
      .split(/\r?\n/)
      .map((line) => line.replace(/;.*$/, "").replace(/\([^)]*\)/g, "").trim())
      .filter((line) => line.length > 0);
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

  function selectedFiles(source, generatedFiles, uploadedFiles) {
    const raw = source === "uploaded" ? uploadedFiles : generatedFiles;
    if (!raw) {
      return [];
    }
    return Array.isArray(raw) ? raw.filter(Boolean) : [raw];
  }

  async function loadSelectedGcode(source, generatedFiles, uploadedFiles) {
    const files = selectedFiles(source, generatedFiles, uploadedFiles);
    if (files.length === 0) {
      throw new Error(source === "uploaded" ? "アップロード G-code 未選択" : "生成済み G-code 未選択");
    }
    const chunks = [];
    for (let i = 0; i < files.length; i += 1) {
      const text = await fetchFileText(files[i]);
      chunks.push(`; --- ${fileName(files[i], i)} ---\n${text}`);
    }
    return normalizeGcode(chunks.join("\n"));
  }

  async function start(source, generatedFiles, uploadedFiles) {
    if (state.streaming) {
      log("すでに送信中", "warn");
      return;
    }
    try {
      ensureConnected();
      const lines = await loadSelectedGcode(source, generatedFiles, uploadedFiles);
      if (lines.length === 0) {
        throw new Error("送信可能な G-code 行なし");
      }
      state.streaming = true;
      state.cancelRequested = false;
      state.sent = 0;
      state.total = lines.length;
      state.lastResponse = "";
      log(`[start] stream ${lines.length} lines`);
      for (let i = 0; i < lines.length; i += 1) {
        if (state.cancelRequested) {
          log(`[cancelled] stream ${state.sent} / ${state.total}`, "warn");
          return;
        }
        try {
          const response = await sendCommand(lines[i], {
            timeoutMs: lines[i] === "$H" ? HOME_TIMEOUT_MS : DEFAULT_TIMEOUT_MS,
          });
          state.sent = i + 1;
          state.lastResponse = response;
          refreshStatus();
        } catch (error) {
          throw new Error(`line ${i + 1}: ${lines[i]} -> ${error.message || error}`);
        }
      }
      log("[done] stream");
    } catch (error) {
      log(`[error] stream: ${error.message || error}`, "error");
    } finally {
      state.streaming = false;
      state.cancelRequested = false;
      state.currentLine = "";
      refreshStatus();
    }
  }

  function stop() {
    if (!state.streaming) {
      log("送信中ではない", "warn");
      return;
    }
    state.cancelRequested = true;
    log("通常停止要求: 次行送信前に停止", "warn");
    refreshStatus();
  }

  async function emergencyStop() {
    try {
      ensureConnected();
      state.cancelRequested = true;
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
    home: () => runCommandSequence("home", HOME_SEQUENCE),
    penUp: () => runCommandSequence("pen up", [PEN_UP_COMMAND]),
    penDown: () => runCommandSequence("pen down", [PEN_DOWN_COMMAND]),
    start,
    stop,
    emergencyStop,
    normalizeGcode,
    _constants: {
      HOME_SEQUENCE,
      PEN_UP_COMMAND,
      PEN_DOWN_COMMAND,
      BAUD_RATE,
    },
  };

  window.setTimeout(init, 0);
})();
