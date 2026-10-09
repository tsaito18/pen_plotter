// Pen Plotter — 手書きスタジオ
// 書く（原稿＋ライブ下書き）→ 清書（手書きストローク＋同一の G-code）→ 描く（WebSerial でプロッタへ）。
// 状態と操作は studio/state.js、画面は studio/view.js（Preact + htm、ビルド不要）。

import { render } from "preact";
import { html, toast } from "./common.js";
import { boot } from "./studio/state.js";
import { App } from "./studio/view.js";

boot()
  .then(() => render(html`<${App} />`, document.getElementById("root")))
  .catch((error) => {
    console.error(error);
    toast(`起動に失敗しました: ${error.message || error}`, "error", 10000);
  });
