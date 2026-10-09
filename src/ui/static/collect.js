// 筆跡 — 集める（iPad＋Apple Pencil）・見直す・学習。スタジオと同じサーバー・同じ部品。
// 状態と操作は collect/state.js、画面は collect/view.js（Preact + htm、ビルド不要）。

import { render } from "preact";
import { html, toast } from "./common.js";
import { boot } from "./collect/state.js";
import { App } from "./collect/view.js";

// 画面を先に出してから（パッド・書き順の canvas を結び付けてから）データを読む
render(html`<${App} />`, document.getElementById("root"));
boot().catch((error) => {
  console.error(error);
  toast(`起動に失敗しました: ${error.message || error}`, "error", 10000);
});
