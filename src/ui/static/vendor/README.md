# 同梱ライブラリ（ビルド不要・オフラインで動くようにリポジトリに置く）

| ファイル | パッケージ | 版 | ライセンス |
|---|---|---|---|
| `preact.module.js` | preact | 10.27.2 | MIT（`LICENSE-preact`） |
| `hooks.module.js` | preact/hooks | 10.27.2 | MIT（`LICENSE-preact`） |
| `htm.module.js` | htm | 3.1.1 | Apache-2.0（`LICENSE-htm`） |

npm の配布物（`dist/*.module.js`）をそのまま置き、sourceMappingURL の行だけ消している。
`index.html` / `collect.html` の import map で `preact` / `preact/hooks` / `htm` に割り当てる。
更新するときは `npm pack preact@<版> htm@<版>` で取り出して差し替え、この表を直す。
