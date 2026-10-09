// 画面の状態を 1 か所に持つ小さなストア。set すると購読中のコンポーネントが描き直される。
// プロッタ通信や用紙ビューアなど、コンポーネントの外のコードからも同じ状態を更新できる。

import { useEffect, useState } from "preact/hooks";

export function createStore(initial) {
  let state = initial;
  const listeners = new Set();
  return {
    get: () => state,
    /** 一部を差し替える（関数なら現在の状態から差分を作る）。 */
    set(patch) {
      const next = typeof patch === "function" ? patch(state) : patch;
      if (!next) return;
      state = { ...state, ...next };
      for (const fn of listeners) fn(state);
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}

/** ストアの変化で描き直す。 */
export function useStore(store) {
  const [, force] = useState(0);
  useEffect(() => store.subscribe(() => force((n) => n + 1)), [store]);
  return store.get();
}
