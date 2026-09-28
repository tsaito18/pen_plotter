from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PinkNoise1D:
    """Voss-McCartney法による1/f(ピンク)ノイズ生成器。

    手書き揺らぎを白色ガウスノイズから低周波優位の1/fノイズへ変える用途。
    系列(文字読み順)で相関を持ち、隣接サンプルが正相関する。

    アルゴリズム:
        K本(octaves)の白色源を持ち、row k は 2^k サンプル毎に更新する。
        出力は全rowの和。低い行ほど更新が稀=低周波成分を担い、
        和のスペクトルが PSD ∝ 1/f に近づく。

    各rowは独立に分散1の正規乱数なので、和の分散は K に比例する。
    出力を /√K でスケールし std≈1 に正規化する。

    決定論的(seed指定で再現)・更新コストO(1)・1サンプルずつのストリーミング。
    """

    def __init__(self, octaves: int = 16, seed: int | None = None) -> None:
        """初期化する。

        Args:
            octaves: 白色源の本数K。多いほど低周波の裾が伸びる。
            seed: 乱数シード。指定すると系列が再現可能になる。
        """
        if octaves < 1:
            raise ValueError("octaves must be >= 1")
        self._octaves = octaves
        self._seed = seed
        self._norm = 1.0 / np.sqrt(octaves)
        self.reset()

    def reset(self) -> None:
        """内部状態を初期化する。同seedなら同一系列を再現する。"""
        self._rng = np.random.default_rng(self._seed)
        # 各rowの現在値を初期化(全rowが寄与した状態から開始)
        self._rows: NDArray[np.float64] = self._rng.standard_normal(self._octaves)
        # サンプルカウンタ。row k の更新判定に下位ビットを使う
        self._counter = 0

    def sample(self) -> float:
        """1サンプルを返し状態を前進させる。

        Returns:
            std≈1 に正規化された1/fノイズ1サンプル。
        """
        self._counter += 1
        # row k は 2^k サンプル毎に更新 = counter の下位ビット遷移で判定。
        # counter の trailing zero 数までの row を更新する(Voss-McCartney)。
        c = self._counter
        for k in range(self._octaves):
            self._rows[k] = self._rng.standard_normal()
            if c & 1:
                break
            c >>= 1

        return float(self._rows.sum() * self._norm)

    def samples(self, n: int) -> NDArray[np.float64]:
        """n個のサンプルをまとめて返す。

        Args:
            n: 生成するサンプル数。

        Returns:
            形状 (n,) の1/fノイズ系列。
        """
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self.sample()
        return out
