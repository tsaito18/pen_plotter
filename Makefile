.PHONY: ui collect train pretrain finetune preview test lint format help

VENV := . .venv/bin/activate &&

# デフォルトパラメータ
CHECKPOINT  ?= data/models/finetuned.pt
PRETRAIN_CP ?= data/models/pretrain_checkpoint.pt
USER_DIR    ?= data/user_strokes
REF_DIR     ?= data/strokes
PORT        ?= 7860
COLLECT_PORT?= 8080
EPOCHS_PRE  ?= 80
EPOCHS_FT   ?= 20

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

ui: ## Web UIを起動
	$(VENV) python scripts/run_ui.py --checkpoint $(CHECKPOINT) --port $(PORT)

collect: ## ストローク収集UIを起動
	$(VENV) python scripts/collect_strokes.py --port $(COLLECT_PORT)

train: pretrain finetune ## 訓練（pretrain + finetune）

pretrain: ## 事前訓練
	$(VENV) python scripts/pretrain.py \
		--model-version v3-user \
		--hand-dir $(USER_DIR) \
		--ref-dir $(REF_DIR) \
		--epochs $(EPOCHS_PRE) \
		--batch-size 256 \
		--hidden-dim 128 \
		--style-dim 128 \
		--learning-rate 0.001

finetune: ## ファインチューニング
	$(VENV) python scripts/finetune.py \
		--checkpoint $(PRETRAIN_CP) \
		--user-dir $(USER_DIR) \
		--ref-dir $(REF_DIR) \
		--epochs $(EPOCHS_FT) \
		--batch-size 8 \
		--learning-rate 0.0005

preview: ## プレビュー画像を生成（TEXT変数で指定）
	@$(VENV) python -c "\
	from src.ui.web_app import PlotterPipeline; \
	from pathlib import Path; \
	import time; \
	p = PlotterPipeline( \
		checkpoint_path=Path('$(CHECKPOINT)'), \
		kanjivg_dir=Path('$(REF_DIR)'), \
		user_strokes_dir=Path('$(USER_DIR)'), \
	); \
	text = '$(TEXT)' if '$(TEXT)' else '1. 実験目的\n抵抗とコンデンサを組み合わせたCR直列回路の基本的な特性を測定する。'; \
	out = f'/tmp/preview_{int(time.time())}.png'; \
	p.generate_preview(text, out); \
	print(f'Saved: {out}')"

test: ## テスト実行
	$(VENV) pytest

lint: ## リント
	$(VENV) ruff check src/ tests/ scripts/

format: ## フォーマット
	$(VENV) ruff format src/ tests/ scripts/
