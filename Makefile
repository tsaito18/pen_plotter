.PHONY: help ui collect pretrain finetune train preview test test-fast lint format

RUN := . .venv/bin/activate &&

# デフォルトパラメータ
CHECKPOINT   ?= data/models/finetuned.pt
PRETRAIN_CP  ?= data/models/pretrain_checkpoint.pt
USER_DIR     ?= data/user_strokes
REF_DIR      ?= data/strokes
PORT         ?= 7860
EPOCHS_PRE   ?= 80
EPOCHS_FT    ?= 20
TAG          ?= latest

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

ui: ## Web UI（スタジオ / と筆跡 /collect）を起動
	$(RUN) python scripts/run_ui.py --checkpoint $(CHECKPOINT) --kanjivg-dir $(REF_DIR) \
		--user-strokes-dir $(USER_DIR) --port $(PORT)

collect: ## 筆跡の収集画面（Web UI の /collect）を起動
	$(RUN) python scripts/collect_strokes.py --output-dir $(USER_DIR) --port $(PORT) --checkpoint $(CHECKPOINT)

pretrain: ## 変形モデルをユーザー筆跡で訓練
	$(RUN) python scripts/train.py pretrain --user-dir $(USER_DIR) --ref-dir $(REF_DIR) \
		--epochs $(EPOCHS_PRE) --batch-size 256 --hidden-dim 128 --style-dim 128 \
		--learning-rate 0.001 --use-aligner

finetune: ## StyleEncoder を微調整
	$(RUN) python scripts/train.py finetune --checkpoint $(PRETRAIN_CP) --user-dir $(USER_DIR) \
		--ref-dir $(REF_DIR) --epochs $(EPOCHS_FT) --batch-size 8 --learning-rate 0.0005 \
		--use-aligner

train: pretrain finetune ## pretrain → finetune

preview: ## 固定 seed の手書きプレビュー（A/B 比較用, TAG で世代名）
	$(RUN) python scripts/compare_handwriting.py --tag $(TAG) --checkpoint $(CHECKPOINT) \
		--kanjivg-dir $(REF_DIR) --user-strokes-dir $(USER_DIR)

test: ## 全テスト
	$(RUN) pytest

test-fast: ## 重いテストを除く
	$(RUN) pytest -m "not slow and not hardware"

lint: ## リント
	$(RUN) ruff check src tests scripts

format: ## フォーマット
	$(RUN) ruff format src tests scripts
