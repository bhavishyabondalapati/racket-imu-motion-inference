# One command per pipeline step. Run inside the conda env:
#   conda activate racket-imu-motion-inference
PYTHON ?= python

.PHONY: help install test lint format inspect preprocess classify quality skill \
        ablation errors latency report all

help:
	@echo "make install     install pinned packages (editable) + dev tools"
	@echo "make test        unit tests (synthetic data only)"
	@echo "make lint        ruff + large-file check"
	@echo "make format      auto-format with ruff"
	@echo "make inspect     D1   make preprocess D2   make classify D3"
	@echo "make quality     D4   make skill      D5   make ablation D6"
	@echo "make errors      D7   make latency    D8   make report   D9"
	@echo "make all         D2..D9 in order"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) scripts/check_large_files.py

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

# Pipeline steps. Each placeholder fails loudly until its milestone is built,
# so `make all` can never look like it succeeded on missing steps.
define not_yet
	@echo "make $@: not implemented yet ($(1))." >&2; exit 1
endef

inspect:
	$(PYTHON) scripts/inspect_dataset.py
preprocess:
	$(call not_yet,D2)
classify:
	$(call not_yet,D3)
quality:
	$(call not_yet,D4)
skill:
	$(call not_yet,D5)
ablation:
	$(call not_yet,D6)
errors:
	$(call not_yet,D7)
latency:
	$(call not_yet,D8)
report:
	$(call not_yet,D9)

all: preprocess classify quality skill ablation errors latency report
