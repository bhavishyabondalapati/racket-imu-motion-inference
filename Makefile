# One command per pipeline step. Run inside the conda env:
#   conda activate racket-imu-motion-inference
PYTHON ?= python

.PHONY: help install models test lint format sync frames segment reference classify trajectory \
        latency report all

help:
	@echo "make install     install pinned packages (editable) + dev tools"
	@echo "make models      download the pose model and verify its SHA-256"
	@echo "make test        unit tests (synthetic data only)"
	@echo "make lint        ruff + large-file check"
	@echo "make format      auto-format with ruff"
	@echo "make sync        M1   make frames     M2   make segment  M3"
	@echo "make reference   M4   make classify   M5   make trajectory M6"
	@echo "make latency     M7   make report     M8   make all      M1..M8 in order"

install:
	$(PYTHON) -m pip install -r requirements.txt

models:
	$(PYTHON) scripts/download_pose_model.py

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

sync:
	$(call not_yet,M1)
frames:
	$(call not_yet,M2)
segment:
	$(call not_yet,M3)
reference:
	$(call not_yet,M4)
classify:
	$(call not_yet,M5)
trajectory:
	$(call not_yet,M6)
latency:
	$(call not_yet,M7)
report:
	$(call not_yet,M8)

all: sync frames segment reference classify trajectory latency report
