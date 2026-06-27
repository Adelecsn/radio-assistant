PYTHON3 ?= python3
VENV_PYTHON := .venv/bin/python

.PHONY: setup test check ingest-help rsna-extract-help inference-help inference-run improved-run medgemma-run medgemma-improved-run evaluate-help evaluate-run improved-evaluate-run compare-run webapp-help webapp-setup webapp-run

$(VENV_PYTHON):
	$(PYTHON3) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip

setup: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install -r requirements-test.txt

test: setup
	$(VENV_PYTHON) -m pytest -q

check: test
	$(VENV_PYTHON) -m compileall -q src eval tests

ingest-help: setup
	$(VENV_PYTHON) -m src.ingest --help

rsna-extract-help: setup
	$(VENV_PYTHON) -m src.ingest.rsna_extract --help

inference-help: setup
	$(VENV_PYTHON) -m src.inference --help

evaluate-help: setup
	$(VENV_PYTHON) -m eval.evaluate --help

inference-run: setup
	$(VENV_PYTHON) -m src.inference \
		--manifest data/manifests/ingest_manifest.csv \
		--output-dir data/predictions/baseline_v1

improved-run: setup
	$(VENV_PYTHON) -m src.inference \
		--variant improved \
		--manifest data/manifests/ingest_manifest.csv \
		--output-dir data/predictions/improved_v1

# Requires `pip install -r requirements.txt`, an accepted MedGemma license and HF_TOKEN.
medgemma-run:
	$(VENV_PYTHON) -m src.inference \
		--variant medgemma \
		--manifest data/manifests/ingest_manifest.csv \
		--output-dir data/predictions/medgemma_v1

medgemma-improved-run:
	$(VENV_PYTHON) -m src.inference \
		--variant medgemma \
		--prompt-file prompts/improved_v1.txt \
		--prompt-version medgemma-v2.0 \
		--manifest data/manifests/ingest_manifest.csv \
		--output-dir data/predictions/medgemma_v2

evaluate-run: setup
	$(VENV_PYTHON) -m eval.evaluate \
		--predictions-dir data/predictions/baseline_v1 \
		--output-dir eval/outputs/baseline_v1

improved-evaluate-run: setup
	$(VENV_PYTHON) -m eval.evaluate \
		--predictions-dir data/predictions/improved_v1 \
		--output-dir eval/outputs/improved_v1

compare-run: setup
	$(VENV_PYTHON) -m eval.compare \
		--baseline-dir data/predictions/baseline_v1 \
		--improved-dir data/predictions/improved_v1 \
		--output-dir eval/outputs/comparison

webapp-help: setup
	$(VENV_PYTHON) -m src.webapp --help

webapp-setup: setup
	$(VENV_PYTHON) -m pip install -r requirements-web.txt

webapp-run: webapp-setup
	$(VENV_PYTHON) -m src.webapp \
		--predictions-dir data/predictions/baseline_v1 \
		--db-path logs/webapp.sqlite
