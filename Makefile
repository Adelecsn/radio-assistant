PYTHON3 ?= python3
VENV_PYTHON := .venv/bin/python

.PHONY: setup test check ingest-help rsna-extract-help inference-help evaluate-help evaluate-run webapp-help webapp-setup webapp-run

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

evaluate-run: setup
	$(VENV_PYTHON) -m eval.evaluate \
		--predictions-dir data/predictions/baseline_v1 \
		--output-dir eval/outputs/baseline_v1

webapp-help: setup
	$(VENV_PYTHON) -m src.webapp --help

webapp-setup: setup
	$(VENV_PYTHON) -m pip install -r requirements-web.txt

webapp-run: webapp-setup
	$(VENV_PYTHON) -m src.webapp \
		--predictions-dir data/predictions/baseline_v1 \
		--db-path logs/webapp.sqlite
