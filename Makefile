PYTHON3 ?= python3
VENV_PYTHON := .venv/bin/python

.PHONY: setup test check ingest-help

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
