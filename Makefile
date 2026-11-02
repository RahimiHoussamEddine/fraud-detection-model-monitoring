PYTHON ?= python3.12
VENV := .venv
RUN := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

.PHONY: install lint format-check test quality data-download data-audit experiment-select locked-evaluate package-release verify-artifact smoke-release api app

install:
	test -x $(RUN) || $(PYTHON) -m venv $(VENV)
	PIP_CACHE_DIR=.cache/pip $(RUN) -m pip install --requirement requirements.lock
	PIP_CACHE_DIR=.cache/pip $(RUN) -m pip install --no-deps --no-build-isolation --editable .
	PIP_CACHE_DIR=.cache/pip $(RUN) -m pip check

lint:
	$(RUFF) check .

format-check:
	$(RUFF) format --check .

test:
	mkdir -p .tmp
	$(RUN) -m pytest -W error --basetemp=.tmp/pytest --cov=fraud_monitoring --cov-report=term-missing --cov-fail-under=75

quality: lint format-check test

data-download:
	$(PYTHON) scripts/download_data.py

data-audit:
	$(RUN) scripts/data_audit.py --input data/raw/creditcard.parquet --output reports/data_audit.json

experiment-select:
	$(RUN) scripts/select_model.py --input data/raw/creditcard.parquet --output artifacts/v1

locked-evaluate:
	$(RUN) scripts/run_locked_evaluation.py --input data/raw/creditcard.parquet --artifact-dir artifacts/v1 --output reports/locked/v1 --protocol docs/EXPERIMENT_PROTOCOL.md

package-release:
	$(RUN) scripts/package_artifact.py --artifact-dir artifacts/v1 --locked reports/locked/v1

verify-artifact:
	$(RUN) scripts/verify_artifact.py --artifact-dir artifacts/v1

smoke-release:
	mkdir -p .tmp
	$(RUN) scripts/smoke_release_pipeline.py

api:
	FRAUD_ARTIFACT_DIR=artifacts/v1 $(VENV)/bin/uvicorn fraud_monitoring.api:app --host 127.0.0.1 --port 8001

app:
	FRAUD_ARTIFACT_DIR=artifacts/v1 $(VENV)/bin/streamlit run app/main.py --server.headless=true
