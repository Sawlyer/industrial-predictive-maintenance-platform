.PHONY: help install data demo-data train report figures app api test lint format docker-build docker-up all

PYTHON ?= python

help:
	@echo "install     install the package and dev tools"
	@echo "data        check for the NASA C-MAPSS dataset"
	@echo "demo-data   generate a synthetic fleet so everything runs offline"
	@echo "train       build features, cross-validate, fit, save artifacts"
	@echo "report      print the last scorecard"
	@echo "figures     export README figures to reports/figures/"
	@echo "app         run the Streamlit dashboard"
	@echo "api         run the FastAPI service"
	@echo "test        run the test suite"
	@echo "lint        ruff check"
	@echo "all         demo-data + train + figures"

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m predmaint.cli data

demo-data:
	$(PYTHON) -m predmaint.cli demo-data

train:
	$(PYTHON) -m predmaint.cli train

report:
	$(PYTHON) -m predmaint.cli report

figures:
	$(PYTHON) -m predmaint.cli figures

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api:app --reload --port 8000

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src app tests

format:
	$(PYTHON) -m ruff format src app tests

docker-build:
	docker build -t predmaint:latest .

docker-up:
	docker compose up --build

all: demo-data train figures
