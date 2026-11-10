VENV ?= .venv
PY := $(VENV)/bin/python
BIN := $(VENV)/bin

.PHONY: install run test lint eval demo samples docker clean

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

run:
	$(BIN)/medrag serve --host 0.0.0.0 --port 8000

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check .

eval:
	$(BIN)/medrag eval --strict

demo:
	$(BIN)/medrag demo

samples:
	$(BIN)/medrag samples

docker:
	[ -f .env ] || cp .env.example .env
	docker compose up --build

clean:
	rm -rf var eval_results .pytest_cache .ruff_cache
