PYTHON ?= python

.PHONY: install dev serve migrate test lint format format-check typecheck check wheel-smoke docker-up docker-down doctor release-archive

install:
	$(PYTHON) -m pip install .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

serve:
	$(PYTHON) -m app.cli serve

migrate:
	$(PYTHON) -m alembic upgrade head

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy app

check: lint format-check typecheck test

wheel-smoke:
	rm -rf dist/wheel-smoke
	mkdir -p dist/wheel-smoke
	$(PYTHON) -m build --wheel --no-isolation --outdir dist/wheel-smoke
	$(PYTHON) scripts/wheel_smoke.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

doctor:
	$(PYTHON) -m app.cli doctor

release-archive:
	$(PYTHON) scripts/build_release.py
