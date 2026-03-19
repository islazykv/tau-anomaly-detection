.PHONY: setup preprocess feature-engineer eda \
       train train-ae train-vae tune evaluate evaluate-ae evaluate-vae \
       pipeline test lint typecheck format clean \
       ui docker-build docker-run serve

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

setup:
	uv sync
	pre-commit install

# --------------------------------------------------------------------------- #
# Pipeline stages (unified entry point)
# --------------------------------------------------------------------------- #

preprocess:
	uv run python run.py stage=preprocess

feature-engineer:
	uv run python run.py stage=feature_engineer

eda:
	uv run python run.py stage=eda

train: train-ae

train-ae:
	uv run python run.py stage=train model=ae

train-vae:
	uv run python run.py stage=train model=vae

tune:
	uv run python run.py stage=tune

evaluate: evaluate-ae

evaluate-ae:
	uv run python run.py stage=evaluate model=ae

evaluate-vae:
	uv run python run.py stage=evaluate model=vae

pipeline: preprocess feature-engineer train evaluate

# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/

format:
	uv run pre-commit run --all-files

# --------------------------------------------------------------------------- #
# Experiment tracking
# --------------------------------------------------------------------------- #

ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ outputs/

# --------------------------------------------------------------------------- #
# Docker
# --------------------------------------------------------------------------- #

docker-build:
	docker build -t tau-anomaly-detection .

docker-run:
	docker run --rm tau-anomaly-detection

# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #

serve:
	uv run python run.py stage=serve model=$(MODEL)
