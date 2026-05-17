.PHONY: install data prepare train-all dowhy score eval test lint type fmt precommit ci clean app

PY := uv run

install:
	uv sync --extra dev
	$(PY) pre-commit install

data:
	$(PY) python scripts/download_data.py

prepare:
	$(PY) python scripts/prepare_data.py

train-all:
	$(PY) python scripts/train.py --estimator s-learner
	$(PY) python scripts/train.py --estimator t-learner
	$(PY) python scripts/train.py --estimator x-learner
	$(PY) python scripts/train.py --estimator dml
	$(PY) python scripts/train.py --estimator dr-learner

dowhy:
	$(PY) python scripts/run_dowhy.py --sample 1000000

score:
	$(PY) python scripts/score_sample.py --n 200000

eval:
	$(PY) python scripts/evaluate_all.py

test:
	$(PY) pytest -q

lint:
	$(PY) ruff check src tests scripts streamlit_app

fmt:
	$(PY) ruff format src tests scripts streamlit_app

type:
	$(PY) mypy src

precommit:
	$(PY) pre-commit run --all-files

ci: lint type test

app:
	$(PY) streamlit run streamlit_app/app.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
