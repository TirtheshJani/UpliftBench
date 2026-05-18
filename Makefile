.PHONY: install data prepare train-all dowhy score eval test test-cov lint type fmt fmt-check em-dash-check precommit ci clean app

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

test-cov:
	$(PY) pytest -q --cov=upliftbench --cov-report=term

lint:
	$(PY) ruff check src tests scripts streamlit_app

fmt:
	$(PY) ruff format src tests scripts streamlit_app

fmt-check:
	$(PY) ruff format --check src tests scripts streamlit_app

type:
	$(PY) mypy src

em-dash-check:
	@if grep -rn $$'\xe2\x80\x94' src tests scripts streamlit_app blog README.md CLAUDE.md 2>/dev/null; then \
		echo "Em dashes found. Use commas, parens, or semicolons instead."; \
		exit 1; \
	fi

precommit:
	$(PY) pre-commit run --all-files

# Mirrors .github/workflows/ci.yml so a green `make ci` predicts a green GHA run.
ci: lint fmt-check type test-cov em-dash-check

app:
	$(PY) streamlit run streamlit_app/app.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
