# Contributing

## Dev setup

```bash
# Clone, then:
uv sync --extra dev                       # core + dev tools
uv sync --extra dev --extra streamlit     # adds the Streamlit Cloud slim extra
uv run pre-commit install                 # one-time, registers git hooks
```

Python 3.11+ is required (DoWhy's networkx pin is the floor; 3.12 has not been tested).

## Running checks locally

```bash
make lint        # ruff check
make fmt         # ruff format (writes)
make type        # mypy src
make test        # pytest -q
make precommit   # pre-commit run --all-files
make ci          # lint + type + test (what CI runs)
```

## TDD policy

This repo uses TDD for the math-bearing modules:

- `src/upliftbench/eval/`
- `src/upliftbench/segmentation.py`
- `src/upliftbench/data/loader.py`, `src/upliftbench/data/prepare.py`
- `src/upliftbench/persistence.py`
- `src/upliftbench/refute/dowhy_pipeline.py`

For each change to these, follow the cycle:

1. **RED**: write the failing test in `tests/`.
2. **VERIFY RED**: `uv run pytest tests/path/test_x.py -v`. The test must fail with the expected message.
3. **GREEN**: write the minimal implementation to make it pass.
4. **VERIFY GREEN**: rerun pytest.
5. **REFACTOR**: tidy without changing behavior; stay green.
6. **COMMIT**: stage specific paths, no `git add -A`.

TDD is **not** required for the LightGBM/EconML/CausalML training-loop internals or for the Streamlit app. Those get smoke tests in `tests/estimators/` instead.

## Pre-commit hooks

Configured in `.pre-commit-config.yaml`:

- `trailing-whitespace`, `end-of-file-fixer`, `check-merge-conflict`, `check-yaml`, `check-toml`
- `check-added-large-files` with a 500 KB cap (excluded for `.claude/skills/` only)
- `ruff` (fix mode) + `ruff-format`
- `no-em-dash` (forbids `U+2014` anywhere in committed text)

The em-dash hook is intentional. Use commas, parentheses, semicolons, or a period. CI re-checks via `grep` on every push.

## CI

GitHub Actions, `.github/workflows/ci.yml`. Runs on `ubuntu-latest` only (macOS minutes cost 10x; public repo gets unlimited Linux minutes).

Steps:

1. `uv sync --extra dev`
2. `uv run ruff check ...`
3. `uv run ruff format --check ...`
4. `uv run mypy src`
5. `uv run pytest -q --cov=upliftbench`
6. em-dash grep

CI is intentionally fast (under 5 minutes). Heavy stuff (full-data training, DoWhy refutation) runs on the developer's laptop, not in CI.

## Commit message style

Prefix with the phase tag and the area:

```
feat(phase2): eval harness + S-learner + T-learner + train CLI
fix(eval): handle q_total == 0 in qini_coefficient
docs: polish README and add docs/ tree
```

Don't sneak in unrelated changes; one commit, one logical purpose.

## Branch policy

Develop on `claude/causal-inference-uplift-x0DZw`. Do not push to `main` without explicit user request. Open one PR per logical unit; pushing to the branch updates the existing PR automatically.

## When in doubt

- "Should I add a comment?" -> Only when the WHY is non-obvious. Names should carry the WHAT.
- "Should I refactor adjacent code?" -> No, unless your change requires it. See `.claude/skills/karpathy-guidelines/SKILL.md` (surgical changes).
- "Should I use a sub-agent in parallel?" -> If the work fans out cleanly into independent files with no shared state, yes. See `.claude/skills/dispatching-parallel-agents/SKILL.md`.
- "Should I extend the plan?" -> Edit `/root/.claude/plans/...md` first, run the writing-plans review, then code.
