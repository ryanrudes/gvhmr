# Dev shortcuts. `make check` runs the REQUIRED CI gates locally — run it before you push.
# (CI: `.github/workflows/ci.yml` → `ruff format --check gvhmr tools tests` + `pytest`; ruff-check /
#  pyright are advisory.) These mirror CI's exact scope so "green locally" == "green in CI".
.PHONY: help fmt lint typecheck test check hooks

help:  ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | sort

fmt:  ## Format the whole tree (matches CI's ruff scope) — do this, not per-file
	uv run ruff format gvhmr tools tests

lint:  ## Ruff lint (advisory in CI)
	uv run ruff check gvhmr tools tests

typecheck:  ## Pyright (advisory in CI)
	uv run pyright

test:  ## CPU/MPS characterization suite
	uv run pytest -q

check:  ## The REQUIRED CI gates, locally: format-check + tests
	uv run ruff format --check gvhmr tools tests
	uv run pytest -q

hooks:  ## Install the pre-commit hook (one-time; auto-formats staged files)
	uv run pre-commit install
