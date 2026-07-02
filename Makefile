# Dev shortcuts. `make check` runs the REQUIRED CI gates locally — run it before you push.
# (CI: `.github/workflows/ci.yml` → `ruff format --check gvhmr tools tests` + `pytest`; ruff-check /
#  pyright are advisory.) These mirror CI's exact scope so "green locally" == "green in CI".
#
# --no-sync: a plain `uv run` re-syncs the env to the lock's default extras first, which on a GPU box
# reverts the cuXXX torch and prunes DPVO/preproc (see docs/INSTALL.md). These targets only need the
# tools already installed — `uv sync --extra dev` once beforehand.
UV_RUN := uv run --no-sync

.PHONY: help fmt lint typecheck test check hooks

help:  ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | sort

fmt:  ## Format the whole tree (matches CI's ruff scope) — do this, not per-file
	$(UV_RUN) ruff format gvhmr tools tests

lint:  ## Ruff lint (advisory in CI)
	$(UV_RUN) ruff check gvhmr tools tests

typecheck:  ## Pyright (advisory in CI)
	$(UV_RUN) pyright

test:  ## CPU/MPS characterization suite
	$(UV_RUN) pytest -q

check:  ## The REQUIRED CI gates, locally: format-check + tests
	$(UV_RUN) ruff format --check gvhmr tools tests
	$(UV_RUN) pytest -q

hooks:  ## Install the pre-commit hook (one-time; auto-formats staged files)
	$(UV_RUN) pre-commit install
