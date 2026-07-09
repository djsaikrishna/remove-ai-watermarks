#!/usr/bin/env bash

set -euo pipefail

uv sync --all-extras
# uv-outdated / uv-secure run via uvx (isolated env), NOT `uv run`: resolving them
# inside the project env crashes (uv-secure -> "annotated-doc raised exception") and,
# with set -e, aborts the whole gate before ruff/pyright/tests. uvx sidesteps the
# in-project dependency conflict (see CLAUDE.md "Test and lint").
uvx uv-outdated
uvx uv-secure --ignore-unfixed
uv run ruff check --fix
uv run ruff format
# Scoped to src/: a full-project pyright run OOM-crashes node on this ML-heavy
# repo (see CLAUDE.md "Test and lint"); src/ is the authoritative strict gate.
uv run pyright src/
uv run pytest -n auto
