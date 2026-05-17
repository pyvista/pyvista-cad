# List available recipes
default:
    @just --list

# Sync the virtual environment with [dev] extras (default)
sync:
    uv sync --extra dev

# Sync with all optional CAD backends installed
sync-full:
    uv sync --extra dev --extra full --extra openscad

# Delete and recreate the virtual environment
reset:
    rm -rf .venv
    uv sync --extra dev

# Run all tests with coverage
test *args:
    uv run pytest -vv tests/ --cov {{ args }}

# Run only the lazy-imports test — fast guardrail
test-lazy:
    uv run pytest -vv tests/test_lazy_imports.py

# Run only the cross-package compose tests (requires .manifold and .trimesh installed)
test-compose:
    uv run pytest -vv tests/test_ecosystem_compose.py

# Run type checking
typecheck:
    uv run mypy src/pyvista_cad

# Run all linters and formatters (pre-commit hooks)
lint:
    uvx pre-commit run --all-files

# Build a wheel and sdist
build:
    uv build

# Build Sphinx docs (HTML, including gallery)
docs:
    uv run --no-sync --extra docs make -C doc html

# Rebuild docs from a clean state
docs-clean:
    rm -rf doc/_build doc/examples doc/sg_execution_times.rst
    uv run --no-sync --extra docs make -C doc html

# Serve the built docs locally on http://localhost:11000
docs-serve: docs
    @echo "Open http://localhost:11000"
    cd doc/_build/html && uv run --no-sync python -m http.server 11000

# Regenerate the README's hero render asset
render-hero:
    uv run --no-sync python assets/render_hero.py
