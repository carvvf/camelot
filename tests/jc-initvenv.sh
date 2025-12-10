#!/usr/bin/env bash

# Bootstrap a local virtual environment for the Camelot project.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
DEV_PIP_PACKAGES=(
    "Pygments>=2.10.0"
    "black>=23.1.0"
    "build"
    "coverage[toml]>=6.2"
    "flake8>=4.0.1"
    "flake8-bandit>=2.1.2"
    "flake8-bugbear>=21.9.2"
    "flake8-rst-docstrings>=0.2.5"
    "ghostscript>=0.7"
    "isort>=5.10.1"
    "jsonschema>=4.21.1"
    "mypy>=0.930"
    "myst-parser>=2.0.0"
    "nox>=2024.10.9"
    "pep8-naming>=0.12.1"
    "pre-commit>=2.16.0"
    "pre-commit-hooks>=4.1.0"
    "pyupgrade>=2.29.1"
    "pytest>=9.0.0"
    "pytest-mpl>=0.17.0"
    "safety>=2.2.3"
    "sphinx>=4.3.2"
    "sphinx-autobuild>=2021.3.14"
    "sphinx-book-theme>=1.0.1"
    "sphinx-click>=3.0.2"
    "sphinx-copybutton>=0.5.0"
    "sphinx-prompt>=1.5.0"
    "twine"
    "typeguard>=2.13.3"
    "xdoctest[colors]>=0.15.10"
)

python_bin="${PYTHON:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Error: ${python_bin} is not available on PATH." >&2
    exit 1
fi

echo "Creating virtual environment at ${VENV_PATH}" >&2
if [[ ! -d "${VENV_PATH}" ]]; then
    "${python_bin}" -m venv "${VENV_PATH}"
fi

# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo "Installing project in editable mode (runtime dependencies only)" >&2
python -m pip install -e "${PROJECT_ROOT}"

if ((${#DEV_PIP_PACKAGES[@]} > 0)); then
    echo "Installing development tooling: ${DEV_PIP_PACKAGES[*]}" >&2
    python -m pip install "${DEV_PIP_PACKAGES[@]}"
fi

if ! command -v gs >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Warning: 'gs' (ghostscript) is not available on PATH.
Install ghostscript via your system package manager (for example: apt install ghostscript, brew install ghostscript)
to enable the Ghostscript backend and associated tests.
EOF
else
    echo "Detected ghostscript binary: $(command -v gs)" >&2
fi

echo "Virtual environment ready at ${VENV_PATH}" >&2
