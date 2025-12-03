#!/usr/bin/env bash

# Bootstrap a local virtual environment for the Camelot project.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
EXTRA_PIP_PACKAGES=(
    "ghostscript>=0.7"
    "pytest>=9.0.0"
    "jsonschema>=4.21.1"
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

echo "Installing project in editable mode with dev extras" >&2
python -m pip install -e "${PROJECT_ROOT}[dev]"

if ((${#EXTRA_PIP_PACKAGES[@]} > 0)); then
    echo "Installing supplemental tooling: ${EXTRA_PIP_PACKAGES[*]}" >&2
    python -m pip install "${EXTRA_PIP_PACKAGES[@]}"
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
