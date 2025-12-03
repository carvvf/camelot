#!/usr/bin/env bash
set -euo pipefail

# Overlay helper used by the json-coords QA suite: the script consumes a tables
# payload (produced by Camelot exports), draws the reported bounding boxes on
# top of the source PDF, and emits an annotated PDF for human review. Consumers
# provide the paths to the JSON metadata, the original PDF, and the annotated
# PDF target.

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <tables-json> <input-pdf> <output-pdf>" >&2
    exit 1
fi

# Discover the repository layout so we can locate the shared virtualenv that
# contains pypdf and other heavy dependencies needed by the drawing script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

TABLES_JSON="$1"
INPUT_PDF="$2"
OUTPUT_PDF="$3"

if [[ ! -f "${TABLES_JSON}" ]]; then
    echo "Error: JSON file '${TABLES_JSON}' not found." >&2
    exit 1
fi

if [[ ! -f "${INPUT_PDF}" ]]; then
    echo "Error: PDF file '${INPUT_PDF}' not found." >&2
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Error: virtual environment not found at ${VENV_DIR}. Run tests/jc-initvenv.sh first." >&2
    exit 1
fi

# Activate the test virtualenv to ensure the Python block below has access to
# pypdf and friends.
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python - "$TABLES_JSON" "$INPUT_PDF" "$OUTPUT_PDF" <<'PY'
import sys

from camelot.core import draw_boxes
from camelot.core import crop_boxes

draw_boxes(sys.argv[1], sys.argv[2], sys.argv[3])
#crop_boxes(sys.argv[1], sys.argv[2], sys.argv[3])
PY

deactivate
