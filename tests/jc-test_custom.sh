#!/usr/bin/env bash
set -euo pipefail

# Simple utility to test the PDF processing scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#### USER DEFINED ENVIRONMENT ################################

CLI_FILE_NAME="${1-}"
CONFIG_FILE="$(realpath "$SCRIPT_DIR/../jc-config.debug.env")"

##############################################################

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Missing configuration file: $CONFIG_FILE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

if [[ -n "$CLI_FILE_NAME" ]]; then
    FILE_NAME="$CLI_FILE_NAME"
fi

: "${BASE_DATA_DIR:?BASE_DATA_DIR must be configured in $CONFIG_FILE}"
: "${OUT_DIR:?OUT_DIR must be configured in $CONFIG_FILE}"
: "${FILE_NAME:?FILE_NAME must be provided as an argument or configured in $CONFIG_FILE}"

export FILE_NAME

source "$VENV_DIR/bin/activate"

PDF_PATH="$BASE_DATA_DIR$FILE_NAME"
DRAW_BOXES_SCRIPT="$SCRIPT_DIR/jc-draw-boxes.sh"
mkdir -p "$OUT_DIR"

elapsed_start=$(date +%s)

run_flavor() {
    local flavor="$1"
    local label="custom_${flavor}"
    local base_path="$OUT_DIR$FILE_NAME.$flavor"
    local json_output="${base_path}.json"
    local html_output="${base_path}.html"
    local overlay_output="${base_path}.pdf"

    local cmd=(
        camelot
        --parallel
        -p all
        --format json-coords
        --output "$json_output"
        "$flavor"
        "$PDF_PATH"
    )

    if [[ "$flavor" == "lattice" || "$flavor" == "hybrid" || "$flavor" == "stream" ]]; then
        local remove_background_value="${REMOVE_BACKGROUND_ARTIFACTS:-}"
        if [[ -z "$remove_background_value" ]]; then
            remove_background_value="true"
        fi
        case "${remove_background_value,,}" in
            true|1|yes|on|enable|enabled)
                cmd+=(--remove_background_artifacts)
                ;;
            false|0|no|off|disable|disabled)
                cmd+=(--no-remove_background_artifacts)
                ;;
        esac
    fi

    if [[ "$flavor" == "lattice" || "$flavor" == "hybrid" ]]; then
        local remove_text_value="${REMOVE_NATIVE_TEXT:-}"
        if [[ -z "$remove_text_value" ]]; then
            remove_text_value="true"
        fi
        case "${remove_text_value,,}" in
            true|1|yes|on|enable|enabled)
                cmd+=(--remove_text)
                ;;
            false|0|no|off|disable|disabled)
                cmd+=(--no-remove_text)
                ;;
        esac
    fi

    "${cmd[@]}"

    local command_preview
    printf -v command_preview '%q ' "${cmd[@]}"
    command_preview=${command_preview% }

    PYTHONPATH="$PROJECT_ROOT" \
    TABLES_JSON="$json_output" \
    HTML_OUTPUT="$html_output" \
    LABEL="$label" \
    FLAVOR="$flavor" \
    PDF_PATH="$PDF_PATH" \
    COMMAND_PREVIEW="$command_preview" \
    REMOVE_BACKGROUND_ARTIFACTS="${REMOVE_BACKGROUND_ARTIFACTS:-}" \
    REMOVE_NATIVE_TEXT="${REMOVE_NATIVE_TEXT:-}" \
    python - <<'PY'
import os
from pathlib import Path

from camelot.core import generate_html_report


def _interpret_bool(value, default=False):
    if value is None:
        return default
    lowered = value.strip().lower()
    if not lowered:
        return default
    if lowered in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if lowered in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    return default

tables_json = Path(os.environ["TABLES_JSON"])
html_output = Path(os.environ["HTML_OUTPUT"])
label = os.environ["LABEL"]
flavor = os.environ["FLAVOR"]
pdf_path = Path(os.environ["PDF_PATH"])
command_preview = os.environ.get("COMMAND_PREVIEW", "")
remove_background_default = _interpret_bool(
    os.environ.get("REMOVE_BACKGROUND_ARTIFACTS"), True
)
remove_text_default = _interpret_bool(
    os.environ.get("REMOVE_NATIVE_TEXT"), True
)

specification = {
    "label": label,
    "flavor": flavor,
    "format": "json-coords",
    "html_report": True,
    "remove_background_artifacts": remove_background_default,
    "remove_text": remove_text_default,
}

spec_defaults = {
    "pages": "all",
    "format": "json-coords",
    "remove_background_artifacts": remove_background_default,
    "remove_text": remove_text_default,
    "html_report": True,
}

generate_html_report(
    tables_json=tables_json,
    html_output=html_output,
    label=label,
    flavor=flavor,
    pdf_path=pdf_path,
    command_preview=command_preview,
    specification=specification,
    specification_defaults=spec_defaults,
)
PY

    if [[ -x "$DRAW_BOXES_SCRIPT" ]]; then
        "$DRAW_BOXES_SCRIPT" "$json_output" "$PDF_PATH" "$overlay_output"
    fi
}

run_flavor lattice
run_flavor stream
run_flavor hybrid
run_flavor network

elapsed_end=$(date +%s)
duration=$((elapsed_end - elapsed_start))
echo "Completed in ${duration}s"

deactivate
