#!/usr/bin/env python3
"""
Convenience runner to iterate on Camelot table-extraction parameters.

Edit tests/jc-config.debug.env to point at the desired PDF and tweak the parameter
sets you would like to try. Each entry in PARAM_SETS produces one
Camelot CLI invocation, optionally followed by jc-draw-boxes.sh to visualise the
detected tables, and composes a single HTML overview containing all extracted
tables per configuration.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PATH = PROJECT_ROOT / ".venv"

# Path to the camelot executable. By default, expect the project virtualenv.
CAMELOT_BIN = PROJECT_ROOT / ".venv" / "bin" / "camelot"

# Optional bootstrap script for the venv (see tests/jc-initvenv.sh).
INIT_VENV_SCRIPT = Path(__file__).with_name("jc-initvenv.sh")

CONFIG_PATH = PROJECT_ROOT / "jc-config.debug.env"
_CONFIG_VAR_PATTERN = re.compile(r"\$(\w+)|\${(\w+)}")


def _interpret_bool(value: object, default: bool = False) -> bool:
    """Best-effort coercion of configuration values to booleans."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return default
        if lowered in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
        if lowered in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        return default
    return bool(value)


def load_test_config(path: Path) -> dict[str, str]:
    raw_config: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file not found: {path}") from exc

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if not value:
            raw_config[key] = ""
            continue
        if value[0] in {"'", '"'} and value[-1] == value[0]:
            try:
                parsed_value: Any = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed_value = value[1:-1]
        else:
            parsed_value = value
        raw_config[key] = parsed_value

    resolved: dict[str, Any] = {}

    def resolve(key: str, stack: set[str]) -> Any:
        if key in resolved:
            return resolved[key]
        if key not in raw_config:
            raise KeyError(key)
        if key in stack:
            raise SystemExit(
                f"Circular reference detected while resolving {key} in {path}"
            )
        stack.add(key)
        value = raw_config[key]
        if isinstance(value, str):
            def replacement(match: re.Match[str]) -> str:
                var = match.group(1) or match.group(2)
                if var is None:
                    return match.group(0)
                if var in resolved:
                    return str(resolved[var])
                if var in raw_config:
                    return str(resolve(var, stack))
                env_val = os.environ.get(var)
                return env_val if env_val is not None else match.group(0)

            expanded = _CONFIG_VAR_PATTERN.sub(replacement, value)
            expanded = os.path.expanduser(expanded)
            resolved_value: Any = expanded
        else:
            resolved_value = value
        stack.remove(key)
        resolved[key] = resolved_value
        return resolved_value

    for config_key in list(raw_config):
        resolve(config_key, set())

    def _evaluate_via_bash(key: str) -> str:
        command = (
            f"source {shlex.quote(str(path))} >/dev/null 2>&1; "
            f'printf "%s" "${{{key}}}"'
        )
        completed = subprocess.run(
            ["bash", "-lc", command],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    for key, value in list(resolved.items()):
        if isinstance(value, str) and ("$(" in value or "`" in value):
            try:
                resolved[key] = _evaluate_via_bash(key)
            except subprocess.CalledProcessError as exc:
                raise SystemExit(
                    f"Failed to evaluate {key} from {path}: {exc.stderr.strip()}"
                ) from exc

    return {k: str(v) for k, v in resolved.items()}


_CONFIG = load_test_config(CONFIG_PATH)

# Input/output locations derived from the shared configuration.
BASE_DATA_DIR = Path(_CONFIG.get("BASE_DATA_DIR", "/tmp")).expanduser()
OUTPUT_DIR = Path(
    _CONFIG.get("OUT_DIR", str(BASE_DATA_DIR / "experiments"))
).expanduser()

def resolve_file_name(cli_value: str | None) -> str:
    """Determine the target PDF file name from CLI, env, or config."""
    for candidate in (cli_value, os.environ.get("FILE_NAME"), _CONFIG.get("FILE_NAME", "")):
        if not candidate:
            continue
        value = str(candidate).strip()
        if value:
            return value
    raise SystemExit(
        f"FILE_NAME is not configured in {CONFIG_PATH} or the process environment. "
        "Provide a PDF name on the command line (e.g. `./tests/jc-test_params.py my.pdf`) "
        "or update tests/jc-config.debug.env."
    )


def _ensure_project_on_path() -> None:
    project_path = str(PROJECT_ROOT)
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


def _ensure_venv_on_path() -> None:
    """Add the local venv site-packages to sys.path when present."""
    candidates = []
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates.append(VENV_PATH / "lib" / py_ver / "site-packages")
    candidates.append(VENV_PATH / "Lib" / "site-packages")
    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def _load_generate_html_report():
    """Import camelot.generate_html_report, bootstrapping the venv if needed."""
    global _GENERATE_HTML_REPORT
    if _GENERATE_HTML_REPORT is not None:
        return _GENERATE_HTML_REPORT

    def _attempt_import():
        _ensure_project_on_path()
        _ensure_venv_on_path()
        from camelot.core import generate_html_report as _fn
        return _fn

    try:
        fn = _attempt_import()
    except ImportError:
        if AUTO_BOOTSTRAP_VENV and INIT_VENV_SCRIPT.exists():
            print(
                f"[info] camelot not importable, running {INIT_VENV_SCRIPT} to provision the venv...",
                file=sys.stderr,
            )
            subprocess.run(
                ["bash", str(INIT_VENV_SCRIPT)],
                check=True,
            )
            try:
                fn = _attempt_import()
            except ImportError as exc:
                raise SystemExit(
                    "Unable to import camelot even after bootstrapping the virtualenv."
                ) from exc
        else:
            raise

    _GENERATE_HTML_REPORT = fn
    return fn

REMOVE_BACKGROUND_ARTIFACTS_DEFAULT = _interpret_bool(
    _CONFIG.get("REMOVE_BACKGROUND_ARTIFACTS"), True
)
REMOVE_NATIVE_TEXT_DEFAULT = _interpret_bool(
    _CONFIG.get("REMOVE_NATIVE_TEXT"), True
)

# Global camelot options.
PAGES = "all"
PASSWORD: str | None = None
QUIET = False
PARALLEL = True
OUTPUT_FORMAT = "json-coords"
ZIP_RESULTS = False
SPLIT_TEXT = False
FLAG_SIZE = False
STRIP_TEXT = ""  # Characters to strip, leave empty to disable.
MARGINS: tuple[float, float, float] | None = None  # (char, line, word)
GLOBAL_EXTRA_ARGS: Sequence[str] = ()

# Default geometry hints forwarded to the lattice/stream commands.
DEFAULT_TABLE_REGIONS: Sequence[str] = ()
DEFAULT_TABLE_AREAS: Sequence[str] = ()
DEFAULT_COLUMNS: Sequence[str] = ()

# Misc quality-of-life toggles.
DRY_RUN = False
DRAW_BOXES = True
DRAW_BOXES_SCRIPT = Path(__file__).with_name("jc-draw-boxes.sh")
ANNOTATED_SUFFIX = ".pdf"

# Automatically bootstrap the venv if the Camelot CLI is missing.
AUTO_BOOTSTRAP_VENV = True
_GENERATE_HTML_REPORT = None

# Stop the run on the first failure (set to True) or keep trying other sets.
STOP_ON_ERROR = False

# Build a single HTML overview per parameter set (can be disabled per-set).
GENERATE_HTML_REPORT = True
HTML_REPORT_SUFFIX = ".html"

# Additional export formats generated after the primary run (comma separated
# values such as "csv" or "markdown"). Leave empty to skip extra exports.
ADDITIONAL_FORMATS: Sequence[str] = ()

# When providing stream column hints, replicate the first spec this many times
# to avoid Camelot raising IndexError if more tables are detected than hints.
STREAM_COLUMN_HINT_REPEAT = 8

HTML_SPEC_DEFAULTS: dict[str, object] = {
    "pages": PAGES,
    "format": OUTPUT_FORMAT,
    "zip": ZIP_RESULTS,
    "quiet": QUIET,
    "parallel": PARALLEL,
    "password": PASSWORD,
    "split_text": SPLIT_TEXT,
    "flag_size": FLAG_SIZE,
    "strip_text": STRIP_TEXT,
    "margins": MARGINS,
    "global_args": GLOBAL_EXTRA_ARGS,
    "args": (),
    "table_regions": DEFAULT_TABLE_REGIONS,
    "table_areas": DEFAULT_TABLE_AREAS,
    "columns": DEFAULT_COLUMNS,
    "column_repeat": STREAM_COLUMN_HINT_REPEAT,
    "column_pad": 0,
    "remove_background_artifacts": REMOVE_BACKGROUND_ARTIFACTS_DEFAULT,
    "remove_text": REMOVE_NATIVE_TEXT_DEFAULT,
    "draw_boxes": DRAW_BOXES,
    "html_report": GENERATE_HTML_REPORT,
    "extra_formats": ADDITIONAL_FORMATS,
}

# Parameter sets to explore. Provide command-specific arguments in "args".
PARAM_SETS: Sequence[dict[str, object]] = (
    {
        "label": "lattice_default",
        "flavor": "lattice",
        "args": [],
    },
    {
        "label": "lattice_background",
        "flavor": "lattice",
        "args": [
            "--process_background",
            "--line_scale",
            "36",
            "--line_tol",
            "2",
            "--joint_tol",
            "2",
        ],
    },
    {
        "label": "lattice_high_res",
        "flavor": "lattice",
        "args": [
            "--resolution",
            "400",
            "--line_scale",
            "28",
            "--threshold_blocksize",
            "17",
            "--threshold_constant",
            "-1",
            "--iterations",
            "1",
        ],
    },
    {
        "label": "lattice_copy_flow",
        "flavor": "lattice",
        "args": [
            "--copy_text",
            "h",
            "--shift_text",
            "l",
            "--shift_text",
            "t",
        ],
    },
    {
        "label": "stream_columns_hint",
        "flavor": "stream",
        "args": [
            "--edge_tol",
            "25",
            "--row_tol",
            "1",
            "--column_tol",
            "1",
        ],
        "columns": ["70,180,310"],
        "column_repeat": 6,
        "table_regions": DEFAULT_TABLE_REGIONS,
        "table_areas": DEFAULT_TABLE_AREAS,
    },
)
# ---------------------------------------------------------------------------

FORMAT_EXTENSIONS = {
    "csv": ".csv",
    "excel": ".xlsx",
    "html": ".html",
    "json": ".json",
    "json-coords": ".json",
    "markdown": ".md",
    "sqlite": ".db",
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Camelot experiments defined in tests/jc-test_params.py"
    )
    parser.add_argument(
        "file_name",
        nargs="?",
        help="PDF file name (relative to BASE_DATA_DIR) or absolute path to analyse.",
    )
    args = parser.parse_args()

    file_name = resolve_file_name(args.file_name)
    os.environ["FILE_NAME"] = file_name

    camelot_bin = resolve_camelot_bin()
    pdf_path = (BASE_DATA_DIR / file_name).expanduser()
    if not pdf_path.exists():
        raise SystemExit(f"Input PDF not found: {pdf_path}")

    output_dir = OUTPUT_DIR.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    draw_boxes_script = DRAW_BOXES_SCRIPT.expanduser()
    if DRAW_BOXES and not draw_boxes_script.exists():
        print(
            f"[warn] jc-draw-boxes helper not found at {draw_boxes_script}. "
            "Overlay generation will be skipped.",
            file=sys.stderr,
        )
        draw_boxes_script = None
    elif not DRAW_BOXES:
        draw_boxes_script = None

    outcomes: list[tuple[str, bool]] = []
    for param in PARAM_SETS:
        label = str(param.get("label", "") or "").strip() or "<unnamed>"
        success = run_param_set(
            camelot_bin=camelot_bin,
            pdf_path=pdf_path,
            output_dir=output_dir,
            draw_boxes_script=draw_boxes_script,
            specification=param,
        )
        outcomes.append((label, success))

    failures = [label for label, success in outcomes if not success]
    if failures:
        print(
            f"\n[warn] {len(failures)} parameter set(s) failed: {', '.join(filter(None, failures))}",
            file=sys.stderr,
        )
        if STOP_ON_ERROR:
            print("[info] STOP_ON_ERROR is enabled, script exited on first failure.", file=sys.stderr)
        else:
            print("[info] Set STOP_ON_ERROR = True to abort on the first failure.", file=sys.stderr)
    else:
        print("\n[info] All parameter sets completed successfully.")


def resolve_camelot_bin() -> Path:
    """Return the camelot executable, preferring the configured venv."""
    bootstrap_attempted = False
    if CAMELOT_BIN:
        candidate = Path(CAMELOT_BIN).expanduser()
        if candidate.exists():
            return candidate
        if AUTO_BOOTSTRAP_VENV and INIT_VENV_SCRIPT.exists():
            print(
                f"[info] camelot binary missing at {candidate}, running {INIT_VENV_SCRIPT}...",
                file=sys.stderr,
            )
            subprocess.run(
                ["bash", str(INIT_VENV_SCRIPT)],
                check=True,
            )
            bootstrap_attempted = True
            if candidate.exists():
                return candidate
        if INIT_VENV_SCRIPT.exists():
            if bootstrap_attempted:
                hint = f" {INIT_VENV_SCRIPT} completed but camelot is still unavailable."
            elif AUTO_BOOTSTRAP_VENV:
                hint = f" {INIT_VENV_SCRIPT} is available for bootstrapping the virtualenv."
            else:
                hint = f" Run {INIT_VENV_SCRIPT} to create the project virtualenv."
        else:
            hint = ""
        print(
            (
                f"[warn] camelot binary not found at {candidate}, falling back to PATH."
                f"{hint}"
            ),
            file=sys.stderr,
        )
    fallback = shutil.which("camelot")
    if fallback:
        return Path(fallback)
    if INIT_VENV_SCRIPT.exists():
        if bootstrap_attempted or AUTO_BOOTSTRAP_VENV:
            hint = f" {INIT_VENV_SCRIPT} completed but camelot is still unavailable."
        else:
            hint = f" Run {INIT_VENV_SCRIPT} to provision it."
    else:
        hint = ""
    raise SystemExit(
        "Unable to locate the camelot executable. Update CAMELOT_BIN or ensure "
        f"camelot is available on PATH.{hint}"
    )


def run_param_set(
    camelot_bin: Path,
    pdf_path: Path,
    output_dir: Path,
    draw_boxes_script: Path | None,
    specification: dict[str, object],
) -> bool:
    label = str(specification.get("label") or "").strip()
    if not label:
        raise ValueError("Each parameter set must define a non-empty 'label'.")
    flavor = str(specification.get("flavor", "lattice"))
    if flavor not in {"lattice", "stream", "hybrid"}:
        raise ValueError(f"Unsupported flavor '{flavor}' in set '{label}'.")

    pages = str(specification.get("pages", PAGES))
    fmt = str(specification.get("format", OUTPUT_FORMAT))
    fmt_extension = FORMAT_EXTENSIONS.get(fmt, ".out")
    zip_results = bool(specification.get("zip", ZIP_RESULTS))
    quiet = bool(specification.get("quiet", QUIET))
    parallel = bool(specification.get("parallel", PARALLEL))

    password = specification.get("password", PASSWORD)
    strip_text = specification.get("strip_text", STRIP_TEXT)
    split_text = bool(specification.get("split_text", SPLIT_TEXT))
    flag_size = bool(specification.get("flag_size", FLAG_SIZE))
    margins = specification.get("margins", MARGINS)

    base_name = f"{pdf_path.stem}.{label}.{flavor}"
    output_file = output_dir / f"{base_name}{fmt_extension}"

    def build_command_for_format(
        target_format: str, target_output: Path
    ) -> tuple[list[str], list[str]]:
        cmd: list[str] = [str(camelot_bin), "-p", pages]

        if password:
            cmd.extend(["--password", str(password)])
        if parallel:
            cmd.append("--parallel")
        if quiet:
            cmd.append("--quiet")
        cmd.extend(["--format", target_format])
        cmd.extend(["--output", str(target_output)])
        if zip_results:
            cmd.append("--zip")
        if split_text:
            cmd.append("--split_text")
        if flag_size:
            cmd.append("--flag_size")
        if strip_text:
            cmd.extend(["--strip_text", str(strip_text)])
        if margins:
            if not isinstance(margins, Sequence) or len(margins) != 3:
                raise ValueError(
                    "Margins must be an iterable with three values (char, line, word)."
                )
            cmd.extend(["--margins", *(str(value) for value in margins)])
        cmd.extend(str(arg) for arg in GLOBAL_EXTRA_ARGS)
        cmd.extend(str(arg) for arg in specification.get("global_args", []))
        cmd.append(flavor)
        flavor_args = _compose_flavor_args(
            flavor=flavor,
            specification=specification,
        )
        cmd.extend(flavor_args)
        cmd.append(str(pdf_path))
        return cmd, flavor_args

    command, flavor_args = build_command_for_format(fmt, output_file)

    print(f"\n[{label}]")
    command_str = format_command(command)
    print(f"$ {command_str}")

    if DRY_RUN:
        return True

    overall_success = True
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"! command failed with exit code {exc.returncode}: {format_command(command)}",
            file=sys.stderr,
        )
        if STOP_ON_ERROR:
            raise
        return False

    print(f"→ results: {output_file}")

    should_draw = (
        draw_boxes_script is not None
        and str(specification.get("draw_boxes", DRAW_BOXES)).lower() != "false"
        and fmt == "json-coords"
        and not zip_results
    )

    if should_draw:
        overlay_path = output_dir / f"{base_name}{ANNOTATED_SUFFIX}"
        draw_command = [
            str(draw_boxes_script),
            str(output_file),
            str(pdf_path),
            str(overlay_path),
        ]
        print(f"$ {format_command(draw_command)}")
        try:
            subprocess.run(draw_command, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"! jc-draw-boxes failed with exit code {exc.returncode}: {format_command(draw_command)}",
                file=sys.stderr,
            )
            if STOP_ON_ERROR:
                raise
            overall_success = False
        else:
            print(f"→ overlay: {overlay_path}")

    html_report_enabled = bool(specification.get("html_report", GENERATE_HTML_REPORT))
    if html_report_enabled:
        html_report_path = output_dir / f"{base_name}{HTML_REPORT_SUFFIX}"
        try:
            generate_html_report_fn = _load_generate_html_report()
            generate_html_report_fn(
                tables_json=output_file,
                html_output=html_report_path,
                label=label,
                flavor=flavor,
                pdf_path=pdf_path,
                command_preview=command_str,
                specification=specification,
                command_parts=command,
                flavor_args=flavor_args,
                specification_defaults=HTML_SPEC_DEFAULTS,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"! html report failed: {exc}",
                file=sys.stderr,
            )
            if STOP_ON_ERROR:
                raise
            overall_success = False
        else:
            print(f"→ html report: {html_report_path}")

    extra_formats_value = specification.get("extra_formats", ADDITIONAL_FORMATS)
    if extra_formats_value is None:
        extra_format_list: list[str] = []
    else:
        extra_format_list = [fmt_str for fmt_str in _normalized_sequence(extra_formats_value) if fmt_str]
    extra_format_list = list(dict.fromkeys(f for f in extra_format_list if f != fmt))

    for extra_fmt in extra_format_list:
        extra_ext = FORMAT_EXTENSIONS.get(extra_fmt, ".out")
        extra_output = output_dir / f"{base_name}{extra_ext}"
        extra_command, _ = build_command_for_format(extra_fmt, extra_output)
        print(f"$ {format_command(extra_command)}")
        try:
            subprocess.run(extra_command, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"! command failed with exit code {exc.returncode}: {format_command(extra_command)}",
                file=sys.stderr,
            )
            if STOP_ON_ERROR:
                raise
            overall_success = False
        else:
            print(f"→ results: {extra_output}")

    return overall_success


def _compose_flavor_args(
    flavor: str,
    specification: dict[str, object],
) -> list[str]:
    args: list[str] = []
    table_regions = specification.get("table_regions", DEFAULT_TABLE_REGIONS)
    table_areas = specification.get("table_areas", DEFAULT_TABLE_AREAS)
    columns = specification.get("columns", DEFAULT_COLUMNS)

    for region in _normalized_sequence(table_regions):
        args.extend(["--table_regions", region])
    for area in _normalized_sequence(table_areas):
        args.extend(["--table_areas", area])
    if flavor in {"stream", "hybrid"}:
        column_specs = [spec for spec in _normalized_sequence(columns) if spec]
        if column_specs:
            repeat = int(specification.get("column_repeat", STREAM_COLUMN_HINT_REPEAT))
            if repeat and len(column_specs) == 1:
                column_specs = column_specs * repeat
            pad = int(specification.get("column_pad", 0))
            if pad > 0:
                column_specs.extend([""] * pad)
        for column in column_specs:
            args.extend(["--columns", column])

    if flavor in {"lattice", "hybrid", "stream"}:
        remove_background_value = specification.get("remove_background_artifacts")
        if remove_background_value is None:
            remove_background = REMOVE_BACKGROUND_ARTIFACTS_DEFAULT
        else:
            remove_background = _interpret_bool(
                remove_background_value, REMOVE_BACKGROUND_ARTIFACTS_DEFAULT
            )
        positive_flag = "--remove_background_artifacts"
        negative_flag = "--no-remove_background_artifacts"
        if positive_flag in args or negative_flag in args:
            pass
        elif remove_background:
            args.append(positive_flag)
        else:
            args.append(negative_flag)

    if flavor in {"lattice", "hybrid"}:
        remove_text_value = specification.get("remove_text")
        if remove_text_value is None:
            remove_text = REMOVE_NATIVE_TEXT_DEFAULT
        else:
            remove_text = _interpret_bool(remove_text_value, REMOVE_NATIVE_TEXT_DEFAULT)
        positive_flag = "--remove_text"
        negative_flag = "--no-remove_text"
        if positive_flag in args or negative_flag in args:
            pass
        elif remove_text:
            args.append(positive_flag)
        else:
            args.append(negative_flag)

    plot_type = specification.get("plot")
    if plot_type:
        args.extend(["--plot_type", str(plot_type)])

    args.extend(str(value) for value in specification.get("args", []))
    return args


def _normalized_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def format_command(parts: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


if __name__ == "__main__":
    main()
