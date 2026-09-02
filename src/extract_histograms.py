"""Command-line wrapper for the ROOT histogram plotting API.

The plotting implementation itself lives in :mod:`lib.core` and
:mod:`lib.plotting`. This module is responsible for CLI parsing, cache
handling, output paths, SVG writing and opening the generated file.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import uproot

# Support running this file directly from a source checkout.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lib.config import PlotConfig
from lib.constants import green, red, reset
from lib.canvas import (
    extract_canvas_histogram,
    find_canvas_histograms,
    is_canvas_reference,
    parse_canvas_reference,
)
from lib.core import plot_from_root
from lib.utils import (
    find_histograms,
    getMember,
    load_cache,
    optimize_svg,
    parse_limits,
    request_additional_config,
    save_cache,
    select_histogram_keys,
)


PathLike = Union[str, os.PathLike[str]]
RunNumber = Union[int, str]
HistogramPaths = Optional[List[List[str]]]


def parse_run_number(run_str: Optional[str]) -> Optional[RunNumber]:
    """Parse a single run number or a run range."""
    if run_str is None:
        return None

    if re.match(r"^\d+$", run_str):
        return int(run_str)

    if re.match(r"^\d+-\d+$", run_str):
        return run_str

    raise ValueError(
        "Run number must be a single number or range "
        f"(e.g., '510' or '510-515'), got '{run_str}'"
    )


def _default_anapath(value: Optional[PathLike]) -> Path:
    """Resolve the analysis base path without storing global state."""
    return Path(value if value is not None else os.getcwd()).expanduser().resolve()


def _resolve_input_root_path(value: PathLike) -> Path:
    """Resolve a direct ROOT input relative to the current directory."""
    return Path(value).expanduser().resolve()


def _config_from_cache(cache_config: Dict[str, Any]) -> PlotConfig:
    """Restore a :class:`PlotConfig` from the legacy cache representation."""
    config = PlotConfig()

    for key, parameter in cache_config.items():
        # Cache entries created before the refactor store
        # (value, used, hidden). Be slightly defensive for malformed entries.
        if isinstance(parameter, (list, tuple)) and len(parameter) == 3:
            value, used, hidden = parameter
            config.set(key, value, used=used, hidden=hidden)
        else:
            config.set(key, parameter)

    return config


def _collect_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Return only plotting options explicitly supplied on the command line."""
    arg_dict = vars(args)
    overrides: Dict[str, Any] = {}

    cli_to_config = {
        "x_log": "x-log",
        "y_log": "y-log",
        "z_log": "z-log",
        "model_stats": "model-stats",
    }

    for argument_name, config_key in cli_to_config.items():
        if argument_name in arg_dict and arg_dict[argument_name] is not argparse.SUPPRESS:
            overrides[config_key] = arg_dict[argument_name]

    direct_keys = {
        "title", "limits", "legend", "stats", "errors", "colors", "palette",
        "alphas", "thickness", "grid", "models", "cutoff", "flat",
        "angles", "raster",
    }

    for key in direct_keys:
        if key not in arg_dict or arg_dict[key] is argparse.SUPPRESS:
            continue
        value = arg_dict[key]
        # --title without an argument means use the histogram title.
        overrides[key] = None if key == "title" and value == "" else value

    return overrides


def _apply_config_overrides(config: PlotConfig, overrides: Dict[str, Any]) -> None:
    """Apply explicitly supplied CLI overrides to a config object."""
    for key, value in overrides.items():
        config.set(key, value)


def _resolve_root_path(
    anapath: Path,
    run: Optional[RunNumber],
    type_name: Optional[str],
    comment: Optional[str],
) -> Path:
    """Build the ROOT path using the original naming convention."""
    if run is None or type_name is None:
        raise ValueError("Both run and type are required to construct a ROOT path.")

    filename = f"histograms_ana_{run}"
    if comment:
        filename += f"-{comment}"
    filename += ".root"
    return anapath / type_name / filename


def _resolve_matching_keys(
    root_path: Path,
    paths: HistogramPaths,
    matching_keys: List[str],
    stack: bool,
) -> List[str]:
    """Resolve histogram keys for interactive configuration when needed."""
    if matching_keys or not paths:
        return matching_keys

    with uproot.open(root_path) as root_file:
        found_keys: List[str] = []
        for path_keywords in paths:
            keys = find_histograms(
                root_file,
                path_keywords,
                stack or (len(paths) > 1 and len(path_keywords) > 1),
                quiet=True,
            )
            found_keys.extend(keys)

    if not found_keys and paths:
        canvas_references = find_canvas_histograms(root_path)
        for path_keywords in paths:
            matches = [
                reference
                for reference, _canvas, _pad, _histogram in canvas_references
                if all(word.lower() in reference.lower() for word in path_keywords)
            ]
            found_keys.extend(
                select_histogram_keys(
                    matches,
                    {reference: reference.split("|title=", 1)[-1] for reference in matches},
                    stack=stack,
                    display_paths={
                        reference: (
                            f"canvas/{canvas}/pad_{pad}/{name}"
                        )
                        for reference, canvas, pad, _histogram in canvas_references
                        for name in [reference.split("|name=", 1)[-1].split("|title=", 1)[0]]
                        if reference in matches
                    },
                )
            )

    return list(dict.fromkeys(found_keys))


def _load_histograms_for_interactive_config(
    root_path: Path,
    matching_keys: Sequence[str],
) -> List[Any]:
    """Load selected ROOT objects for the interactive config UI."""
    hist_list: List[Any] = []
    temporary_roots = []
    temporary_paths = []

    with uproot.open(root_path) as root_file:
        classnames = root_file.classnames()
        for key in matching_keys:
            try:
                if is_canvas_reference(key):
                    canvas_key, pad_index, histogram_index = parse_canvas_reference(key)
                    temporary_path = extract_canvas_histogram(
                        root_path,
                        canvas_key,
                        pad_index=pad_index,
                        histogram_index=histogram_index,
                    )
                    temporary_root = uproot.open(temporary_path)
                    temporary_paths.append(temporary_path)
                    temporary_roots.append(temporary_root)
                    hist = temporary_root["selected_histogram"]
                else:
                    hist = root_file[key]
                if not hasattr(hist, "title"):
                    setattr(hist, "title", getMember(hist, "fTitle", "Untitled"))
                hist_list.append(hist)
                print(
                    f"  - {getattr(hist, 'classname', 'TH1')}: {green}{hist.title}{reset}"
                )
            except uproot.exceptions.KeyInFileError as exc:
                raise KeyError(
                    f"Key '{key}' not found in ROOT file {root_path}."
                ) from exc

    return hist_list, temporary_roots, temporary_paths


def _sanitize_title(value: Any) -> str:
    return re.sub(r"[^\w\-_.]", "_", str(value or "plot"))


def _extra_artists(fig: plt.Figure) -> List[Any]:
    """Retrieve artists retained by the core API for faithful SVG output."""
    return list(getattr(fig, "_histogram_extra_artists", []))


def extract_and_plot(
    run: Optional[RunNumber],
    type_name: Optional[str],
    *,
    anapath: PathLike,
    config: PlotConfig,
    cli_overrides: Optional[Dict[str, Any]] = None,
    comment: Optional[str] = None,
    paths: HistogramPaths = None,
    config_interactive: bool = False,
    stack: bool = False,
    edit_index: Optional[int] = None,
    root_file: Optional[PathLike] = None,
    output: Optional[PathLike] = None,
) -> Optional[str]:
    """CLI-level orchestration for loading, plotting, saving and caching."""
    analysis_path = _default_anapath(anapath)

    if edit_index is None:
        _apply_config_overrides(config, cli_overrides or {})

    matching_keys: List[str] = []
    root_path: Optional[Path] = None
    cached_output_path = ""
    cached_title = "Histogram"

    # ------------------------------------------------------------------
    # Cache / input selection
    # ------------------------------------------------------------------
    if edit_index is not None:
        load_result = load_cache(str(analysis_path), edit_index)
        if load_result is None:
            return None

        cached_config, matching_keys, cached_root_path, cached_output_path, cached_title = load_result
        config = _config_from_cache(cached_config)
        _apply_config_overrides(config, cli_overrides or {})

        new_root_file_specified = (
            root_file is not None
            or run is not None
            or type_name is not None
            or comment is not None
        )

        if new_root_file_specified:
            print("New run/type/comment provided, ignoring cached root file path.")
            if root_file is None and (run is None or type_name is None):
                print(
                    f"{red}ERROR: --run and --type are required when overriding cache file.{reset}",
                    file=sys.stderr,
                )
                return None

            root_path = (
                _resolve_input_root_path(root_file)
                if root_file is not None
                else _resolve_root_path(analysis_path, run, type_name, comment)
            )
            print(f"Using new ROOT file path: {root_path}")

            if paths:
                matching_keys = []
            else:
                print(
                    "No --path provided with new file, attempting to use cached keys (might fail)."
                )
        else:
            root_path = Path(cached_root_path)
            print(f"Using cached ROOT file: {root_path}")
            if paths:
                print("Using provided --path arguments, ignoring cached keys.")
                matching_keys = []
            else:
                print("Using cached histogram keys.")

    else:
        if root_file is None and (run is None or type_name is None):
            print(
                f"{red}ERROR: --run and --type are required unless --root-file is used or --edit is used.{reset}",
                file=sys.stderr,
            )
            return None

        root_path = (
            _resolve_input_root_path(root_file)
            if root_file is not None
            else _resolve_root_path(analysis_path, run, type_name, comment)
        )
        if root_path.is_file():
            print(f"DEBUG: Found ROOT file at: {root_path}")
        else:
            print(
                f"{red}ERROR: The data path '{root_path}' is not valid.{reset}",
                file=sys.stderr,
            )
            return None

    assert root_path is not None

    # Resolve the selection once. The same keys are used for plotting and
    # caching, so the user is never asked to choose the same histogram twice.
    if not matching_keys and paths:
        matching_keys = _resolve_matching_keys(root_path, paths, [], stack)
        if not matching_keys:
            print(
                f"{red}ERROR: No histograms found for the requested selection.{reset}",
                file=sys.stderr,
            )
            return None

    # Apply CLI plotting overrides after a cached config has been restored.
    # New plots start from PlotConfig defaults, while --edit starts from cache.
    # The caller applies the CLI overrides before this function for both cases.

    # ------------------------------------------------------------------
    # Interactive configuration
    # ------------------------------------------------------------------
    if config_interactive:
        try:
            matching_keys = _resolve_matching_keys(root_path, paths, matching_keys, stack)
            if not matching_keys:
                print(
                    f"{red}ERROR: No histograms found for configuration.{reset}",
                    file=sys.stderr,
                )
                return None

            hist_list, temporary_roots, temporary_paths = _load_histograms_for_interactive_config(
                root_path, matching_keys
            )
            if not hist_list:
                print(f"{red}ERROR: No valid histograms could be loaded.{reset}", file=sys.stderr)
                return None

            try:
                request_additional_config(hist_list, config)
            finally:
                for temporary_root in temporary_roots:
                    temporary_root.close()
                for temporary_path in temporary_paths:
                    temporary_path.unlink(missing_ok=True)

        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"{red}ERROR: {exc}{reset}", file=sys.stderr)
            return None
        except Exception as exc:
            print(f"{red}An unexpected error occurred: {exc}{reset}", file=sys.stderr)
            return None

    # ------------------------------------------------------------------
    # Core plotting
    # ------------------------------------------------------------------
    try:
        fig = plot_from_root(
            root_path,
            paths=paths if paths else None,
            config=config,
            histogram_keys=matching_keys if matching_keys else None,
            stack=stack,
        )
    except FileNotFoundError:
        print(f"{red}ERROR: ROOT file not found: {root_path}{reset}", file=sys.stderr)
        return None
    except KeyError as exc:
        print(f"{red}ERROR: {exc}{reset}", file=sys.stderr)
        return None
    except ValueError as exc:
        print(f"{red}ERROR: {exc}{reset}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"{red}An unexpected error occurred: {exc}{reset}", file=sys.stderr)
        return None

    # ------------------------------------------------------------------
    # Output path and SVG I/O
    # ------------------------------------------------------------------
    output_base_dir = analysis_path / "output" / (type_name or "cached")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if edit_index is not None and cached_output_path:
        cached_dir = Path(cached_output_path).parent
        if cached_dir.is_dir():
            output_base_dir = cached_dir

    current_run = run if run is not None else "cached"
    current_type = type_name if type_name is not None else "cached"
    title = config.get("title", cached_title or "plot")

    filename_parts = [f"run_{current_run}", current_type]
    if comment:
        filename_parts.append(comment)
    filename_parts.append(_sanitize_title(title))
    output_path = Path(output).expanduser() if output else (
        output_base_dir / ("-".join(filename_parts) + ".svg")
    )
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving plot to: {output_path}")
    try:
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=_extra_artists(fig),
        )
    except Exception as exc:
        print(f"{red}ERROR: Failed to save figure: {exc}{reset}", file=sys.stderr)
        plt.close(fig)
        return None
    finally:
        plt.close(fig)

    # ------------------------------------------------------------------
    # Cache and SVG optimization
    # ------------------------------------------------------------------
    cache_title_parts = [str(run) if run else "edit", type_name if type_name else "edit"]
    if comment:
        cache_title_parts.append(comment)
    cache_title_parts.append(config.get("title", "plot"))
    cache_title = " - ".join(filter(None, cache_title_parts))

    save_cache(
        matching_keys,
        str(root_path),
        str(output_path),
        cache_title,
        config,
        str(analysis_path),
        edit_index if edit_index != -1 else None,
    )

    if not config.get("raster", False):
        optimize_svg(str(output_path))

    return str(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and plot ROOT histograms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_argument_group("Input Data Selection")
    input_group.add_argument(
        "--anapath",
        type=str,
        default=None,
        help="Base path for analysis data and output. Defaults to the current working directory.",
    )
    input_group.add_argument(
        "--run",
        type=str,
        help="Run number (e.g., '510') or range (e.g., '510-515') (required unless --edit is used)",
    )
    input_group.add_argument(
        "--type",
        help="Data type (e.g., calibration, physics) (required unless --edit is used)",
    )
    input_group.add_argument("--comment", default=None, help="Optional comment for file naming")
    input_group.add_argument(
        "--path",
        action="append",
        nargs="+",
        help=(
            "Keyword(s) to find histogram(s) in ROOT file. Can be used multiple times. "
            "Example: --path EventLoader layer_0 corr --path Trigger time"
        ),
    )
    input_group.add_argument(
        "--root-file",
        help="Read a ROOT file directly; relative paths use the current directory.",
    )
    input_group.add_argument("--output", help="Output SVG path.")

    plot_group = parser.add_argument_group("Plotting Configuration (Overrides cache/defaults)")
    plot_group.add_argument("--title", nargs="?", const="", default=argparse.SUPPRESS)
    plot_group.add_argument("--limits", type=parse_limits, default=argparse.SUPPRESS)
    plot_group.add_argument("--legend", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    plot_group.add_argument("--stats", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    plot_group.add_argument("--errors", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    plot_group.add_argument("--colors", nargs="+", default=argparse.SUPPRESS)
    plot_group.add_argument("--palette", default=argparse.SUPPRESS)
    plot_group.add_argument("--alphas", nargs="+", type=float, default=argparse.SUPPRESS)
    plot_group.add_argument("--thickness", type=float, default=argparse.SUPPRESS)
    plot_group.add_argument("--grid", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    plot_group.add_argument("--x-log", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, dest="x_log")
    plot_group.add_argument("--y-log", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, dest="y_log")
    plot_group.add_argument("--z-log", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, dest="z_log")
    plot_group.add_argument("--models", nargs="*", default=argparse.SUPPRESS)
    plot_group.add_argument("--model-stats", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, dest="model_stats")
    plot_group.add_argument("--cutoff", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    th3_group = parser.add_argument_group("TH3 Specific Options")
    th3_group.add_argument("--flat", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    th3_group.add_argument("--angles", nargs=2, type=float, default=argparse.SUPPRESS)

    behavior_group = parser.add_argument_group("Script Behavior")
    behavior_group.add_argument("--config", action="store_true", help="Enter interactive configuration mode")
    behavior_group.add_argument("--stack", action="store_true", help="Attempt to stack multiple found histograms")
    behavior_group.add_argument("--raster", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    behavior_group.add_argument(
        "--edit",
        nargs="?",
        const=-1,
        type=int,
        default=None,
        help="Edit previous plot configuration. Optionally provide cache index (e.g., --edit 0 for latest).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.run = parse_run_number(args.run)
    except ValueError as exc:
        print(f"{red}ERROR: {exc}{reset}", file=sys.stderr)
        raise SystemExit(1) from exc

    anapath = _default_anapath(args.anapath)
    print(f"\nDEBUG: Analysis path: {anapath}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")

    # New plots start from defaults. For --edit the selected cache entry is
    # restored inside extract_and_plot(), then explicitly supplied CLI options
    # are applied there by this initial pass only for fresh plots. To keep the
    # precedence rule consistent, apply CLI options after cache restoration
    # below by passing them through a local config object for fresh plots.
    config = PlotConfig()
    cli_overrides = _collect_cli_overrides(args)

    output_file = extract_and_plot(
        args.run,
        args.type,
        anapath=anapath,
        config=config,
        cli_overrides=cli_overrides,
        comment=args.comment,
        paths=args.path,
        config_interactive=args.config or (args.edit is not None),
        stack=args.stack,
        edit_index=args.edit,
        root_file=args.root_file,
        output=args.output,
    )

    if output_file:
        print(f"\n{green}Successfully created: {output_file}{reset}")
        try:
            if sys.platform == "win32":
                os.startfile(output_file)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", output_file], check=True)
            else:
                subprocess.run(["xdg-open", output_file], check=True)
        except Exception as exc:
            print(f"Info: Could not automatically open the output file: {exc}", file=sys.stderr)
    else:
        print(f"\n{red}Plot generation failed.{reset}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
