"""Core API for extracting and plotting ROOT histograms.

This module contains the application-independent workflow for turning a ROOT
file and histogram selection into a Matplotlib figure. It deliberately does
not know about the command line, cache files, output filenames, SVG
optimization, or opening generated files.
"""

from pathlib import Path
import json
import shutil
import subprocess
from typing import List, Optional, Sequence, Union
from contextlib import suppress
from .canvas import (
    extract_canvas_histogram,
    find_canvas_histograms,
    is_canvas_reference,
    parse_canvas_reference,
)

import matplotlib.pyplot as plt
import uproot
from matplotlib.figure import Figure

from .config import PlotConfig
from .plotting import process_histograms
from .utils import find_histograms, getMember, refresh_config, select_histogram_keys


PathLike = Union[str, Path]
HistogramPath = Sequence[str]


def _find_matching_keys(
    root_file,
    paths: Sequence[HistogramPath],
    *,
    stack: bool,
    selection: str,
) -> List[str]:
    """Find histogram keys for the requested keyword paths.

    This preserves the existing search behavior, including the stacking
    heuristic used by the command-line script.
    """
    matching_keys: List[str] = []

    for path_keywords in paths:
        should_stack = stack or (len(paths) > 1 and len(path_keywords) > 1)
        matching_keys.extend(
            find_histograms(
                root_file,
                list(path_keywords),
                should_stack,
                quiet=True,
                selection=selection,
            )
        )

    if not matching_keys:
        root_path = getattr(getattr(root_file, "file", None), "file_path", None)
        if root_path:
            canvas_references = find_canvas_histograms(root_path)
            for path_keywords in paths:
                matches = [
                    reference
                    for reference, _canvas, _pad, _histogram in canvas_references
                    if all(word.lower() in reference.lower() for word in path_keywords)
                ]
                matching_keys.extend(
                    select_histogram_keys(
                        matches,
                        {reference: reference.split("|title=", 1)[-1] for reference in matches},
                        stack=stack,
                        selection=selection,
                        query="".join(path_keywords),
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

    # Preserve insertion order while removing duplicates, as the old
    # extraction workflow did before loading the histograms.
    return list(dict.fromkeys(matching_keys))


def _load_histograms(root_file, matching_keys: Sequence[str], root_path: Path):
    """Load the selected ROOT objects and ensure they have a usable title."""
    hist_list = []
    temporary_roots = []
    temporary_paths = []

    for key in matching_keys:
        if is_canvas_reference(key):
            canvas_key, pad_index, histogram_index = parse_canvas_reference(key)
            temporary_path = extract_canvas_histogram(
                root_path,
                canvas_key,
                pad_index=pad_index,
                histogram_index=histogram_index,
            )
            temporary_root = uproot.open(temporary_path)
            temporary_roots.append(temporary_root)
            temporary_paths.append(temporary_path)
            hist = temporary_root["selected_histogram"]
            hist_list.append(hist)
            continue
        try:
            hist = root_file[key]
        except uproot.exceptions.KeyInFileError as exc:
            raise KeyError(f"Histogram key '{key}' was not found in the ROOT file.") from exc

        # Preserve the existing title fallback used by the CLI workflow.
        if not hasattr(hist, "title"):
            setattr(hist, "title", getMember(hist, "fTitle", "Untitled"))

        hist_list.append(hist)

    return hist_list, temporary_roots, temporary_paths


def _attach_embedded_fits(
    root_path: Path,
    matching_keys: Sequence[str],
    hist_list: Sequence,
) -> None:
    """Attach sampled TF1 functions stored in selected TH1 objects."""
    root_command = shutil.which("root")
    if root_command is None:
        raise RuntimeError("Plotting embedded fits requires the CERN ROOT executable on PATH.")

    for key, hist in zip(matching_keys, hist_list):
        if is_canvas_reference(key) or "TH1" not in getattr(hist, "classname", ""):
            continue

        root_key = key.split(";", 1)[0]
        expression = (
            f"TFile *input=TFile::Open({json.dumps(str(root_path))}); "
            f"if(!input||input->IsZombie()) gSystem->Exit(2); "
            f"TH1 *hist=(TH1*)input->Get({json.dumps(root_key)}); "
            f"if(!hist) gSystem->Exit(3); "
            f"TIter functions(hist->GetListOfFunctions()); TObject *object; "
            f"while((object=functions())) {{ if(!object->InheritsFrom(\"TF1\")) continue; "
            f"TF1 *fit=(TF1*)object; int count=fit->GetNpx(); "
            f"for(int i=0;i<count;++i) {{ double x=fit->GetXmin()+(fit->GetXmax()-fit->GetXmin())*i/(count-1); "
            f"std::cout << \"__EMBEDDED_FIT__\\t\" << x << \"\\t\" << fit->Eval(x) << std::endl; }} "
            f"std::cout << \"__EMBEDDED_FIT_TITLE__\\t\" << fit->GetName() << std::endl; break; }} "
            f"input->Close();"
        )
        result = subprocess.run(
            [root_command, "-l", "-b", "-q", "-e", expression],
            capture_output=True,
            text=True,
            check=False,
        )
        fit_values = []
        fit_title = "embedded fit"
        for line in result.stdout.splitlines():
            if line.startswith("__EMBEDDED_FIT__\t"):
                _, x_value, y_value = line.split("\t", 2)
                fit_values.append((float(x_value), float(y_value)))
            elif line.startswith("__EMBEDDED_FIT_TITLE__\t"):
                fit_title = line.split("\t", 1)[1]
        if fit_values:
            hist._embedded_fit_points = tuple(zip(*fit_values))
            hist._embedded_fit_title = fit_title


def plot_from_root(
    root_path: PathLike,
    paths: Optional[Sequence[HistogramPath]] = None,
    *,
    config: Optional[PlotConfig] = None,
    histogram_keys: Optional[Sequence[str]] = None,
    stack: bool = False,
    selection: str = "prompt",
) -> Figure:
    """Read ROOT histograms and return their Matplotlib figure.

    Parameters
    ----------
    root_path:
        Path to the ROOT file to read.
    paths:
        Keyword paths used to find histograms in the ROOT file. This is the
        same selection mechanism used by the existing CLI. Required when
        ``histogram_keys`` is not supplied.
    config:
        Plot configuration. A fresh :class:`PlotConfig` is created when not
        supplied. The object is updated in place with histogram-dependent
        defaults, matching the existing ``refresh_config`` behavior.
    histogram_keys:
        Explicit ROOT keys to load. This is useful for callers that already
        have resolved keys, for example a future cache-aware CLI wrapper.
        When supplied, ``paths`` is ignored.
    stack:
        Request stacking behavior when supported by ``find_histograms``.
    selection:
        Resolve ambiguous keyword matches with ``"prompt"`` (the default) or
        choose the closest matching histogram basename with ``"most_similar"``.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure. The caller owns the returned figure and is
        responsible for displaying or saving it.

    Raises
    ------
    ValueError
        If neither ``paths`` nor ``histogram_keys`` selects any histograms.
    FileNotFoundError
        If the ROOT file does not exist.
    KeyError
        If an explicitly selected histogram key is missing.
    RuntimeError
        If the selected objects are unsupported by the plotting layer.
    """
    root_file_path = Path(root_path)
    if not root_file_path.is_file():
        raise FileNotFoundError(f"ROOT file not found: {root_file_path}")

    if histogram_keys is None and not paths:
        raise ValueError("Either 'paths' or 'histogram_keys' must select at least one histogram.")

    plot_config = config if config is not None else PlotConfig()

    temporary_roots = []
    temporary_paths = []
    try:
        with uproot.open(root_file_path) as root_file:
            if histogram_keys is not None:
                matching_keys = list(histogram_keys)
            else:
                matching_keys = _find_matching_keys(
                    root_file,
                    paths or [],
                    stack=stack,
                    selection=selection,
                )

            if not matching_keys:
                raise ValueError("No histograms matched the requested selection.")

            hist_list, temporary_roots, temporary_paths = _load_histograms(
                root_file, matching_keys, root_file_path
            )

        if not hist_list:
            raise ValueError("No valid histograms could be loaded from the ROOT file.")

        # Keep the existing behavior where names are initialized from histogram
        # titles before histogram-dependent defaults are resolved.
        if plot_config.get("names", None) is None:
            names = [getattr(hist, "title", f"Hist {i}") for i, hist in enumerate(hist_list)]
            used = plot_config.get("legend", False) or (
                plot_config.get("stats", False) and len(hist_list) > 1
            )
            plot_config.set("names", names, used=used)

        # Resolve all histogram-dependent defaults exactly once before plotting.
        refresh_config(hist_list, plot_config)
        if plot_config.get("fits", False):
            _attach_embedded_fits(root_file_path, matching_keys, hist_list)

        # process_histograms configures rcParams for the plot. Keep that style
        # local to this API call instead of leaking it into the host application.
        with plt.rc_context():
            result = process_histograms(hist_list, plot_config)
            figure, _title, _extra_artists = result

        if figure is None:
            raise RuntimeError("Plotting failed for the selected ROOT histograms.")

        return figure
    except FileNotFoundError:
        raise
    except uproot.exceptions.KeyInFileError as exc:
        raise KeyError(f"A requested histogram key is missing from '{root_file_path}'.") from exc
    finally:
        for temporary_root in temporary_roots:
            temporary_root.close()
        for temporary_path in temporary_paths:
            with suppress(FileNotFoundError):
                temporary_path.unlink()


def plot_canvas_from_root(
    root_path: PathLike,
    canvas_key: str,
    *,
    config: Optional[PlotConfig] = None,
    pad_index: int = 0,
    histogram_index: int = 0,
) -> Figure:
    """Plot a histogram stored as a primitive inside a ROOT canvas."""
    extracted_path = extract_canvas_histogram(
        root_path,
        canvas_key,
        pad_index=pad_index,
        histogram_index=histogram_index,
    )
    try:
        return plot_from_root(
            extracted_path,
            config=config,
            histogram_keys=["selected_histogram"],
        )
    finally:
        with suppress(FileNotFoundError):
            extracted_path.unlink()
