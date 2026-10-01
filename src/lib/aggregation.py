"""Load and combine two-dimensional ROOT histograms.

The implementation uses uproot and NumPy only.  PyROOT is optional and is
loaded lazily by :meth:`Histogram3D.to_pyroot` when a native ROOT object is
needed by another application.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import uproot


PathLike = Union[str, Path]


@dataclass(frozen=True)
class Histogram2D:
    """Materialized contents and binning of one ROOT TH2 histogram."""

    values: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    variances: Optional[np.ndarray] = None
    title: str = "Histogram"
    source: Optional[Path] = None

    def to_numpy(self):
        """Return the uproot-compatible ``(values, x_edges, y_edges)`` tuple."""
        return self.values, self.x_edges, self.y_edges

    def errors(self) -> Optional[np.ndarray]:
        """Return bin errors when the source histogram stored sum-of-weights squared."""
        return None if self.variances is None else np.sqrt(np.maximum(self.variances, 0))


@dataclass(frozen=True)
class Histogram3D:
    """A 3D histogram formed by placing 2D histograms in Z slices."""

    values: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray
    variances: Optional[np.ndarray] = None
    title: str = "Combined histogram"

    @property
    def classname(self) -> str:
        return "TH3"

    def to_numpy(self):
        """Return the uproot-compatible ``(values, x_edges, y_edges, z_edges)`` tuple."""
        return self.values, self.x_edges, self.y_edges, self.z_edges

    def errors(self) -> Optional[np.ndarray]:
        """Return bin errors when source variances were available."""
        return None if self.variances is None else np.sqrt(np.maximum(self.variances, 0))

    def to_pyroot(self, name: str = "combined_histogram") -> Any:
        """Create a detached ROOT ``TH3D`` object.

        ROOT is intentionally imported only here, so normal library use does
        not require a PyROOT installation.  The returned object owns its bin
        contents independently of the input files.
        """
        try:
            import ROOT  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "PyROOT is required for Histogram3D.to_pyroot(); "
                "the uproot/NumPy representation is available without it."
            ) from exc

        histogram = ROOT.TH3D(
            name,
            self.title,
            len(self.x_edges) - 1,
            self.x_edges,
            len(self.y_edges) - 1,
            self.y_edges,
            len(self.z_edges) - 1,
            self.z_edges,
        )
        for x_index, y_index, z_index in np.ndindex(self.values.shape):
            root_x = x_index + 1
            root_y = y_index + 1
            root_z = z_index + 1
            histogram.SetBinContent(root_x, root_y, root_z, float(self.values[x_index, y_index, z_index]))
            if self.variances is not None:
                error = np.sqrt(max(float(self.variances[x_index, y_index, z_index]), 0.0))
                histogram.SetBinError(root_x, root_y, root_z, error)
        histogram.SetDirectory(0)
        return histogram


def load_histograms(
    root_files: Sequence[PathLike],
    histogram_key: str,
) -> List[Histogram2D]:
    """Load one TH2 from each ROOT file into detached array-backed objects.

    Parameters
    ----------
    root_files:
        ROOT files in the desired Z-slice order.
    histogram_key:
        The key of the TH2 in every file.
    """
    histograms: List[Histogram2D] = []
    for root_file_path in root_files:
        path = Path(root_file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ROOT file not found: {path}")
        with uproot.open(path) as root_file:
            try:
                source = root_file[histogram_key]
            except uproot.exceptions.KeyInFileError as exc:
                raise KeyError(f"Histogram key '{histogram_key}' was not found in '{path}'.") from exc

            if not hasattr(source, "to_numpy"):
                raise TypeError(f"Object '{histogram_key}' in '{path}' is not a histogram.")
            result = source.to_numpy(flow=False)
            if len(result) != 3:
                raise TypeError(f"Histogram '{histogram_key}' in '{path}' is not two-dimensional.")

            values, x_edges, y_edges = (np.array(part, copy=True) for part in result)
            source_variances = source.variances(flow=False) if hasattr(source, "variances") else None
            variances = None if source_variances is None else np.array(source_variances, copy=True)
            histograms.append(
                Histogram2D(
                    values=values,
                    x_edges=x_edges,
                    y_edges=y_edges,
                    variances=variances,
                    title=str(getattr(source, "title", histogram_key)),
                    source=path,
                )
            )
    return histograms


def combine_histograms_3d(
    histograms: Sequence[Histogram2D],
    *,
    z_edges: Optional[Sequence[float]] = None,
    title: str = "Combined histogram",
) -> Histogram3D:
    """Stack compatible TH2 histograms into consecutive slices along Z.

    This is a slice stack, not a sum: source histogram ``i`` becomes the
    contents of Z bin ``i``.  By default Z edges are ``0, 1, ..., N``.
    """
    if not histograms:
        raise ValueError("At least one 2D histogram is required.")

    first = histograms[0]
    for index, histogram in enumerate(histograms[1:], start=1):
        if not np.array_equal(histogram.x_edges, first.x_edges) or not np.array_equal(histogram.y_edges, first.y_edges):
            raise ValueError(f"Histogram {index} has incompatible X or Y bin edges.")
        if histogram.values.shape != first.values.shape:
            raise ValueError(f"Histogram {index} has incompatible bin counts.")

    resolved_z_edges = np.arange(len(histograms) + 1, dtype=float) if z_edges is None else np.asarray(z_edges, dtype=float)
    if resolved_z_edges.ndim != 1 or len(resolved_z_edges) != len(histograms) + 1:
        raise ValueError("z_edges must contain exactly one more edge than there are histograms.")
    if not np.all(np.diff(resolved_z_edges) > 0):
        raise ValueError("z_edges must be strictly increasing.")

    variances = None
    if all(histogram.variances is not None for histogram in histograms):
        variances = np.stack([histogram.variances for histogram in histograms], axis=2)

    return Histogram3D(
        values=np.stack([histogram.values for histogram in histograms], axis=2),
        x_edges=np.array(first.x_edges, copy=True),
        y_edges=np.array(first.y_edges, copy=True),
        z_edges=resolved_z_edges,
        variances=variances,
        title=title,
    )


__all__ = ["Histogram2D", "Histogram3D", "load_histograms", "combine_histograms_3d"]