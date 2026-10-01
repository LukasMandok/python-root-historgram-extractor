"""Public API for extracting and plotting ROOT histograms."""

from lib import (
    Histogram2D,
    Histogram3D,
    PlotConfig,
    combine_histograms_3d,
    load_histograms,
    plot_canvas_from_root,
    plot_from_root,
)

__all__ = [
    "PlotConfig",
    "plot_canvas_from_root",
    "plot_from_root",
    "Histogram2D",
    "Histogram3D",
    "load_histograms",
    "combine_histograms_3d",
]