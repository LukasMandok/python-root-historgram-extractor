"""Public API for ROOT histogram extraction and plotting."""

from .config import PlotConfig
from .core import plot_canvas_from_root, plot_from_root
from .aggregation import Histogram2D, Histogram3D, combine_histograms_3d, load_histograms

__all__ = [
	"PlotConfig",
	"plot_canvas_from_root",
	"plot_from_root",
	"Histogram2D",
	"Histogram3D",
	"load_histograms",
	"combine_histograms_3d",
]