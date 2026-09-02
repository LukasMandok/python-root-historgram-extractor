"""Public API for ROOT histogram extraction and plotting."""

from .config import PlotConfig
from .core import plot_canvas_from_root, plot_from_root

__all__ = ["PlotConfig", "plot_canvas_from_root", "plot_from_root"]