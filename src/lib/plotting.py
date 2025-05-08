import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import scipy.stats # For chi2 probability in model stats box
from matplotlib.collections import PolyCollection
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker, HPacker
from typing import List, Dict, Any, Optional, Tuple

# Import from other library modules
from .constants import HistogramType
from .utils import get_config, get_color, set_equal_nice_ticks, getMember
from .models import model_functions, quick_plot_model

# --- Statistics Box Creation ---

def create_stats_box(ax: plt.Axes, stats: List[Tuple[str, str]], title: Optional[str] = None,
                     color: str = "black", size: int = 10, offset: float = 0.0) -> None:
    """Creates an anchored stats box with aligned columns."""
    if not stats: return # Don't create empty boxes

    prop_boxes = [TextArea(prop, textprops={"size": size, "family": "monospace"}) for prop, _ in stats]
    val_boxes  = [TextArea(val,  textprops={"size": size, "family": "monospace"}) for _, val in stats]

    col1 = VPacker(children=prop_boxes, align="left", pad=0, sep=2)
    col2 = VPacker(children=val_boxes,  align="right", pad=0, sep=5) # Increased sep for values

    hbox = HPacker(children=[col1, col2], align="center", pad=0, sep=10) # Increased sep between columns

    if title is not None:
        # Use the actual color value, not the name
        actual_color = get_color(color, get_config("palette", "color1")) # Provide default palette
        title_area = TextArea(title, textprops={"size": size, "color": actual_color, "weight": "bold"})
        vbox = VPacker(children=[title_area, hbox], align="left", pad=0, sep=5)
    else:
        vbox = hbox

    anchored_box = AnchoredOffsetbox(loc="upper right", child=vbox,
                                     bbox_to_anchor=(0.99, 0.99 - offset), # Anchor to top-right
                                     bbox_transform=ax.transAxes,
                                     pad=0.5, borderpad=0.5, # Added borderpad
                                     frameon=True)

    anchored_box.patch.set_boxstyle("round,pad=0.3,rounding_size=0.2") # Adjusted pad
    anchored_box.patch.set_edgecolor('lightgrey') # Make frame less prominent
    ax.add_artist(anchored_box)


def create_stats(ax: plt.Axes, hist: HistogramType, i: int = 0,
                 name: Optional[str] = None, limits: Optional[List[float]] = None,
                 color: str = "black") -> None:
    """Computes and creates a statistics box for TH1 or TH2 histograms."""
    stats_list = []
    try:
        result = hist.to_numpy()
        textsize = plt.rcParams['font.size'] * 0.9 # Slightly smaller for stats

        if isinstance(hist, (uproot.behaviors.TH1.TH1, uproot.behaviors.TH1.TH1F, uproot.behaviors.TH1.TH1D)):
            values, edges = result
            midpoints = (edges[:-1] + edges[1:]) / 2
            mask = np.ones_like(midpoints, dtype=bool)
            if limits is not None and len(limits) >= 1 and limits[0] is not None:
                 mask = (midpoints >= limits[0][0]) & (midpoints <= limits[0][1])

            masked_values = values[mask]
            masked_midpoints = midpoints[mask]

            total = masked_values.sum()
            mean = np.average(masked_midpoints, weights=masked_values) if total > 0 else 0.0
            variance = np.average((masked_midpoints - mean)**2, weights=masked_values) if total > 0 else 0.0
            std = np.sqrt(variance) if total > 0 else 0.0

            stats_list = [("Entries", f"{int(total)}"),
                          ("Mean", f"{mean:.3g}"), # Use general format
                          ("Std Dev", f"{std:.3g}")]

        elif isinstance(hist, (uproot.behaviors.TH2.TH2, uproot.behaviors.TH2.TH2F, uproot.behaviors.TH2.TH2D)):
            data, x_edges, y_edges = result
            mid_x = (x_edges[:-1] + x_edges[1:]) / 2.0
            mid_y = (y_edges[:-1] + y_edges[1:]) / 2.0
            X, Y = np.meshgrid(mid_x, mid_y, indexing='ij')

            mask = np.ones_like(X, dtype=bool)
            if limits is not None:
                if len(limits) >= 1 and limits[0] is not None:
                    mask &= (X >= limits[0][0]) & (X <= limits[0][1])
                if len(limits) >= 2 and limits[1] is not None:
                    mask &= (Y >= limits[1][0]) & (Y <= limits[1][1]) # Assuming limits[1] is for Y

            masked_data = data[mask]
            masked_X = X[mask]
            masked_Y = Y[mask]

            total = masked_data.sum()
            meanX = np.average(masked_X, weights=masked_data) if total > 0 else 0.0
            stdX = np.sqrt(np.average((masked_X - meanX)**2, weights=masked_data)) if total > 0 else 0.0
            meanY = np.average(masked_Y, weights=masked_data) if total > 0 else 0.0
            stdY = np.sqrt(np.average((masked_Y - meanY)**2, weights=masked_data)) if total > 0 else 0.0

            stats_list = [("Entries", f"{int(total)}"),
                          ("Mean X", f"{meanX:.3g}"),
                          ("Std Dev X", f"{stdX:.3g}"),
                          ("Mean Y", f"{meanY:.3g}"),
                          ("Std Dev Y", f"{stdY:.3g}")]
        else:
            # Add stats for TGraph, TF1, TH3 if needed
            pass

        if stats_list:
            # Calculate vertical offset based on number of previous boxes and text size
            vertical_offset = i * (len(stats_list) + (1 if name else 0) + 1.5) * (textsize / 10.0) * 0.035 # Empirical scaling
            create_stats_box(ax, stats_list, title=name, color=color, size=int(textsize), offset=vertical_offset)

    except Exception as e:
        print(f"Warning: Could not compute statistics for {getattr(hist, 'title', 'histogram')}: {e}", file=sys.stderr)


def create_model_stats_box(ax: plt.Axes, model_name: str, model_info: Dict[str, Any], offset: float = 0.0, color: str = 'black') -> None:
    """Creates a statistics box for model fit parameters."""
    textsize = plt.rcParams['font.size'] * 0.85 # Slightly smaller

    params = model_info.get("params", {})
    errors = model_info.get("errors", {})
    chi2 = model_info.get("chi2")
    ndf = model_info.get("ndf")

    stats = []
    if chi2 is not None and ndf is not None and ndf > 0:
        stats.append(("χ²/ndf", f"{chi2:.1f}/{ndf}"))
        prob = scipy.stats.chi2.sf(chi2, ndf)
        stats.append(("Prob", f"{prob:.3f}"))

    for name, value in params.items():
        error = errors.get(name)
        if error is not None and error > 0:
            # Determine precision based on error
            magnitude = np.floor(np.log10(error))
            digits = int(max(0, -magnitude + 1)) # Show 2 significant digits of error
            formatted_value = f"{value:.{digits}f} ± {error:.{digits}f}"
        elif error is not None and error == 0:
             formatted_value = f"{value:.4g} (fixed)" # Indicate fixed params
        else:
            formatted_value = f"{value:.4g}" # General format if no error
        stats.append((f"{name}", formatted_value))

    if not stats: return

    # Use the actual color value
    actual_color = get_color(color, get_config("palette", "color1"))

    prop_boxes = [TextArea(prop, textprops={"size": textsize, "family": "monospace"}) for prop, _ in stats]
    val_boxes = [TextArea(val, textprops={"size": textsize, "family": "monospace"}) for _, val in stats]

    col1 = VPacker(children=prop_boxes, align="left", pad=0, sep=2)
    col2 = VPacker(children=val_boxes, align="right", pad=0, sep=5)
    hbox = HPacker(children=[col1, col2], align="center", pad=0, sep=10)

    title_area = TextArea(f"{model_name} Fit", textprops={"size": textsize, "weight": "bold", "color": actual_color})
    vbox = VPacker(children=[title_area, hbox], align="left", pad=0, sep=5)

    # Position near bottom right, adjust offset
    anchored_box = AnchoredOffsetbox(loc="lower right", child=vbox,
                                     bbox_to_anchor=(0.99, 0.01 + offset),
                                     bbox_transform=ax.transAxes,
                                     pad=0.5, borderpad=0.5,
                                     frameon=True)

    anchored_box.patch.set_boxstyle("round,pad=0.3,rounding_size=0.2")
    anchored_box.patch.set_edgecolor('lightgrey')
    ax.add_artist(anchored_box)


# --- Histogram Processing Functions ---

def add_TH1(ax: plt.Axes, index: int, hist: HistogramType, color: str, alpha: float,
           limits: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
           is_ratio: bool = False, count: int = 1, thickness: float = 2.0, cutoff: bool = False) -> None:
    """Adds a single TH1 histogram to the axes."""
    try:
        values, edges = hist.to_numpy()[:2]
        midpoints = (edges[:-1] + edges[1:]) / 2
        errors = hist.errors() if hasattr(hist, 'errors') else None
        has_errors = errors is not None and len(errors) == len(values)

    except Exception as e:
        print(f"Error converting histogram {getattr(hist, 'title', 'TH1')}: {e}", file=sys.stderr)
        return # Skip this histogram

    bin_width = edges[1] - edges[0] if len(edges) > 1 else 1.0

    # Apply cutoff if enabled and limits are provided
    mask = np.ones_like(midpoints, dtype=bool)
    if not is_ratio and cutoff and limits and limits[0] is not None:
        xmin, xmax = limits[0]
        if xmin is not None: mask &= (midpoints >= xmin)
        if xmax is not None: mask &= (midpoints <= xmax)

        values = values[mask]
        midpoints = midpoints[mask]
        if has_errors: errors = errors[mask]
        # Adjust edges to match the masked bins (optional, maybe just limit axis later)
        # edge_indices = np.where(mask)[0]
        # if len(edge_indices) > 0:
        #     min_idx, max_idx = edge_indices[0], edge_indices[-1]
        #     edges = edges[min_idx : max_idx + 2] # Include one edge beyond last bin

    if is_ratio:
        # Bar plot for categorical data
        bar_width = 0.8 * bin_width / count # Adjust width/spacing
        offset = (index - (count - 1) / 2) * bar_width
        ax.bar(midpoints + offset, values, color=color, alpha=alpha, label=hist.title,
               width=bar_width, align="center")

        labels = getMember(getMember(hist, "fXaxis"), "fLabels")
        if labels:
            str_labels = [str(lbl) for lbl in labels]
            ax.set_xticks(midpoints)
            ax.set_xticklabels(str_labels, rotation=45, ha='right') # Rotate labels if needed
            # Adjust x-limits slightly for padding
            padding = bin_width * 0.1
            ax.set_xlim(edges[0] - padding, edges[-1] + padding)
        else:
            print("Warning: No labels found for categorical histogram", file=sys.stderr)
    else:
        # Step plot or error bar plot for numerical data
        show_errors = get_config("errors", False)
        if show_errors and has_errors:
            ax.errorbar(midpoints, values, yerr=errors, fmt='.', color=color, # Use '.' for marker
                       markersize=thickness * 2, ecolor=color, elinewidth=thickness/2, capsize=thickness, # Scale sizes with thickness
                       alpha=alpha, label=hist.title, linewidth=0) # Linewidth 0 for errorbar line itself
        else:
            # Use drawstyle='steps-mid' for standard ROOT histogram appearance
            ax.step(edges[:-1], values, where='post', color=color, alpha=alpha,
                    label=hist.title, linewidth=thickness)
            # Alternative: ax.hist(midpoints, bins=edges, weights=values, color=color, alpha=alpha,
            #         histtype="step", label=hist.title, linewidth=thickness)


def process_TH1(ax: plt.Axes, hist_list: List[HistogramType]) -> None:
    """Processes and plots a list of TH1 histograms."""
    limits_config = get_config("limits", None) # Can be [[xmin, xmax], [ymin, ymax]] or just [[xmin, xmax]]
    limits = limits_config[0] if limits_config and len(limits_config) > 0 else None
    y_limits = limits_config[1] if limits_config and len(limits_config) > 1 else None

    colors = get_config("colors", [])
    color_palette = get_config("palette", "color1") # Default palette
    alphas = get_config("alphas", [])
    log_x = get_config("x-log", False)
    log_y = get_config("y-log", False)
    thickness = get_config("thickness", 1.5) # Default thickness
    cutoff = get_config("cutoff", False)
    show_stats = get_config("stats", False)
    show_legend = get_config("legend", False)
    names = get_config("names", [getattr(h, 'title', f'Hist {i}') for i, h in enumerate(hist_list)])
    models = get_config("models", [])
    model_params_config = get_config("model-params", {})
    show_model_stats = get_config("model-stats", True)

    # Ensure enough colors and alphas, cycle if necessary
    num_hists = len(hist_list)
    if len(colors) < num_hists:
        colors = [colors[i % len(colors)] for i in range(num_hists)] if colors else ['blue'] * num_hists
    if len(alphas) < num_hists:
        alphas = [alphas[i % len(alphas)] for i in range(num_hists)] if alphas else [1.0] * num_hists
    if len(names) < num_hists:
        names = [names[i % len(names)] for i in range(num_hists)] if names else [getattr(h, 'title', f'Hist {i}') for i, h in enumerate(hist_list)]


    is_ratio = False
    if hist_list:
        # Check first histogram for labels (heuristic for categorical)
        first_hist_xaxis = getMember(hist_list[0], "fXaxis")
        if first_hist_xaxis and getMember(first_hist_xaxis, "fLabels"):
             is_ratio = True
             print("Detected categorical histogram (ratio plot style).")


    # Plot histograms
    plotted_items_count = 0
    for i, hist in enumerate(hist_list):
        actual_color = get_color(colors[i], color_palette)
        add_TH1(ax, i, hist, actual_color, alphas[i], [limits] if limits else None, is_ratio, num_hists, thickness, cutoff)
        plotted_items_count += 1

    # Plot models
    model_lines = []
    model_stat_boxes_count = 0
    if models and not is_ratio:
        num_models = len(models)
        model_colors = get_config("model-colors", colors[num_hists:]) # Use extra colors if provided
        model_alphas = get_config("model-alphas", alphas[num_hists:])

        if len(model_colors) < num_models:
             # Cycle through histogram colors or default if none provided
             base_model_colors = colors if colors else ['red']
             model_colors = [base_model_colors[i % len(base_model_colors)] for i in range(num_models)]
        if len(model_alphas) < num_models:
             # Use histogram alphas or default
             base_model_alphas = alphas if alphas else [0.8]
             model_alphas = [base_model_alphas[i % len(base_model_alphas)] for i in range(num_models)]

        for i, model_name in enumerate(models):
            if not model_name or i >= num_hists: continue # Skip empty names or if no corresponding hist

            hist = hist_list[i] # Associate model with i-th histogram
            if model_name not in model_functions:
                print(f"Warning: Model '{model_name}' not found, skipping.", file=sys.stderr)
                continue

            actual_model_color = get_color(model_colors[i], color_palette)
            model_alpha = model_alphas[i]

            # Prepare parameters for quick_plot_model
            params = model_params_config.get(model_name, {})
            params.setdefault("color", actual_model_color)
            params.setdefault("alpha", model_alpha)
            params.setdefault("label", f"{names[i]}_{model_name}" if num_hists > 1 else f"{model_name} model") # Label based on hist name if multiple

            # Determine x_range for the model plot
            if "x_range" not in params:
                if limits:
                    x_min, x_max = limits
                    if x_min is not None and x_max is not None:
                         params["x_range"] = (x_min, x_max, 1000)
                if "x_range" not in params: # Fallback to histogram edges
                    try:
                        _, edges = hist.to_numpy()[:2]
                        params["x_range"] = (edges[0], edges[-1], 1000)
                    except Exception:
                         params["x_range"] = (0, 10, 1000) # Default if edges fail
                         print(f"Warning: Could not determine x_range for model '{model_name}', using default.", file=sys.stderr)

            # Plot the model using quick_plot_model
            line = quick_plot_model(ax, model_name, **params)
            if line:
                model_lines.append(line)
                plotted_items_count += 1

                # Display model statistics if enabled and available
                if show_model_stats and "stats" in params:
                    model_stats_data = params["stats"]
                    # Offset model stat boxes from the bottom
                    model_offset = model_stat_boxes_count * 0.25 # Adjust spacing as needed
                    create_model_stats_box(ax, model_name, model_stats_data, offset=model_offset, color=model_colors[i])
                    model_stat_boxes_count += 1


    # Axis labels and limits
    if not is_ratio:
        ax.set_xlabel(get_config("x-label", "X-axis"))
        if limits and limits[0] is not None and limits[1] is not None and not cutoff:
             ax.set_xlim(limits)
        ax.set_xscale("log" if log_x else "linear")
    else:
        # For ratio plots, x-label might be less meaningful or needs specific handling
        ax.set_xlabel(get_config("x-label", "")) # Often empty for categorical

    ax.set_ylabel(get_config("y-label", "Entries"))
    if y_limits and y_limits[0] is not None and y_limits[1] is not None:
        ax.set_ylim(y_limits)
    ax.set_yscale("log" if log_y else "linear")

    # Add histogram statistics boxes
    if show_stats:
        hist_stat_boxes_count = 0
        for i, hist in enumerate(hist_list):
            # Pass limits[0] if it exists for TH1 stats calculation range
            stat_limits = [limits] if limits else None
            create_stats(ax, hist, i, name=names[i] if num_hists > 1 else None, limits=stat_limits, color=colors[i])
            hist_stat_boxes_count += 1


    # Legend
    if show_legend and plotted_items_count > 0:
        # Decide legend location based on stats boxes
        # If stats are shown, put legend maybe center left or lower center?
        loc = "best"
        if show_stats or model_stat_boxes_count > 0:
             loc = "center left" # Try to avoid overlapping stats boxes in corners

        # Get handles and labels from axes, filter out duplicates if necessary
        handles, labels = ax.get_legend_handles_labels()
        # Create a unique mapping from label to handle
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc=loc)


def process_TH2(ax: plt.Axes, hist_list: List[HistogramType]) -> None:
    """Processes and plots a list of TH2 histograms (currently only supports one)."""
    if len(hist_list) > 1:
        print("ERROR: Stacking of TH2 histograms not yet implemented.", file=sys.stderr)
        sys.exit(1)
        
    if not hist_list:
        print("Warning: No TH2 histogram provided to process_TH2.", file=sys.stderr)
        return

    hist: HistogramType = hist_list[0]
    try:
        data, x_edges, y_edges = hist.to_numpy()
    except Exception as e:
        print(f"Error converting histogram {getattr(hist, 'title', 'TH2')}: {e}", file=sys.stderr)
        return

    # Configuration
    color_map_name = get_config("colors", ["viridis_red"])[0] # Expect list, take first
    cmap = plt.get_cmap(color_map_name)
    cmap.set_bad(alpha=0) # Make empty bins transparent
    cmap.set_under(alpha=0) # Make underflow transparent (if vmin is set > 0)

    log_z = get_config("z-log", False)
    show_stats = get_config("stats", False)
    show_colorbar = get_config("legend", True) # Use legend flag for colorbar
    rasterized = get_config("raster", True)
    limits_config = get_config("limits", None) # [[xmin, xmax], [ymin, ymax]]

    limit_x = limits_config[0] if limits_config and len(limits_config) > 0 else (x_edges[0], x_edges[-1])
    limit_y = limits_config[1] if limits_config and len(limits_config) > 1 else (y_edges[0], y_edges[-1])

    ##### Alternative plotting using pcolormesh (turned out not to be suitable)

    # Determine normalization for color mapping
    # vmin = None
    # vmax = data.max()
    # if log_z:
    #     # Find minimum non-zero value for log scale
    #     min_positive = data[data > 0].min() if np.any(data > 0) else 1.0
    #     vmin = max(min_positive, 1e-5) # Avoid log(0) or very small numbers
    #     norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    # else:
    #     vmin = 0 # Start colorbar at 0 for linear scale
    #     norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # # Use pcolormesh for efficient plotting
    # # Note: data needs to be indexed [y, x] for pcolormesh if using default shading='flat'
    # # However, uproot TH2.to_numpy() returns [x, y], so we transpose data.
    # mesh = ax.pcolormesh(x_edges, y_edges, data.T, cmap=cmap, norm=norm,
    #                      rasterized=rasterized, shading='flat') # Use flat shading for direct bin mapping


    ##### Plot using PolyCollection
    
    maxcount = data.max()
    
    # Create vertices for each quad
    verts = []
    colors = []

    for i in range(len(x_edges)-1):
        for j in range(len(y_edges)-1):
            if data[i,j] > 0:  # Only add non-zero bins
                quad = [
                    (x_edges[i], y_edges[j]),
                    (x_edges[i+1], y_edges[j]),
                    (x_edges[i+1], y_edges[j+1]),
                    (x_edges[i], y_edges[j+1])
                ]
                verts.append(quad)
                
                if log_z:
                    # Use log10 scaling but avoid taking log of 0
                    norm_value = np.log10(max(data[i,j], 1e-10)) / np.log10(max(maxcount, 1e-10))
                else:
                    norm_value = data[i,j] / maxcount
                
                colors.append(cmap(norm_value))
    
    # Create collection
    collection = PolyCollection(verts,
                              facecolors=colors,
                              rasterized=rasterized)
    
    ax.add_collection(collection)


    ##### Format Histogram
    
    if limits_config is None:
        limit_x = (x_edges[0], x_edges[-1])
        limit_y = (y_edges[0], y_edges[-1])
    else: 
        limit_x = limits_config[0]
        limit_y = limits_config[1]

    # Set limits and labels
    ax.set_xlim(limit_x)
    ax.set_ylim(limit_y)
    
    ax.set_xlabel(rf"{get_config('x-label')}")
    ax.set_ylabel(rf"{get_config('y-label')}")

    # Add statistics box
    if show_stats:
        create_stats(ax, hist) # Pass full limits config #, limits=limits_config

    # Add colorbar
    if show_colorbar:
        if log_z:
            min_val = 1.0  # Minimum non-zero value
            norm = mcolors.LogNorm(vmin=min_val, vmax=maxcount)
        else:
            norm = plt.Normalize(vmin=0.0, vmax=maxcount)
            
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Count")
        
        # Update ticks for log scale
        if log_z:
            min_exp = math.floor(math.log10(min_val))
            max_exp = math.ceil(math.log10(maxcount))
            
            # Generate tick positions at powers of 10
            tick_locations = [10**i for i in range(min_exp, max_exp+1)]
            
            # Filter out tick locations outside the data range
            tick_locations = [t for t in tick_locations if min_val <= t <= maxcount]
            
            # Set the tick locations and format them
            cbar.set_ticks(tick_locations)
            cbar.set_ticklabels([f"{int(t)}" if t >= 1 else f"{t:.1g}" for t in tick_locations])
    

def add_flat_TH3(ax: plt.Axes, hist: HistogramType, cmap, alpha: float, raster: bool, norm) -> None:
    """Adds a TH3 histogram as flat 2D slices to a 3D axis."""
    try:
        data, x_edges, y_edges, z_edges = hist.to_numpy()
    except Exception as e:
        print(f"Error converting histogram {getattr(hist, 'title', 'TH3')}: {e}", file=sys.stderr)
        return

    # Create meshgrid for X, Y coordinates (centers)
    # x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    # y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    # X, Y = np.meshgrid(x_centers, y_centers, indexing='ij') # Match data indexing

    maxcount = norm.vmax # Get max from norm

    # Iterate through Z slices
    for iz in range(len(z_edges) - 1):
        data_slice = data[:, :, iz] # Data for this Z slice (X, Y)
        if data_slice.max() <= (norm.vmin if norm.vmin is not None else 0): # Skip empty slices based on norm
            continue

        z_position = (z_edges[iz] + z_edges[iz+1]) / 2.0 # Z position for this slice

        # --- Use PolyCollection for better control ---
        verts = []
        colors = []
        slice_data_T = data_slice.T # Transpose for iteration matching PolyCollection expectation (y, x)

        for j in range(len(y_edges)-1): # Iterate y
            for i in range(len(x_edges)-1): # Iterate x
                count = slice_data_T[j, i]
                if count > (norm.vmin if norm.vmin is not None else 0): # Check against norm vmin
                    quad = [
                        (x_edges[i], y_edges[j]),
                        (x_edges[i+1], y_edges[j]),
                        (x_edges[i+1], y_edges[j+1]),
                        (x_edges[i], y_edges[j+1])
                    ]
                    verts.append(quad)
                    # Apply colormap and norm
                    colors.append(cmap(norm(count)))

        if not verts: continue # Skip if slice becomes empty after filtering

        poly = PolyCollection(verts, facecolors=colors, alpha=alpha, rasterized=raster)
        ax.add_collection3d(poly, zs=z_position, zdir='z')


def add_3d_TH3(ax: plt.Axes, hist: HistogramType, cmap, alpha: float, norm) -> None:
    """Adds a TH3 histogram as 3D bars to a 3D axis."""
    try:
        data, x_edges, y_edges, z_edges = hist.to_numpy()
    except Exception as e:
        print(f"Error converting histogram {getattr(hist, 'title', 'TH3')}: {e}", file=sys.stderr)
        return

    dx = x_edges[1] - x_edges[0] if len(x_edges) > 1 else 1
    dy = y_edges[1] - y_edges[0] if len(y_edges) > 1 else 1
    dz = z_edges[1] - z_edges[0] if len(z_edges) > 1 else 1

    maxcount = norm.vmax

    # Iterate through bins
    for ix in range(len(x_edges)-1):
        for iy in range(len(y_edges)-1):
            for iz in range(len(z_edges)-1):
                count = data[ix, iy, iz]
                if count > (norm.vmin if norm.vmin is not None else 0): # Check against norm vmin
                    # Calculate color based on count and norm
                    color = cmap(norm(count))
                    # Apply alpha to the color tuple
                    final_color = (*color[:3], alpha)

                    # Add bar3d for the bin
                    ax.bar3d(x_edges[ix], y_edges[iy], z_edges[iz], dx, dy, dz,
                             color=final_color, shade=alpha > 0.8) # Shade if mostly opaque


def process_TH3(ax: plt.Axes, hist_list: List[HistogramType]) -> plt.Artist:
    """Processes and plots a list of TH3 histograms (currently only supports one)."""
    if len(hist_list) > 1:
        print("ERROR: Stacking of TH3 histograms not yet implemented.", file=sys.stderr)
        sys.exit(1)
    if not hist_list:
        print("Warning: No TH3 histogram provided to process_TH3.", file=sys.stderr)
        return None # Return None if no histogram

    hist = hist_list[0]

    # Configuration
    angles = get_config("angles", [30, -60]) # Default view angles (elev, azim)
    color_map_name = get_config("colors", ["viridis_red"])[0]
    cmap = plt.get_cmap(color_map_name)
    cmap.set_bad(alpha=0)
    alpha = get_config("alphas", [0.7])[0] # Default alpha for 3D
    is_flat = get_config("flat", False)
    raster = get_config("raster", True)
    show_colorbar = get_config("legend", True)
    show_grid = get_config("grid", True)
    limits_config = get_config("limits", None) # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    log_z = get_config("z-log", False) # Use z-log for color mapping

    ax.view_init(angles[0], angles[1])

    # Determine normalization based on data and log_z for color mapping
    try:
        data, x_edges, y_edges, z_edges = hist.to_numpy()
        vmax = data.max()
        vmin = None
        if log_z:
            min_positive = data[data > 0].min() if np.any(data > 0) else 1.0
            vmin = max(min_positive, 1e-5)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            vmin = 0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    except Exception as e:
        print(f"Error getting data for TH3 norm: {e}", file=sys.stderr)
        norm = mcolors.Normalize(vmin=0, vmax=1) # Fallback norm
        x_edges, y_edges, z_edges = [0, 1], [0, 1], [0, 1] # Fallback edges

    # Plotting
    if is_flat:
        add_flat_TH3(ax, hist, cmap, alpha, raster, norm)
    else:
        add_3d_TH3(ax, hist, cmap, alpha, norm)

    # Set limits
    if limits_config:
        if len(limits_config) > 0 and limits_config[0]: ax.set_xlim(limits_config[0])
        if len(limits_config) > 1 and limits_config[1]: ax.set_ylim(limits_config[1])
        if len(limits_config) > 2 and limits_config[2]: ax.set_zlim(limits_config[2])
    else: # Auto-limits based on edges
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_zlim(z_edges[0], z_edges[-1])

    # Labels and grid
    ax.set_xlabel(get_config("x-label", "X"))
    ax.set_ylabel(get_config("y-label", "Y"))
    zlabel_text = get_config("z-label", "Z")
    zlabel = ax.set_zlabel(zlabel_text) # Store the Z-label artist

    if show_grid:
        ax.grid(True, linestyle='--', color='grey', alpha=0.3)

    # Improve tick spacing and appearance
    set_equal_nice_ticks(ax, max_ticks=6) # Use helper for nicer ticks
    ax.tick_params(axis='both', which='major', pad=-2) # Adjust padding

    # Add colorbar
    if show_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) # Important for ScalarMappable
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=10, pad=0.1, label="Count") # Adjust size/padding
        # Optional: Customize log ticks
        # if log_z: ...

    # Adjust layout slightly (might need manual tweaking)
    # plt.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1) # May clip labels

    return zlabel # Return the Z-label artist for potential inclusion in bbox_extra_artists


def process_TGraph(ax: plt.Axes, hist_list: List[HistogramType]) -> None:
    """Processes and plots a list of TGraph or TF1 objects."""
    limits_config = get_config("limits", None) # [[xmin, xmax], [ymin, ymax]]
    x_limits = limits_config[0] if limits_config and len(limits_config) > 0 else None
    y_limits = limits_config[1] if limits_config and len(limits_config) > 1 else None

    names = get_config("names", [getattr(h, 'title', f'Graph {i}') for i, h in enumerate(hist_list)])
    colors = get_config("colors", [])
    color_palette = get_config("palette", "color1")
    alphas = get_config("alphas", [])
    log_x = get_config("x-log", False)
    log_y = get_config("y-log", False)
    show_legend = get_config("legend", False)
    # Add thickness? Errors? Markers?
    marker_style = get_config("marker", None) # e.g., 'o', '.', '+'
    line_style = get_config("linestyle", '-') # e.g., '-', '--', ':'
    line_width = get_config("thickness", 1.5)

    num_graphs = len(hist_list)
    if len(colors) < num_graphs:
        colors = [colors[i % len(colors)] for i in range(num_graphs)] if colors else ['blue'] * num_graphs
    if len(alphas) < num_graphs:
        alphas = [alphas[i % len(alphas)] for i in range(num_graphs)] if alphas else [1.0] * num_graphs
    if len(names) < num_graphs:
        names = [names[i % len(names)] for i in range(num_graphs)] if names else [getattr(h, 'title', f'Graph {i}') for i, h in enumerate(hist_list)]

    plotted_items_count = 0
    for i, hist in enumerate(hist_list):
        x, y, xerr, yerr = None, None, None, None
        actual_color = get_color(colors[i], color_palette)
        label = names[i]

        try:
            if isinstance(hist, (uproot.behaviors.TGraph.TGraph, uproot.behaviors.TGraph.TGraphErrors)):
                x = hist.values()[0]
                y = hist.values()[1]
                if isinstance(hist, uproot.behaviors.TGraph.TGraphErrors):
                     xerr = hist.errors("x")
                     yerr = hist.errors("y")
            elif isinstance(hist, uproot.behaviors.TF1.TF1):
                # Evaluate TF1 over its range
                xmin, xmax = getMember(hist, "fXmin"), getMember(hist, "fXmax")
                npx = getMember(hist, "fNpx", 100) # Default 100 points
                if xmin is not None and xmax is not None:
                    x = np.linspace(xmin, xmax, npx)
                    # TF1 evaluation might require a specific library or manual interpretation
                    # For now, we skip plotting TF1 directly, assuming it might be handled by models
                    print(f"Warning: Direct plotting of TF1 '{label}' not fully implemented. Use --models if it's a known function.", file=sys.stderr)
                    continue # Skip to next item
                else:
                    print(f"Warning: Could not get range for TF1 '{label}'.", file=sys.stderr)
                    continue
            else:
                 print(f"Warning: Unsupported type for TGraph processing: {hist.classname}", file=sys.stderr)
                 continue

            if x is not None and y is not None:
                 # Plot with or without errors
                 if yerr is not None and get_config("errors", False): # Check config for errors
                     ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=marker_style or '.', # Default marker '.' if errors shown
                                 markersize=(line_width*2 if marker_style else 0), # Only show marker if specified
                                 linestyle='none', # No line connecting error points by default
                                 color=actual_color, ecolor=actual_color, elinewidth=line_width/2, capsize=line_width,
                                 alpha=alphas[i], label=label)
                     # Optionally add a line connecting the points if linestyle is not 'none'
                     if line_style and line_style.lower() != 'none':
                          ax.plot(x, y, marker=None, linestyle=line_style, linewidth=line_width, color=actual_color, alpha=alphas[i], label="_nolegend_") # Hide duplicate label
                 else:
                     # Plot as line and/or markers
                     ax.plot(x, y, marker=marker_style, linestyle=line_style, linewidth=line_width,
                             markersize=(line_width*2 if marker_style else None), # Scale marker size
                             color=actual_color, alpha=alphas[i], label=label)
                 plotted_items_count += 1

        except Exception as e:
            print(f"Error processing TGraph/TF1 {label}: {e}", file=sys.stderr)

    # Set limits and scales
    if x_limits and x_limits[0] is not None and x_limits[1] is not None: ax.set_xlim(x_limits)
    if y_limits and y_limits[0] is not None and y_limits[1] is not None: ax.set_ylim(y_limits)
    ax.set_xlabel(get_config("x-label", "X"))
    ax.set_ylabel(get_config("y-label", "Y"))
    ax.set_xscale("log" if log_x else "linear")
    ax.set_yscale("log" if log_y else "linear")

    if get_config("grid", False): # Add grid option
        ax.grid(True, linestyle='--', alpha=0.6)

    if show_legend and plotted_items_count > 0:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best")


# --- Main Plotting Orchestration ---

def process_histograms(hist_list: List[HistogramType]) -> Tuple[Optional[plt.Figure], Optional[str], List[plt.Artist]]:
    """Determines histogram type and calls the appropriate plotting function."""
    if not hist_list:
        print("ERROR: No histograms provided to process.", file=sys.stderr)
        return None, None, []

    first_hist = hist_list[0]
    title = get_config("title", getattr(first_hist, 'title', "Histogram")) # Use getattr for safety
    extra_artists = [] # For artists like Z-label that need to be included in bbox_inches='tight'

    # Apply general plot styling from config
    textsize = get_config("textsize", 12.0) # Default text size
    plt.rcParams.update({
        'font.size': textsize,
        'axes.titlesize': textsize * 1.1,
        'axes.labelsize': textsize,
        'xtick.labelsize': textsize * 0.9,
        'ytick.labelsize': textsize * 0.9,
        'legend.fontsize': textsize * 0.9,
        'figure.titlesize': textsize * 1.2
    })

    # Create figure
    fig = plt.figure(figsize=get_config("figsize", (8, 6))) # Configurable figure size

    # Determine plot type and call processor
    ax = None
    if isinstance(first_hist, (uproot.behaviors.TH1.TH1, uproot.behaviors.TH1.TH1F, uproot.behaviors.TH1.TH1D)):
        ax = fig.add_subplot(111)
        process_TH1(ax, hist_list)
    elif isinstance(first_hist, (uproot.behaviors.TH2.TH2, uproot.behaviors.TH2.TH2F, uproot.behaviors.TH2.TH2D)):
        ax = fig.add_subplot(111)
        process_TH2(ax, hist_list)
    elif isinstance(first_hist, (uproot.behaviors.TH3.TH3, uproot.behaviors.TH3.TH3F, uproot.behaviors.TH3.TH3D)):
        ax = fig.add_subplot(111, projection='3d')
        zlabel_artist = process_TH3(ax, hist_list)
        if zlabel_artist: extra_artists.append(zlabel_artist)
    elif isinstance(first_hist, (uproot.behaviors.TGraph.TGraph, uproot.behaviors.TGraph.TGraphErrors, uproot.behaviors.TF1.TF1)):
        ax = fig.add_subplot(111)
        process_TGraph(ax, hist_list)
    else:
        print(f"ERROR: Unsupported histogram type: {first_hist.classname}", file=sys.stderr)
        plt.close(fig) # Close the empty figure
        return None, title, []

    if ax:
        ax.set_title(title) # Set title on the axes
        # Add legend artists to extra_artists if legend exists
        legend = ax.get_legend()
        if legend:
            extra_artists.append(legend)
            # Also add stat boxes if they exist
            for artist in ax.artists:
                if isinstance(artist, AnchoredOffsetbox):
                    extra_artists.append(artist)


    # Apply tight layout after all plotting is done
    try:
        fig.tight_layout() # rect=[0, 0.03, 1, 0.95]
    except ValueError as e:
         print(f"Warning: tight_layout failed: {e}", file=sys.stderr)
         # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1) # Manual adjustment fallback

    return fig, title, extra_artists