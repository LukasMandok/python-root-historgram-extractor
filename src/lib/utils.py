import os
import json
import math
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import ANSI
from datetime import datetime
import subprocess
import re
import sys
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

# Import constants and models from sibling modules
from .constants import (CACHE_FILE, COMMON_CACHE_BASE_DIR, allowed_classes, color_list, color_palettes,
                       color_maps, reset, red, blue, green, HistogramType)
from .models import model_functions

# --- Global Configuration Store ---
# Default anapath to current working directory. Can be overridden by set_global_anapath.
anapath: str = os.getcwd() 

config_parameters: Dict[str, Tuple[Any, bool, bool]] = {} # Stores (value, used_by_current_hist_type, hidden_in_config_menu)

# --- Configuration Management ---

def set_global_anapath(path: Optional[str]) -> None:
    """
    Sets the global anapath variable.
    If a path is provided, anapath is set to that path (resolved to absolute).
    If path is None, anapath remains its initial value (os.getcwd()).
    This function should be called early by the main script if an --anapath CLI arg is used.
    """
    global anapath
    if path:
        anapath = os.path.abspath(path)
        # print(f"Global anapath has been set to: {anapath}")
    # else:
        # print(f"Global anapath remains: {anapath} (no override path provided)")


def get_config(key: str, default: Any = None) -> Any:
    """Gets a configuration value, returning default if not set."""
    tpl = config_parameters.get(key)
    # Return the value if the key exists, otherwise the default
    return tpl[0] if tpl is not None else default

def set_config(key: str, value: Any, used: Optional[bool] = None, hide: Optional[bool] = None) -> None:
    """Sets or updates a configuration parameter."""
    current_val, current_used, current_hide = config_parameters.get(key, (None, True, False))
    # Only update used/hide if explicitly provided
    new_used = used if used is not None else current_used
    new_hide = hide if hide is not None else current_hide
    config_parameters[key] = (value, new_used, new_hide)

def read_config_defaults(hist_list: List[HistogramType], key: str, current_value: Any) -> Tuple[bool, Any]:
    """
    Determines the default value and applicability ('used') of a config key
    based on the histogram type(s). Returns (used, default_value).
    """
    if not hist_list: return False, current_value # Not used if no hists

    hist = hist_list[0]
    count = len(hist_list)
    classname = hist.classname

    used: bool = True
    default_value: Any = None # Start with None, override below

    is_th1 = "TH1" in classname
    is_th2 = "TH2" in classname
    is_th3 = "TH3" in classname
    is_tgraph = "TGraph" in classname
    is_tf1 = "TF1" in classname
    is_categorical = is_th1 and getMember(getMember(hist, "fXaxis"), "fLabels") is not None

    # Determine defaults based on key
    if key == "limits":
        # Default limits based on combined range of all histograms
        all_limits = []
        for h in hist_list:
            axis_limits = extract_axis_limits(h)
            if not all_limits:
                all_limits = [[lim] for lim in axis_limits] # Initialize with first hist's limits
            else:
                for i, lims in enumerate(axis_limits):
                    if i < len(all_limits):
                         all_limits[i].append(lims)
                    else:
                         all_limits.append([lims]) # Should not happen if types are consistent

        # Combine limits: min of mins, max of maxes for each axis
        combined_limits = []
        for axis_lims_list in all_limits:
            valid_mins = [l[0] for l in axis_lims_list if l and l[0] is not None]
            valid_maxes = [l[1] for l in axis_lims_list if l and l[1] is not None]
            min_val = min(valid_mins) if valid_mins else None
            max_val = max(valid_maxes) if valid_maxes else None
            combined_limits.append((min_val, max_val))
        default_value = combined_limits
        used = is_th1 or is_th2 or is_th3 or is_tgraph or is_tf1 # Generally used

    elif key == "x-label":
        default_value = convert_to_latex(getMember(getMember(hist, "fXaxis"), "fTitle", ""))
    elif key == "y-label":
        default_value = convert_to_latex(getMember(getMember(hist, "fYaxis"), "fTitle", ""))
    elif key == "z-label":
        used = is_th3
        default_value = convert_to_latex(getMember(getMember(hist, "fZaxis"), "fTitle", "")) if used else None
    elif key == "x-log": used = not is_categorical # Log scale not typical for categorical
    elif key == "y-log": pass # Always potentially usable
    elif key == "z-log": used = is_th2 or is_th3 # Only for TH2 (color) or TH3 (color)
    elif key == "title": default_value = getattr(hist, 'title', '')
    elif key == "legend": default_value = (count > 1 or bool(get_config("models"))) # Default true if multiple hists or models
    elif key == "names":
        default_value = [getattr(h, 'title', f'Hist {i}') for i, h in enumerate(hist_list)]
        used = get_config("legend", False) or (get_config("stats", False) and count > 1)
    elif key == "stats": used = is_th1 or is_th2 # TH3 stats less common in this style
    elif key == "grid": used = not is_th2 # Grid less common for TH2 pcolormesh
    elif key == "lines": used = is_th3 # Specific to TH3 polyline drawing (if implemented)
    elif key == "errors": used = is_th1 or is_tgraph
    elif key == "colors":
        if is_th1 or is_tgraph or is_tf1:
            # Default colors for multiple lines/markers
            num_items = count + len(get_config("models", [])) # Total items needing colors
            default_value = list(color_list.keys())[:num_items]
        elif is_th2 or is_th3:
            # Default colormap for 2D/3D
            default_value = [color_maps[0]] # Use list for consistency, plotting takes first
    elif key == "palette":
        used = is_th1 or is_tgraph or is_tf1 # Only relevant when using named colors
        default_value = list(color_palettes.keys())[0] if used else None
    elif key == "alphas":
        num_items = count + len(get_config("models", []))
        default_value = [1.0] * num_items
        used = is_th1 or is_th3 or is_tgraph or is_tf1 # Where alpha makes sense
    elif key == "flat": used = is_th3
    elif key == "raster": used = is_th2 or is_th3
    elif key == "angles": used = is_th3; default_value = [30, -60]
    elif key == "thickness": used = is_th1 or is_tgraph or is_tf1; default_value = 1.5
    elif key == "models": used = is_th1 or is_tgraph or is_tf1 # Models typically for 1D
    elif key == "model-params": used = is_th1 or is_tgraph or is_tf1
    elif key == "cutoff": used = is_th1 and not is_categorical # Cutoff for numerical TH1
    elif key == "model-stats": used = bool(get_config("models")) # Used if models are active

    # If current_value is None, use the determined default, otherwise keep current_value
    final_value = default_value if current_value is None else current_value

    # Special handling for colors/alphas list length
    if key in ["colors", "alphas", "names"]:
         num_hists = len(hist_list)
         num_models = len(get_config("models", []))
         expected_len = num_hists + num_models if key in ["colors", "alphas"] else num_hists

         if isinstance(final_value, list):
             if len(final_value) < expected_len:
                 # Extend with defaults if list is too short
                 default_element = "blue" if key == "colors" else (1.0 if key == "alphas" else "DefaultName")
                 final_value.extend([default_element] * (expected_len - len(final_value)))
             elif len(final_value) > expected_len:
                 # Truncate if too long
                 final_value = final_value[:expected_len]
         elif final_value is None: # Ensure it's a list of correct length if None
              default_element = "blue" if key == "colors" else (1.0 if key == "alphas" else "DefaultName")
              final_value = [default_element] * expected_len


    # Ensure limits is always a list (even if empty)
    if key == "limits" and final_value is None:
        final_value = []

    return used, final_value


def refresh_config(hist_list: List[HistogramType]) -> None:
    """Updates config_parameters with defaults and applicability based on histograms."""
    # Need a copy of keys because dict size might change if new keys are added by read_config_defaults
    keys_to_process = list(config_parameters.keys())
    for key in keys_to_process:
        current_value, _, current_hide = config_parameters[key]
        used, new_value = read_config_defaults(hist_list, key, current_value)
        set_config(key, new_value, used, current_hide) # Update value and used status


def request_additional_config(hist_list: List[HistogramType]):
    """Interactive prompt to modify configuration parameters."""
    hist = hist_list[0]
    show_hidden = False

    while True:
        refresh_config(hist_list) # Ensure defaults and 'used' status are up-to-date

        options: Dict[int, Tuple[str, Tuple[type, Optional[type]]]] = {}
        print(f"\n--- Configure Plot: {green}{get_config('title', 'Untitled')}{reset} ---")
        i = 0
        sorted_keys = sorted(config_parameters.keys()) # Display alphabetically
        for key in sorted_keys:
            value, used, hide = config_parameters[key]
            if not used or (hide and not show_hidden):
                continue

            # Determine type for input parsing help
            outer_type = type(value) if value is not None else None
            inner_type = None
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                inner_type = type(value[0]) if value[0] is not None else None
            elif isinstance(value, dict) and value:
                 inner_type = type(next(iter(value.values()))) # Type of first value in dict

            options[i] = (key, (outer_type, inner_type))
            print(f"  {f'({i})':>3} {display_property(key, value)}")
            i += 1

        print("\nCommands: (idx) edit, (r) reset all, (h) hist info,")
        hidden_text = "hide unused" if show_hidden else "show unused"
        print(f"          (e) {hidden_text}, (ENTER) plot, (q) quit")

        input_indices = [str(j) for j in options.keys()]
        answer = custom_input("Choose command or index: ", keys=["r", "h", "e", "q"] + input_indices)

        if answer.strip() == "": break # ENTER -> plot
        elif answer == "q": sys.exit("Exited by user.")
        elif answer == "r":
            print("\nResetting parameters to defaults...")
            # Reset by setting all values to None, then refresh
            for key in config_parameters.keys():
                set_config(key, None)
            refresh_config(hist_list) # Recalculate defaults
            continue
        elif answer == "h":
            print("\n--- Histogram Info ---")
            try:
                print(f"Class: {hist.classname}")
                print(f"Title: {getattr(hist, 'title', 'N/A')}")
                print(f"Entries: {hist.num_entries}" if hasattr(hist, 'num_entries') else 'N/A')
                # Add more members if useful
                # print("All members:", hist.all_members) # Can be very verbose
            except Exception as e:
                print(f"Could not get histogram info: {e}")
            input("Press Enter to continue...") # Pause
            continue
        elif answer == "e":
            show_hidden = not show_hidden
            continue

        # --- Edit Parameter ---
        try:
            idx = int(answer)
            if idx not in options: raise ValueError("Index out of range")
            option_key, option_type_info = options[idx]
            current_value = get_config(option_key)

            # Toggle booleans directly
            if option_type_info[0] == bool:
                set_config(option_key, not current_value)
                print(f"Set {option_key} to {get_config(option_key)}")
                continue

            # Prompt for new value
            prompt_msg = f"Enter new value for {blue}{option_key}{reset} (Type: {option_type_info[0].__name__ if option_type_info[0] else 'Any'}, current: {red}{current_value}{reset}): "
            value_in = custom_input(prompt_msg).strip()

            if value_in == "escape": continue # User cancelled edit

            # Parse the input based on expected type
            new_value = parse_input_value(value_in, option_key, option_type_info)
            set_config(option_key, new_value)

        except ValueError as e:
            print(f"{red}Invalid input: {e}{reset}", file=sys.stderr)
        except Exception as e:
            print(f"{red}Error setting value: {e}{reset}", file=sys.stderr)


def parse_input_value(value_in: str, key: str, type_info: Tuple[Optional[type], Optional[type]]) -> Any:
    """Parses string input based on expected type information."""
    outer_type, inner_type = type_info

    if value_in.lower() == 'none': return None
    if outer_type is None: return value_in # No type info, return as string

    try:
        if outer_type == bool:
            if value_in.lower() in ["yes", "y", "true", "t", "1"]: return True
            if value_in.lower() in ["no", "n", "false", "f", "0"]: return False
            raise ValueError("Enter yes/no, true/false, or 1/0")
        elif outer_type == int:
            return int(value_in)
        elif outer_type == float:
            return float(value_in)
        elif outer_type == str:
            return value_in # Already a string
        elif key == "limits": # Special parsing for limits
             return parse_limits(value_in)
        elif outer_type in (list, tuple, np.ndarray):
            if inner_type is None: # No inner type info, assume string
                inner_type = str
            # Split by comma or space
            separator = "," if "," in value_in else " "
            items = [item.strip() for item in value_in.split(separator) if item.strip()]
            # Convert each item
            converted_items = [inner_type(item) for item in items]
            return outer_type(converted_items) if outer_type != np.ndarray else np.array(converted_items)
        elif outer_type == dict:
            # Assume JSON format for dictionary input
            return json.loads(value_in)
        else:
            # Try direct conversion for other types
            return outer_type(value_in)
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"Cannot convert '{value_in}' to {outer_type.__name__ if outer_type else 'target type'}: {e}")


# --- ROOT File Interaction ---

def display_rootpaths(classname: str) -> str:
    """Formats a ROOT object path for display with colors."""
    base_name = classname.split(';')[0]
    parts = base_name.split('/')
    if not parts: return ""
    arrow = f" {reset}→{blue} " # Color the arrow
    if len(parts) == 1:
        return f"{red}{parts[0]}{reset}" # Object in root dir
    formatted = arrow.join([f"{p}" for p in parts[:-1]]) # Path parts in blue
    formatted = f"{blue}{formatted}{arrow}{red}{parts[-1]}{reset}" # Last part (object name) in red
    return formatted


def find_histograms(root_file, path_keywords: List[str], stack: bool) -> List[str]:
    """Finds histograms in a ROOT file matching keywords, handles user selection."""
    try:
        # Use classnames() for efficiency if available, otherwise keys(filter_classname=...)
        if hasattr(root_file, 'classnames'):
            all_items = root_file.classnames()
        else:
            # Fallback for older uproot? Might be slow.
            all_items = {k: root_file[k].classname for k in root_file.keys(recursive=True)}

        # Filter by keywords and allowed classes
        found = [
            key for key, classname in all_items.items()
            if all(word.lower() in key.lower() for word in path_keywords) and
               any(allowed_c in classname for allowed_c in allowed_classes)
        ]
    except Exception as e:
         print(f"Error accessing keys/classnames in ROOT file: {e}", file=sys.stderr)
         return []


    if not found:
        print(f"\n{red}ERROR: No histograms found matching keywords: {' '.join(path_keywords)}{reset}", file=sys.stderr)
        return [] # Return empty list, let caller handle exit

    # Sort found keys for consistent ordering
    found.sort()

    if len(found) > 25: # Limit displayed options
        print(f"\n{red}ERROR: Found {len(found)} histograms matching keywords: {' '.join(path_keywords)}. Please be more specific.{reset}", file=sys.stderr)
        # Optionally list the first few?
        # for i, key in enumerate(found[:10]):
        #      print(f"  ({i}) {all_items[key]}: {display_rootpaths(key)}")
        # print("  ...")
        return [] # Return empty list

    if stack:
        if len(found) == 1:
            print(f"{red}Warning: Only one histogram found, cannot stack. Selecting it.{reset}", file=sys.stderr)
            return found

        print("\nFound histograms for stacking:")
        for i, key in enumerate(found):
            print(f"  ({i}) {all_items[key]}: {display_rootpaths(key)}")

        while True:
            indices_str = custom_input(f"\nEnter indices to stack (e.g., '0 1 3'), or ENTER for all: ")
            if indices_str == "escape": return [] # User cancelled
            if not indices_str.strip():
                selected = found # Select all
                break
            try:
                indices = [int(idx.strip()) for idx in indices_str.split()]
                if not indices: raise ValueError("No indices entered.")
                if any(idx < 0 or idx >= len(found) for idx in indices):
                    raise ValueError("Index out of range.")
                if len(indices) < 2:
                     print(f"{red}Warning: Stacking requires at least two histograms.{reset}", file=sys.stderr)
                     # Allow selecting one? Or force re-entry? For now, allow one.
                     # continue
                selected = [found[idx] for idx in indices]
                # Check type compatibility for stacking (optional but recommended)
                first_type = all_items[selected[0]]
                if not all(all_items[key] == first_type for key in selected):
                     print(f"{red}Warning: Selected histograms have different types. Stacking might fail or produce unexpected results.{reset}", file=sys.stderr)
                break
            except ValueError as e:
                print(f"{red}Invalid selection: {e}. Please try again.{reset}", file=sys.stderr)

        return selected

    else: # Not stacking
        if len(found) == 1:
            return found # Only one found, return it directly

        print("\nFound multiple histograms:")
        for i, key in enumerate(found):
            print(f"  ({i}) {all_items[key]}: {display_rootpaths(key)}")

        input_indices = [str(i) for i in range(len(found))]
        while True:
            choice = custom_input(f"\nSelect index (0-{len(found)-1}), or ENTER for first: ", keys=input_indices)
            if choice == "escape": return [] # User cancelled
            if not choice.strip():
                idx = 0
                break
            try:
                idx = int(choice)
                if not (0 <= idx < len(found)): raise ValueError("Index out of range.")
                break
            except ValueError as e:
                print(f"{red}Invalid selection: {e}. Please try again.{reset}", file=sys.stderr)

        return [found[idx]] # Return selection as a list


def getMember(obj: Any, member_name: str, default: Any = None) -> Optional[Any]:
    """Safely gets a member using uproot's member() method or getattr()."""
    if hasattr(obj, 'member'):
        try:
            return obj.member(member_name)
        except uproot.exceptions.KeyInFileError: # Or other specific uproot exceptions
            return default
        except Exception: # Catch other potential errors during member access
             return default
    elif hasattr(obj, member_name):
         try:
             return getattr(obj, member_name)
         except Exception:
              return default
    else:
        return default


def extract_axis_limits(hist: HistogramType) -> List[Tuple[Optional[float], Optional[float]]]:
    """Extracts axis limits (min, max) from various histogram types."""
    limits: List[Tuple[Optional[float], Optional[float]]] = []
    classname = hist.classname

    try:
        if "TH1" in classname:
            axis = getMember(hist, "fXaxis")
            if axis:
                nbins = getMember(axis, "fNbins")
                xmin = getMember(axis, "fXmin")
                xmax = getMember(axis, "fXmax")
                if nbins is not None and xmin is not None and xmax is not None:
                     # Edges might be more reliable if available
                     try:
                          edges = hist.axis().edges()
                          limits.append((edges[0], edges[-1]))
                     except Exception:
                          limits.append((xmin, xmax)) # Fallback to axis limits
                else: limits.append((None, None))
            else: limits.append((None, None))

        elif "TH2" in classname:
            xaxis = getMember(hist, "fXaxis")
            yaxis = getMember(hist, "fYaxis")
            xlim = (getMember(xaxis, "fXmin"), getMember(xaxis, "fXmax")) if xaxis else (None, None)
            ylim = (getMember(yaxis, "fXmin"), getMember(yaxis, "fXmax")) if yaxis else (None, None)
            # Check if numpy edges are better?
            try:
                 _, x_edges, y_edges = hist.to_numpy()
                 xlim = (x_edges[0], x_edges[-1])
                 ylim = (y_edges[0], y_edges[-1])
            except Exception: pass # Keep axis limits if numpy fails
            limits.extend([xlim, ylim])

        elif "TH3" in classname:
            xaxis = getMember(hist, "fXaxis")
            yaxis = getMember(hist, "fYaxis")
            zaxis = getMember(hist, "fZaxis")
            xlim = (getMember(xaxis, "fXmin"), getMember(xaxis, "fXmax")) if xaxis else (None, None)
            ylim = (getMember(yaxis, "fXmin"), getMember(yaxis, "fXmax")) if yaxis else (None, None)
            zlim = (getMember(zaxis, "fXmin"), getMember(zaxis, "fXmax")) if zaxis else (None, None)
            try:
                 _, x_edges, y_edges, z_edges = hist.to_numpy()
                 xlim = (x_edges[0], x_edges[-1])
                 ylim = (y_edges[0], y_edges[-1])
                 zlim = (z_edges[0], z_edges[-1])
            except Exception: pass
            limits.extend([xlim, ylim, zlim])

        elif "TGraph" in classname:
            try:
                x_vals, y_vals = hist.values()
                xlim = (np.min(x_vals), np.max(x_vals)) if len(x_vals) > 0 else (None, None)
                ylim = (np.min(y_vals), np.max(y_vals)) if len(y_vals) > 0 else (None, None)
                limits.extend([xlim, ylim])
            except Exception: limits.extend([(None, None), (None, None)])

        elif "TF1" in classname:
            xmin = getMember(hist, "fXmin")
            xmax = getMember(hist, "fXmax")
            limits.append((xmin, xmax))
            # Y limits for TF1 are harder, maybe evaluate? For now, None.
            # limits.append((None, None))

        else:
            limits.append((None, None)) # Default for unknown types

    except Exception as e:
        print(f"Warning: Could not extract axis limits for {getattr(hist, 'title', classname)}: {e}", file=sys.stderr)
        # Return Nones based on expected dimension
        if "TH3" in classname: return [(None, None)] * 3
        if "TH2" in classname or "TGraph" in classname: return [(None, None)] * 2
        return [(None, None)] * 1

    # Ensure all limits are tuples of (float/None, float/None)
    sanitized_limits = []
    for lim_pair in limits:
        # Check if lim_pair is None before accessing its elements
        if lim_pair is None:
            sanitized_limits.append((None, None))
        else:
            l_min = float(lim_pair[0]) if lim_pair[0] is not None else None
            l_max = float(lim_pair[1]) if lim_pair[1] is not None else None
            sanitized_limits.append((l_min, l_max))

    return sanitized_limits


# --- Input/Output & Formatting ---

def custom_input(message: Optional[str] = "", keys: Optional[List[str]] = []) -> str:
    """Gets user input with optional key bindings and escape handling."""
    bindings = KeyBindings()

    # Add specific key bindings if provided
    for key in keys:
        # Use a closure to capture the key value correctly
        def _handler(event, k=key):
            event.app.exit(result=k)
        bindings.add(key)(_handler)

    # Add escape key binding
    @bindings.add("escape")
    def _escape(event):
        event.app.exit(result="escape")

    # Add Ctrl+C binding to exit gracefully
    @bindings.add('c-c')
    def _interrupt(event):
         print("\nInterrupted by user.")
         sys.exit(1)


    session = PromptSession(key_bindings=bindings)
    try:
        # Use ANSI to allow colored messages
        result = session.prompt(ANSI(message))
        return result
    except EOFError: # Handle Ctrl+D
         print("\nExiting due to EOF.")
         sys.exit(0)
    except KeyboardInterrupt: # Should be caught by c-c binding, but as fallback
         print("\nInterrupted by user.")
         sys.exit(1)


def set_equal_nice_ticks(ax: plt.Axes, max_ticks: int = 7):
    """Sets reasonably spaced 'nice' ticks on all axes of a 3D plot."""
    if not hasattr(ax, 'get_zlim'): return # Only for 3D axes

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    zmin, zmax = ax.get_zlim()

    def compute_tick_step(lo, hi, max_ticks):
        span = hi - lo
        if span <= 0: return 1.0 # Avoid division by zero or log(0)

        raw_step = span / max(1, max_ticks - 1)
        if raw_step <= 0: return 1.0

        exponent = np.floor(np.log10(raw_step))
        factor = raw_step / (10 ** exponent)

        if factor <= 1: nice_factor = 1
        elif factor <= 2: nice_factor = 2
        elif factor <= 5: nice_factor = 5
        else: nice_factor = 10
        step = nice_factor * (10 ** exponent)
        # Ensure step is not zero
        return max(step, 1e-9)


    ax.xaxis.set_major_locator(MultipleLocator(compute_tick_step(xmin, xmax, max_ticks)))
    ax.yaxis.set_major_locator(MultipleLocator(compute_tick_step(ymin, ymax, max_ticks)))
    # Allow slightly more ticks on Z potentially
    ax.zaxis.set_major_locator(MultipleLocator(compute_tick_step(zmin, zmax, max_ticks + 1)))


def get_color(color_name: str, palette: str) -> str:
    """Resolves a color name (like 'red') or hex string using the specified palette."""
    if color_name in color_list:
        index = color_list[color_name]
        try:
            return color_palettes[palette][index]
        except (KeyError, IndexError):
            print(f"Warning: Palette '{palette}' or index {index} not found. Using fallback.", file=sys.stderr)
            # Fallback to first color of first palette or a default
            return color_palettes[list(color_palettes.keys())[0]][0] if color_palettes else "#0000FF" # Default blue
    elif mcolors.is_color_like(color_name):
        return color_name # It's already a valid color (hex, rgb, etc.)
    else:
        print(f"Warning: Invalid color specified: '{color_name}'. Using fallback.", file=sys.stderr)
        return "#FF0000" # Default red


def convert_to_latex(text: Optional[str]) -> str:
    """Converts a string for use in Matplotlib LaTeX labels."""
    if text is None: return ""
    # Basic replacements
    text = text.strip()
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("%", r"\%")
    # Handle units in brackets, add space before
    text = re.sub(r'(?<!\\)\s*\[([^\]]+)\]', r'\\;[\1]', text) # Add space before [units]
    # Greek letters (add more as needed)
    text = text.replace("mu", r"\mu").replace("sigma", r"\sigma").replace("Delta", r"\Delta")
    text = text.replace("omega", r"\omega").replace("phi", r"\phi").replace("tau", r"\tau")
    text = text.replace("chi2", r"\chi^2")
    # Wrap in $ if not already done
    if not text.startswith("$") and not text.endswith("$"):
        text = f"${text}$"
    # Handle cases like $\chi^2$/ndf
    text = text.replace("$/$", "/").replace("/$", "/") # Avoid breaking math mode for slash
    return text


def parse_limits(arg_value: str) -> List[Optional[Tuple[Optional[float], Optional[float]]]]:
    """Parses a string like '-5:5,,0:10' or '-5 5 0 10' into limits list."""
    if not arg_value: return []

    # Standardize separators: replace whitespace sequences with single space, then treat comma and space similarly
    standardized_value = re.sub(r'\s+', ' ', arg_value).strip()

    # Split by comma first if present, otherwise by space
    tokens = re.split(r'\s*,\s*', standardized_value) if ',' in standardized_value else standardized_value.split(' ')

    result: List[Optional[Tuple[Optional[float], Optional[float]]]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token and len(result) < 3: # Allow empty token for skipping axis (e.g., ,,)
             result.append(None)
             i += 1
             continue

        # Try parsing token as a pair (e.g., "min:max")
        try:
            parsed_pair = parse_group(token)
            result.append(parsed_pair)
            i += 1
        except argparse.ArgumentTypeError:
            # If not a pair, assume it's a single value (min) and next token is max
            if i + 1 < len(tokens):
                token2 = tokens[i+1]
                try:
                    min_val = float(token) if token else None
                    max_val = float(token2) if token2 else None
                    result.append((min_val, max_val))
                    i += 2
                except ValueError:
                     raise argparse.ArgumentTypeError(f"Cannot parse '{token} {token2}' as a min max pair.")
            else:
                 # Treat single token as max if it starts with ':', otherwise min? Ambiguous.
                 # Let's require pairs or explicit ':'
                 raise argparse.ArgumentTypeError(f"Incomplete limit pair starting with '{token}'. Use 'min:max' or 'min max'.")

    # Pad with None if fewer than 3 axes were specified
    while len(result) < 3:
        result.append(None)

    return result[:3] # Return max 3 axes (x, y, z)


def parse_group(group: str) -> Tuple[Optional[float], Optional[float]]:
    """Parses 'min:max', 'min max', ':max', 'min:' into (min, max) tuple."""
    group = group.strip()
    if not group: return (None, None)

    parts = []
    if ':' in group:
        parts = [p.strip() for p in group.split(':', 1)]
        if len(parts) == 1: # Handle "val:" or ":val"
             if group.startswith(':'): parts.insert(0, '') # Becomes ['', 'val']
             else: parts.append('') # Becomes ['val', '']
    else:
        # Try splitting by space, only if no colon was present
        parts = [p.strip() for p in group.split(None, 1)] # Split by whitespace

    if len(parts) == 0: return (None, None)
    if len(parts) == 1:
         # Single value - assume it's min OR max? Let's disallow for clarity.
         # Or maybe treat as 'val:val'? No, requires pairs.
         raise argparse.ArgumentTypeError(f"Ambiguous limit '{group}'. Use 'min:max' or 'min max'.")
    if len(parts) > 2:
         raise argparse.ArgumentTypeError(f"Cannot parse limit group '{group}'. Too many parts.")

    def conv(s: str) -> Optional[float]:
        return float(s) if s else None

    try:
        min_val = conv(parts[0])
        max_val = conv(parts[1])
        return (min_val, max_val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Cannot convert '{parts[0]}' or '{parts[1]}' to float.")


# --- Caching ---

def _sanitize_path_for_cache_component(path_str: str) -> str:
    """Sanitizes a path string to be used as a directory component for caching."""
    # Normalize path first to handle mixed separators or redundant parts
    sanitized = os.path.normpath(path_str)
    
    # Remove drive letter prefix like C: (Windows)
    sanitized = re.sub(r"^[a-zA-Z]:", "", sanitized)
    
    # Remove leading path separator (e.g., / or \)
    if sanitized.startswith(os.path.sep):
        sanitized = sanitized[len(os.path.sep):]
    
    # Replace path separators with underscore
    sanitized = sanitized.replace(os.path.sep, "_")
    
    # Replace any remaining non-alphanumeric characters (allow underscore, hyphen) with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
    
    # Consolidate multiple underscores that might have been introduced
    sanitized = re.sub(r"__+", "_", sanitized)
    
    # Remove leading/trailing underscores that might result from replacements
    sanitized = sanitized.strip("_")
    
    # If the path was something like "/" or "C:/", it might become empty.
    if not sanitized:
        return "root_anapath" # Default for root-like or empty paths
    return sanitized

def _get_cache_file_path() -> str:
    """
    Determines the full path to the cache.json file based on the global anapath.
    The cache will be stored in a subdirectory of COMMON_CACHE_BASE_DIR,
    named after a sanitized version of the current anapath.
    """
    global anapath # Uses the global anapath set by set_global_anapath
    
    sanitized_anapath_component = _sanitize_path_for_cache_component(anapath)
    
    anapath_specific_cache_dir = os.path.join(COMMON_CACHE_BASE_DIR, sanitized_anapath_component)
    os.makedirs(anapath_specific_cache_dir, exist_ok=True) # Ensure directory exists
    
    return os.path.join(anapath_specific_cache_dir, CACHE_FILE)


def save_cache(matching_keys: List[str], root_path: str, output_path: str, title: str, edit_index: Optional[int] = None) -> None:
    """Saves the current configuration and context to a JSON cache file."""
    # Make config serializable (convert numpy arrays if any)
    serializable_config = {}
    for key, (value, used, hide) in config_parameters.items():
        if isinstance(value, np.ndarray):
            serializable_config[key] = (value.tolist(), used, hide)
        else:
            serializable_config[key] = (value, used, hide)

    new_entry = {
        "title": title,
        "config": serializable_config,
        "keys": matching_keys,
        "root_path": root_path,
        "output_path": output_path, # Save output path used
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # ISO-like format
    }

    cache_path: str = _get_cache_file_path()
    print(f"Saving cache entry to: {cache_path}")
    try:
        with open(cache_path, "r") as f:
            cache_entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache_entries = []

    if edit_index is not None and 0 <= edit_index < len(cache_entries):
        cache_entries[edit_index] = new_entry # Overwrite existing
    else:
        cache_entries.insert(0, new_entry) # Prepend new entry
        cache_entries = cache_entries[:50] # Limit cache size

    try:
        with open(cache_path, "w") as f:
            json.dump(cache_entries, f, indent=2) # Use indent=2 for readability
    except IOError as e:
        print(f"{red}Error writing cache file {cache_path}: {e}{reset}", file=sys.stderr)


def load_cache(entry_index: int = -1) -> Optional[Tuple[Dict[str, Any], List[str], str, str, str]]:
    """Loads configuration and context from the JSON cache file."""
    cache_path: str = _get_cache_file_path()
    try:
        with open(cache_path, "r") as f:
            cache_entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"{red}Cache file '{cache_path}' not found or invalid.{reset}", file=sys.stderr)
        return None

    if not cache_entries:
        print(f"{red}No cache entries found in '{cache_path}'.{reset}", file=sys.stderr)
        return None

    # --- Interactive Selection if index is -1 ---
    if entry_index == -1:
        print("\nAvailable cache entries:")
        for i, entry_data in enumerate(cache_entries):
            date_str = entry_data.get('date', 'unknown date')
            title_str = entry_data.get('title', 'Untitled')
            print(f"  ({i}) {date_str:<20} {green}{title_str}{reset}")

        input_indices = [str(i) for i in range(len(cache_entries))]
        while True:
            choice = custom_input(f"\nSelect cache entry index (0-{len(cache_entries)-1}): ", keys=input_indices)
            if choice == "escape": return None # User cancelled
            try:
                entry_index = int(choice)
                if not (0 <= entry_index < len(cache_entries)): raise ValueError("Index out of range.")
                break
            except ValueError as e:
                print(f"{red}Invalid selection: {e}. Please try again.{reset}", file=sys.stderr)
    # --- End Interactive Selection ---

    if not (0 <= entry_index < len(cache_entries)):
        print(f"{red}Cache entry index {entry_index} out of range.{reset}", file=sys.stderr)
        return None

    selected = cache_entries[entry_index]
    title = selected.get('title', 'Untitled')
    print(f"Loading cache entry {entry_index}: {green}{title}{reset}")

    # Load config back into global state
    global config_parameters
    config_parameters.clear()
    config_parameters.update(selected.get("config", {}))

    # Return loaded data
    return (
        selected.get("config", {}),
        selected.get("keys", []),
        selected.get("root_path", ""),
        selected.get("output_path", ""), # Return output path
        title
    )


# --- SVG Optimization ---

def optimize_svg(svg_file: str) -> None:
    """Optimizes an SVG file using scour if available."""
    try:
        # Check if scour is available
        subprocess.run(["scour", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Info: 'scour' not found. Skipping SVG optimization.", file=sys.stderr)
        return

    print(f"Optimizing SVG file: {svg_file}")
    tmp_file = svg_file + ".tmp"
    try:
        # Run scour
        result = subprocess.run(
            ["scour", "--enable-id-stripping", "--enable-comment-stripping",
             "--shorten-ids", "--indent=none", "-i", svg_file, "-o", tmp_file],
            check=True, capture_output=True, text=True
        )
        if result.stderr:
             print(f"Scour warning: {result.stderr}", file=sys.stderr)

        # Replace original file
        os.replace(tmp_file, svg_file)
        print("SVG optimization complete.")

    except FileNotFoundError:
         # This case should be caught by the initial check, but handle defensively
         print("Error: 'scour' command not found during execution.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"{red}Error during SVG optimization: {e}{reset}", file=sys.stderr)
        print(f"Scour stdout: {e.stdout}")
        print(f"Scour stderr: {e.stderr}")
        # Clean up temp file if it exists
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
    except Exception as e:
        print(f"{red}An unexpected error occurred during SVG optimization: {e}{reset}", file=sys.stderr)
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
