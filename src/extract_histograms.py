import os
import argparse
import sys
import re
import subprocess
import uproot
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Union

# Import from our library
# Need to adjust sys.path if 'script' is run directly and 'src' is not in PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from lib.utils import (anapath, config_parameters, load_cache, find_histograms,
                         request_additional_config, refresh_config, save_cache,
                         optimize_svg, set_config, get_config, parse_limits, getMember,
                         set_global_anapath, display_property) # Added display_property
    from lib.plotting import process_histograms
    from lib.constants import allowed_classes, green, reset, red # Added red for error messages
except ImportError as e:
    print(f"Error importing library modules: {e}", file=sys.stderr)
    print(f"Please ensure the script is run from the project root or 'src' is in PYTHONPATH.", file=sys.stderr)
    sys.exit(1)


def parse_run_number(run_str: str) -> Union[int, str]:
    """
    Parse a run number that could be either a single number (e.g., "510") 
    or a range (e.g., "510-515").
    Returns an int for single run or the original string for a range.
    """
    if run_str is None:
        return None
        
    # Check if it's a simple integer
    if re.match(r'^\d+$', run_str):
        return int(run_str)
    
    # Check if it's a range pattern like "510-515"
    if re.match(r'^\d+-\d+$', run_str):
        return run_str  # Return as string for ranges
    
    # Invalid format
    raise ValueError(f"Run number must be a single number or range (e.g., '510' or '510-515'), got '{run_str}'")


def extract_and_plot(run: Optional[Union[int, str]],
                     type_name: Optional[str],
                     comment: Optional[str] = None,
                     paths: Optional[List[List[str]]] = None,
                     config_interactive: Optional[bool] = False,
                     stack: Optional[bool] = False,
                     edit_index: Optional[int] = None) -> Optional[str]:
    """
    Loads data, finds histograms, handles configuration, plots, and saves.
    Returns the output path if successful, None otherwise.
    """
    global config_parameters # Allow modification of the global dict
    # Access anapath from utils, which would have been set by set_global_anapath
    current_anapath = anapath 

    matching_keys: List[str] = []
    root_path: str = ""
    output_path: str = ""
    plot_title: str = "Histogram"
    loaded_config = None

    # --- Load Cache or Determine Paths ---
    if edit_index is not None:
        load_result = load_cache(edit_index)
        if load_result is None:
            return None # Error handled in load_cache
        loaded_config, matching_keys, root_path, output_path, plot_title = load_result
        # config_parameters is updated globally by load_cache

        # Check if user provided conflicting info (run, type, comment)
        new_root_file_specified = (run is not None or type_name is not None or comment is not None)
        if new_root_file_specified:
             print("New run/type/comment provided, ignoring cached root file path.")
             # Rebuild root_path based on new args
             if run is None or type_name is None:
                  print(f"{red}ERROR: --run and --type are required when overriding cache file.{reset}", file=sys.stderr)
                  return None
             root_file_name = f"histograms_ana_{run}" + (f"-{comment}" if comment else "") + ".root"
             # Use current_anapath which is set by main() via --anapath CLI arg
             root_path = os.path.join(current_anapath, type_name, root_file_name)
             print(f"Using new ROOT file path: {root_path}")
             # Force re-finding histograms unless paths are also cached/provided
             if not paths:
                  print("No --path provided with new file, attempting to use cached keys (might fail).")
                  # matching_keys are already loaded from cache
             else:
                  matching_keys = [] # Force re-finding with new paths
        else:
             print(f"Using cached ROOT file: {root_path}")
             # If paths are provided by user, override cached keys
             if paths:
                  print("Using provided --path arguments, ignoring cached keys.")
                  matching_keys = [] # Force re-finding
             else:
                  print("Using cached histogram keys.")
                  # matching_keys are already loaded

        # Use cached output path as default, but allow override later if needed
        # The output filename might change based on title/run etc.

    else: # Not editing, requires run, type, path
        if run is None or type_name is None or not paths:
            print(f"{red}ERROR: --run, --type, and --path are required when not using --edit.{reset}", file=sys.stderr)
            return None
        root_file_name = f"histograms_ana_{run}" + (f"-{comment}" if comment else "") + ".root"
        # Check first if the file exists directly under type_name directory
        root_path_direct = os.path.join(current_anapath, type_name, root_file_name)
        if os.path.exists(root_path_direct):
            root_path = root_path_direct
            print(f"DEBUG: Found ROOT file at: {root_path}")
        else:
            print(f"{red}ERROR: The data path '{root_path_direct}' is not valid.{reset}", file=sys.stderr)  
            
            # print error, that the datapath is not valid:
        matching_keys = [] # Will be found below

    # --- Open ROOT File and Find Histograms ---
    try:
        with uproot.open(root_path) as root_file:
            if not matching_keys: # Find keys if not loaded from cache or overridden
                found_keys = []
                for p_list in paths:
                    keys = find_histograms(root_file, p_list, stack or (len(paths) > 1 and len(p_list) > 1)) # Heuristic for stacking intent
                    if not keys:
                         print(f"{red}Could not find histograms for path keywords: {' '.join(p_list)}{reset}", file=sys.stderr)
                         # Decide whether to exit or continue with other paths
                         # For now, let's try to continue
                    found_keys.extend(keys)

                if not found_keys:
                     print(f"{red}ERROR: No histograms found for any provided path(s).{reset}", file=sys.stderr)
                     return None
                matching_keys = list(dict.fromkeys(found_keys)) # Remove duplicates while preserving order

            print("\nProcessing histogram(s):")
            hist_list = []
            for key in matching_keys:
                try:
                    hist = root_file[key]
                    # Add title attribute if missing (useful for defaults)
                    if not hasattr(hist, 'title'):
                         setattr(hist, 'title', getMember(hist, "fTitle", "Untitled"))
                    hist_list.append(hist)
                    print(f"  - {root_file.classnames()[key]}: {green}{hist.title}{reset}")
                except uproot.exceptions.KeyInFileError:
                     print(f"{red}ERROR: Key '{key}' not found in file {root_path}.{reset}", file=sys.stderr)
                     return None # Exit if a specified key is missing
                except Exception as e:
                     print(f"{red}ERROR: Could not load histogram '{key}': {e}{reset}", file=sys.stderr)
                     return None

            if not hist_list:
                 print(f"{red}ERROR: No valid histograms could be loaded.{reset}", file=sys.stderr)
                 return None

            # Setup names in config if not already set
            if get_config("names", None) is None:
                names = [getattr(h, 'title', f'Hist {i}') for i, h in enumerate(hist_list)]
                used = get_config("legend", False) or (get_config("stats", False) and len(hist_list) > 1)
                set_config("names", names, used)

            # --- Configuration ---
            # Initial refresh based on loaded histograms
            refresh_config(hist_list)

            # Interactive configuration if requested
            if config_interactive:
                request_additional_config(hist_list)
            else:
                 # Ensure config is consistent even if not interactive
                 refresh_config(hist_list)

            # --- Plotting ---
            fig, final_title, extra_artists = process_histograms(hist_list)
            if fig is None:
                print(f"{red}ERROR: Plotting failed.{reset}", file=sys.stderr)
                return None

            # --- Saving Output ---
            # Determine output directory and filename
            # Use current_anapath
            output_base_dir = os.path.join(current_anapath, "output", type_name) # Example output structure
            os.makedirs(output_base_dir, exist_ok=True)

            # Construct filename (use run/type/comment if not editing, otherwise try to keep consistent)
            if edit_index is None: # New plot
                 filename_parts = [f"run_{run}", type_name]
                 if comment: filename_parts.append(comment)
                 # Use sanitized title from config or hist
                 sanitized_title = re.sub(r'[^\w\-_.]', '_', get_config("title", hist_list[0].title or "plot"))
                 filename_parts.append(sanitized_title)
                 output_filename = "-".join(filename_parts) + ".svg"
                 output_path = os.path.join(output_base_dir, output_filename)
            else: # Editing - try to reuse output path structure if possible
                 # If output_path was loaded and seems valid, use its dir
                 if output_path and os.path.isdir(os.path.dirname(output_path)):
                      output_base_dir = os.path.dirname(output_path)
                 # Update filename based on potentially changed title/run etc.
                 current_run = run if run is not None else "cached" # Placeholder if run wasn't specified for edit
                 current_type = type_name if type_name is not None else "cached"
                 current_comment = comment # Can be None
                 filename_parts = [f"run_{current_run}", current_type]
                 if current_comment: filename_parts.append(current_comment)
                 sanitized_title = re.sub(r'[^\w\-_.]', '_', get_config("title", plot_title or "plot"))
                 filename_parts.append(sanitized_title)
                 output_filename = "-".join(filename_parts) + ".svg"
                 output_path = os.path.join(output_base_dir, output_filename)

            print(f"\nSaving plot to: {output_path}")
            try:
                 fig.savefig(output_path, dpi=300, bbox_inches='tight', bbox_extra_artists=extra_artists)
                 plt.close(fig) # Close figure to free memory
            except Exception as e:
                 print(f"{red}ERROR: Failed to save figure: {e}{reset}", file=sys.stderr)
                 plt.close(fig)
                 return None

            # --- Save Cache and Optimize ---
            # Determine title for cache entry
            cache_title_parts = [str(run) if run else "edit", type_name if type_name else "edit"]
            if comment: cache_title_parts.append(comment)
            cache_title_parts.append(get_config("title", hist_list[0].title or "plot"))
            cache_title = " - ".join(filter(None, cache_title_parts))

            save_cache(matching_keys, root_path, output_path, cache_title, edit_index if edit_index != -1 else None) # Pass original index if editing specific entry

            if not get_config("raster", False): # Only optimize if not rasterized
                optimize_svg(output_path)

            return output_path

    except FileNotFoundError:
        print(f"\n{red}ERROR: ROOT file not found: {root_path}{reset}", file=sys.stderr)
        return None
    except uproot.exceptions.KeyInFileError as e:
        print(f"\n{red}ERROR: Key not found in ROOT file {root_path}: {e}{reset}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\n{red}An unexpected error occurred: {e}{reset}", file=sys.stderr)
        # import traceback
        # traceback.print_exc() # Optional: print full traceback for debugging
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract and plot ROOT histograms.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) # Better help messages

    # --- Input File Arguments ---
    input_group = parser.add_argument_group('Input Data Selection')
    input_group.add_argument("--anapath", type=str, default=None, # Default to None, set_global_anapath handles os.getcwd() if None
                             help="Base path for analysis data and output. If not set, uses current working directory or utils default.")
    input_group.add_argument("--run", type=str, help="Run number (e.g., '510') or range (e.g., '510-515') (required unless --edit is used)")
    input_group.add_argument("--type", help="Data type (e.g., calibration, physics) (required unless --edit is used)")
    input_group.add_argument("--comment", default=None, help="Optional comment for file naming")
    input_group.add_argument("--path", action="append", nargs='+',
                             help="Keyword(s) to find histogram(s) in ROOT file. Can be used multiple times. "
                                  "Example: --path EventLoader layer_0 corr --path Trigger time")

    # --- Plotting Configuration Arguments ---
    plot_group = parser.add_argument_group('Plotting Configuration (Overrides cache/defaults)')
    plot_group.add_argument("--title", nargs="?", const="", default=argparse.SUPPRESS, help="Plot title. If given without value, uses histogram title.")
    plot_group.add_argument("--limits", type=parse_limits, default=argparse.SUPPRESS, help="Axis limits, e.g., 'xmin:xmax,ymin:ymax' or 'xmn xmx ymn ymx'. Use '' or ',' to skip axis.")
    plot_group.add_argument("--legend", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="Show/hide legend")
    plot_group.add_argument("--stats", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="Show/hide statistics box")
    plot_group.add_argument("--errors", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="Show/hide error bars (TH1/TGraph)")
    plot_group.add_argument("--colors", nargs='+', default=argparse.SUPPRESS, help="Colors (names like 'red' or hex) or colormap (for TH2/3)")
    plot_group.add_argument("--palette", default=argparse.SUPPRESS, help="Color palette for named colors (e.g., color1, color2)")
    plot_group.add_argument("--alphas", nargs='+', type=float, default=argparse.SUPPRESS, help="Transparency values (0.0 to 1.0)")
    plot_group.add_argument("--thickness", type=float, default=argparse.SUPPRESS, help="Line/marker thickness (TH1/TGraph)")
    plot_group.add_argument("--grid", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="Show/hide grid")
    plot_group.add_argument("--x-log", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, dest='x_log', help="Use log scale for X axis")
    plot_group.add_argument("--y-log", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, dest='y_log', help="Use log scale for Y axis")
    plot_group.add_argument("--z-log", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, dest='z_log', help="Use log scale for Z axis/colorbar")
    plot_group.add_argument("--models", nargs='*', default=argparse.SUPPRESS, help="Model function(s) to overlay (e.g., lifetime, f1)")
    plot_group.add_argument("--model-stats", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, dest='model_stats', help="Show/hide model fit statistics box")
    plot_group.add_argument("--cutoff", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="Limit histogram data to specified limits")

    # --- TH3 Specific Arguments ---
    th3_group = parser.add_argument_group('TH3 Specific Options')
    th3_group.add_argument("--flat", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="Plot TH3 as flat slices")
    th3_group.add_argument("--angles", nargs=2, type=float, default=argparse.SUPPRESS, help="View angles (elevation azimuth) for TH3 plots")

    # --- Behavior Arguments ---
    behavior_group = parser.add_argument_group('Script Behavior')
    behavior_group.add_argument("--config", action="store_true", help="Enter interactive configuration mode")
    behavior_group.add_argument("--stack", action="store_true", help="Attempt to stack multiple found histograms (requires compatible types)")
    behavior_group.add_argument("--raster", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="Rasterize TH2/TH3 plots (faster, larger file)")
    behavior_group.add_argument("--edit", nargs="?", const=-1, type=int, default=None,
                                help="Edit previous plot configuration. Optionally provide cache index (e.g., --edit 0 for latest).")

    args = parser.parse_args()

    # Parse run number (validates and converts to int if it's a single number)
    if args.run:
        try:
            args.run = parse_run_number(args.run)
        except ValueError as e:
            print(f"{red}ERROR: {e}{reset}", file=sys.stderr)
            sys.exit(1)

    # Set the global anapath using the provided or default value
    # This needs to be done early, before anapath is used by load_cache or other functions.
    set_global_anapath(args.anapath)
    
    # Add debug prints to help diagnose path issues
    print(f"\nDEBUG: Global anapath set to: {anapath}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    if args.run is not None and args.type is not None:
        # Handle run range in debug output
        if isinstance(args.run, str) and '-' in args.run:
            # For range like "510-515"
            expected_root_path = os.path.join(anapath, args.type, 
                                    f"histograms_ana_{args.run}" + (f"-{args.comment}" if args.comment else "") + ".root")
        else:
            # For single run number (already converted to int)
            expected_root_path = os.path.join(anapath, args.type, 
                                    f"histograms_ana_{args.run}" + (f"-{args.comment}" if args.comment else "") + ".root")
            
        print(f"DEBUG: Expected ROOT file path: {expected_root_path}")
        print(f"DEBUG: File exists: {os.path.exists(expected_root_path)}")

    # --- Initialize Configuration ---
    # Start with empty config, defaults will be filled by refresh_config
    global config_parameters
    config_parameters = {
        # Key: (Default Value, Used by default?, Hidden in config?) - Defaults filled later
        "title": (None, True, False), "x-label": (None, True, False), "y-label": (None, True, False),
        "z-label": (None, False, False), "x-log": (False, True, True), "y-log": (False, True, True),
        "z-log": (False, False, True), "limits": (None, True, False), "legend": (None, True, False),
        "stats": (False, True, False), "grid": (None, True, True), "names": (None, False, False),
        "colors": (None, True, False), "palette": (None, False, True), "alphas": (None, True, True),
        "angles": (None, False, True), "thickness": (None, True, True), "lines": (False, False, True), 
        "cutoff": (False, False, True), "flat": (False, False, True), "raster": (None, False, True),
        "errors": (False, False, True), "models": ([], False, True), "model-params": ({}, False, True),
        "model-stats": (True, False, False), "figsize": ((8,6), False, True),
        "textsize": (12.0, False, True)
    }

    # Apply command-line arguments to override initial config values
    # We use hasattr check because args only contains specified args or those with defaults != argparse.SUPPRESS
    arg_dict = vars(args)
    for key, (default_val, default_used, default_hide) in config_parameters.items():
         if hasattr(args, key) and key in arg_dict and arg_dict[key] is not argparse.SUPPRESS:
              # Map dest names like x_log back to config key x-log
              config_key = key.replace('_', '-')
              # Handle nargs='?' const case for title
              if config_key == 'title' and arg_dict[key] == "":
                   # Special case: --title means use hist title, defer decision
                   set_config(config_key, None) # Set to None, refresh_config will get hist title
              else:
                   set_config(config_key, arg_dict[key])

    # --- Run Extraction and Plotting ---
    output_file = extract_and_plot(args.run, args.type, comment=args.comment, paths=args.path,
                                   config_interactive=args.config or (args.edit is not None), # Force config if editing
                                   stack=args.stack, edit_index=args.edit)

    if output_file:
        print(f"\n{green}Successfully created: {output_file}{reset}")
        # Try to open the file (cross-platform)
        try:
            if sys.platform == "win32":
                os.startfile(output_file)
            elif sys.platform == "darwin":
                subprocess.run(["open", output_file], check=True)
            else:
                subprocess.run(["xdg-open", output_file], check=True)
        except Exception as e:
            print(f"Info: Could not automatically open the output file: {e}", file=sys.stderr)
    else:
        print(f"\n{red}Plot generation failed.{reset}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()