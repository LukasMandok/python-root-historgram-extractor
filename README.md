# ROOT Histogram Extractor

A small Python tool for extracting ROOT histograms, plotting them with Matplotlib, and optionally saving the result as optimized SVG files.

The project is designed around two use cases:

1. **Command line usage** for quickly selecting histograms from analysis ROOT files and producing SVG plots.
2. **Python usage** from scripts, notebooks, or other projects through a small public API that returns a `matplotlib.figure.Figure`.

The plotting code is intentionally separated from command line handling, caching, and file output. The central public objects are `PlotConfig` and `plot_from_root`.

## Features

The existing plotting functionality is preserved during the refactor. Depending on the ROOT object, the tool supports:

- TH1 histograms
- TH2 histograms
- TH3 histograms
- TGraph / TGraphErrors
- TF1 objects where supported by the existing plotting/model workflow
- Multiple histograms and optional stacking/selection
- Linear and logarithmic axes
- Optional x- and y-axis multipliers for displayed tick values
- Configurable plot background color, including transparent mode
- Axis limits
- Legends
- Histogram statistics boxes
- Error bars
- Configurable colors, palettes, transparency, and line thickness
- Model overlays and model statistics
- TH3 slice/angle options
- Rasterized TH2/TH3 output
- Interactive configuration from the terminal
- Configuration caching and later editing with `--edit`
- SVG output with the existing optimization step

## Project structure

```text
.
├── pyproject.toml
├── README.md
└── src/
    ├── extract_histograms.py   # CLI wrapper and I/O
    └── lib/
        ├── __init__.py         # Public package API
        ├── config.py            # PlotConfig
        ├── core.py              # plot_from_root()
        ├── constants.py        # Plotting constants and palettes
        ├── models.py            # Model functions and predefined fits
        ├── plotting.py          # Matplotlib plotting implementation
        └── utils.py             # Histogram search, cache and helpers
```

The public package API is exported from `lib`:

```python
from root_extractor import PlotConfig, plot_from_root
```

Histograms can optionally be rebinned by physical bin width. Use a number or
one-item list for TH1, and one value per axis for TH2/TH3. `None` leaves an
axis unchanged:

```python
config = PlotConfig({"bin_width": [1.0, 0.4]})  # TH2: X and Y widths
figure = plot_from_root(root_file, paths=[["detector", "position"]], config=config)
```

For TH1, `bin_width=1`, `bin_width=[1]`, and `bin_width=[None]` are valid.
Requested widths must align with existing histogram edges.

### Combining 2D histograms into a 3D histogram

The library can materialize one TH2 from each of several ROOT files and stack
them as consecutive slices of a TH3-like object. This uses uproot and NumPy;
PyROOT is not required unless a native ROOT object is needed by downstream
code.

```python
from root_extractor import combine_histograms_3d, load_histograms

histograms = load_histograms(
    ["slice_01.root", "slice_02.root", "slice_03.root"],
    "detector/position",
)
combined = combine_histograms_3d(
    histograms,
    z_edges=[0.0, 10.0, 20.0, 30.0],
    title="Detector position by slice",
)

# Compatible with the existing TH3 plotting code:
values, x_edges, y_edges, z_edges = combined.to_numpy()

# Only when a PyROOT TH3D is required:
root_histogram = combined.to_pyroot("detector_position_3d")
```

The input histograms must have identical X and Y bin edges. Their order is
preserved along Z; if `z_edges` is omitted, the slices use edges `0, 1, ...,
N`. `to_pyroot()` raises a clear `ImportError` when PyROOT is unavailable.

## Installation

The project uses a standard `src` layout and setuptools.

Create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project and its runtime dependencies:

```bash
pip install -e .
```

This makes the `lib` package importable from Python code without manually adding `src` to `PYTHONPATH`, and installs the `root-histogram-extractor` command.

## Running the command line tool

From the project root:

```bash
python src/extract_histograms.py --help
```

The script expects analysis files using the naming convention:

```text
<anapath>/<type>/histograms_ana_<run>.root
```

For example:

```text
/data/testbeam/physics/histograms_ana_510.root
```

with:

```text
--anapath /data/testbeam
--type physics
--run 510
```

The default analysis path is the current working directory when `--anapath` is not specified.

### Basic example

```bash
python src/extract_histograms.py \
    --anapath /data/testbeam \
    --run 510 \
    --type physics \
    --path EventLoader layer_0 corr
```

This searches the ROOT file for objects whose ROOT key contains all three keywords, loads the matching histogram(s), plots them, and writes the SVG below:

```text
/data/testbeam/output/physics/
```

### Plotting a histogram stored in a canvas

Files containing saved ROOT canvases are searched automatically through the optional CERN ROOT executable. Use the same path query as for an ordinary ROOT file:

```bash
root-histogram-extractor \
    --root-file /home/mue/mandok/musr/Cone_BeamSpot/Plots/Cone_OverviewDownStream_Runs334To3723_250121172837.root \
    --path "XY Cut" "Muon Position" \
    --output muon-position.svg
```

Only the matching histogram primitive is extracted and plotted. The lower-level `plot_canvas_from_root` function remains available for callers that need explicit pad/index selection.

Canvas histograms are also discovered automatically when a normal path query does not find ordinary ROOT objects. For example, this plots the matching histogram without knowing that it is stored in a canvas:

```python
from root_extractor import plot_from_root

figure = plot_from_root(
    "/home/mue/mandok/musr/Cone_BeamSpot/Plots/Cone_OverviewDownStream_Runs334To3723_250121172837.root",
    paths=[["XY Cut", "Muon Position"]],
)
```

The output filename contains the run, type, optional comment, and plot title.

### Multiple histogram selections

`--path` can be specified more than once:

```bash
python src/extract_histograms.py \
    --run 510 \
    --type physics \
    --path EventLoader layer_0 corr \
    --path Trigger time
```

Each `--path` is a list of keywords. All keywords in one path must match the ROOT key.

The search is case-insensitive and only configured ROOT object classes are considered. If a selection matches more than 25 objects, the tool asks you to make the query more specific rather than presenting an excessive list.

### Stacking / interactive selection

Use `--stack` to request stacking behavior when a keyword selection matches multiple compatible objects:

```bash
python src/extract_histograms.py \
    --run 510 \
    --type physics \
    --path EventLoader layer_0 corr \
    --stack
```

When multiple matching objects are found, the tool can ask which entries should be included in the stack.

## Plot options

The CLI exposes the most commonly used plotting settings directly.

### Title

```bash
--title "Muon time spectrum"
```

Passing `--title` without a value uses the histogram title:

```bash
--title
```

### Axis limits

Limits can be written as `min:max` pairs:

```bash
--limits "0:5000,0:1200"
```

This means:

```text
x: 0 ... 5000
y: 0 ... 1200
```

The parser also accepts space-separated values:

```bash
--limits "0 5000 0 1200"
```

An individual bound can be omitted with `:`. For example:

```bash
--limits ":5000,0:"
```

For multi-axis plots, up to three axis limit pairs can be supplied.

### Legend, statistics and errors

Boolean options use Python's `argparse.BooleanOptionalAction`, so both enabling and disabling are supported:

```bash
--legend
--no-legend

--stats
--no-stats

--errors
--no-errors
```

### Logarithmic axes

```bash
--x-log
--y-log
--z-log
```

and explicitly disable them with:

```bash
--no-x-log
--no-y-log
--no-z-log
```

### Colors and alpha values

Multiple colors can be specified:

```bash
--colors red blue green
```

A palette for named colors can be selected with:

```bash
--palette color1
```

Transparency values are given as one or more floating point values between 0 and 1:

```bash
--alphas 1.0 0.8 0.6
```

### Line thickness

```bash
--thickness 2.0
```

### Models

Predefined model names can be overlaid with:

```bash
--models lifetime
```

or multiple models:

```bash
--models lifetime f1
```

The available predefined model functions are maintained in `src/lib/models.py`. The current implementation includes models such as `lifetime`, `lifetime_oscillation`, `asymmetry`, `f1`, `f2`, `f3`, `gps_lifetime`, `gps_lifetime_v2`, `upstream_lifetime`, `upstream_lifetime_v2`, `downstream_lifetime`, and `mu_decay`.

Model parameters can be managed through the configuration object and the interactive configuration workflow.

### TH3 options

For TH3 plots:

```bash
--flat
```

requests the flat slice representation.

View angles can be supplied as two numbers:

```bash
--angles 15 45
```

where the values are used as the elevation and azimuth configuration of the existing TH3 plotting implementation.

### Rasterization

Rasterization can be controlled with:

```bash
--raster
--no-raster
```

It is mainly relevant for TH2 and TH3 plots and can make large SVGs substantially more practical.

## Interactive configuration

Use:

```bash
python src/extract_histograms.py \
    --run 510 \
    --type physics \
    --path EventLoader layer_0 corr \
    --config
```

After the histogram(s) are loaded, an interactive menu is shown.

The menu allows you to:

- modify supported configuration values
- reset the configuration to histogram-dependent defaults
- inspect histogram information
- show or hide normally hidden parameters
- finish configuration by pressing Enter
- quit the interactive workflow

The reset operation does not hard-code histogram defaults. It resets values to an unresolved state and lets the normal histogram-dependent default calculation run again.

## Cache and `--edit`

The command line tool stores plot configurations in a JSON cache. The cache is kept outside the project and is separated by analysis path so that configurations from different datasets do not interfere with each other.

The cache is intended to make it easy to reproduce or modify a previous plot.

### Edit the most recent cache entry

```bash
python src/extract_histograms.py --edit
```

The tool displays the available entries and asks you to select one.

### Edit a specific cache entry

```bash
python src/extract_histograms.py --edit 0
```

Cache index `0` is the first entry shown by the cache loader.

### Override cached values

Explicit plotting arguments override the values loaded from the cache. For example:

```bash
python src/extract_histograms.py \
    --edit 0 \
    --title "Updated title" \
    --no-stats
```

This restores the cached configuration first, then applies the explicitly supplied CLI overrides.

### Use a different ROOT file with a cached configuration

You can provide new run/type/comment information together with `--edit`:

```bash
python src/extract_histograms.py \
    --edit 0 \
    --run 511 \
    --type physics
```

The cached plotting configuration is reused, while the ROOT file path is rebuilt from the new run/type/comment information.

If a new ROOT file is supplied together with `--path`, the provided paths take precedence over the cached histogram keys.

If a new ROOT file is supplied without `--path`, the cached histogram keys are reused. This is useful when the ROOT object structure is unchanged between runs.

## Output files

New plots are written below:

```text
<anapath>/output/<type>/
```

For example:

```text
/data/testbeam/output/physics/run_510-physics-Muon_time_spectrum.svg
```

The filename is generated from the run, type, optional comment, and plot title. Characters that are not suitable for filenames are replaced with `_`.

SVG output is created with:

- 300 dpi configuration for rasterized components
- `bbox_inches="tight"`
- additional plot artists required by the plotting implementation

Non-rasterized outputs are passed through the existing SVG optimization step afterwards.

The CLI also attempts to open the generated SVG automatically using the platform's default application:

- Windows: `os.startfile`
- macOS: `open`
- Linux/other Unix-like systems: `xdg-open`

Failure to open the file automatically does not invalidate the generated plot.

## Using the tool from Python

The main reason for the refactor is that plotting can now be used without invoking the CLI.

### Minimal example

```python
from root_extractor import PlotConfig, plot_from_root

config = PlotConfig()

fig = plot_from_root(
    "histograms_ana_510.root",
    paths=[["EventLoader", "layer_0", "corr"]],
    config=config,
)

fig.show()
```

The core function returns a real `matplotlib.figure.Figure` object. The caller decides what to do with it.

For example:

```python
fig.savefig("histogram.svg", bbox_inches="tight")
```

or in a notebook:

```python
fig
```

or:

```python
import matplotlib.pyplot as plt

plt.show()
```

No CLI arguments, cache files, output directories, SVG optimization, or external viewer are involved in the core API.

## Configuring plots from Python

`PlotConfig` replaces the old global `config_parameters` dictionary.

### Create and override configuration

```python
from root_extractor import PlotConfig

config = PlotConfig({
    "title": "Muon spectrum",
    "legend": True,
    "stats": True,
    "errors": True,
    "x-log": False,
    "y-log": True,
    "thickness": 2.0,
})
```

Individual values can also be changed later:

```python
config.set("legend", False)
config.set("thickness", 1.5)
```

Multiple values can be updated together:

```python
config.update({
    "legend": True,
    "errors": True,
    "alphas": [1.0, 0.75],
})
```

Read values with:

```python
legend = config.get("legend", False)
```

### Resetting configuration

```python
config.reset()
```

Resetting intentionally sets values to `None`. When histograms are subsequently passed through `plot_from_root`, the normal histogram-dependent default calculation fills the values again.

### Unknown / additional options

`PlotConfig` accepts unknown keys as well. This is intentional because some plotting options are already supported by lower-level code even though they are not part of the original default dictionary.

For example:

```python
config.set("marker", "o")
config.set("linestyle", "--")
```

This preserves the flexibility of the existing plotting code while keeping configuration state local to the object.

## Selecting histograms from Python

There are two ways to select ROOT objects.

### Keyword-based selection

```python
fig = plot_from_root(
    "histograms_ana_510.root",
    paths=[
        ["EventLoader", "layer_0", "corr"],
    ],
    config=config,
)
```

This uses the same matching mechanism as the CLI. Every keyword in a path must occur in the ROOT key, matching is case-insensitive, and only supported ROOT object classes are considered.

Multiple selections can be supplied:

```python
fig = plot_from_root(
    "histograms_ana_510.root",
    paths=[
        ["EventLoader", "layer_0", "corr"],
        ["Trigger", "time"],
    ],
    config=config,
)
```

### Explicit ROOT keys

If the exact ROOT object keys are already known, pass them directly:

```python
fig = plot_from_root(
    "histograms_ana_510.root",
    histogram_keys=[
        "EventLoader/layer_0/corr;1",
    ],
    config=config,
)
```

When `histogram_keys` is supplied, keyword-based `paths` are ignored.

This is particularly useful for applications that have their own database, cache, GUI selection model, or configuration system.

## Working with notebooks

A typical notebook workflow can be kept very small:

```python
from root_extractor import PlotConfig, plot_from_root

config = PlotConfig({
    "title": "Correlation",
    "stats": True,
    "legend": True,
})

fig = plot_from_root(
    "/data/testbeam/physics/histograms_ana_510.root",
    paths=[["EventLoader", "layer_0", "corr"]],
    config=config,
)

fig
```

Because the core API returns a normal Matplotlib figure, it can be combined with the usual Matplotlib notebook workflow. The core API also keeps its Matplotlib `rcParams` changes local to the call instead of permanently modifying the notebook's global plotting configuration.

## Error handling in Python

The core API intentionally raises exceptions instead of printing CLI-style errors and returning `None`.

Typical exceptions are:

- `FileNotFoundError` when the ROOT file does not exist
- `ValueError` when no histogram selection is supplied or no objects match
- `KeyError` when an explicitly requested ROOT key is missing
- `RuntimeError` when the plotting layer cannot create a figure

Example:

```python
try:
    fig = plot_from_root(
        "histograms_ana_510.root",
        paths=[["EventLoader", "layer_0", "corr"]],
    )
except FileNotFoundError:
    print("ROOT file does not exist")
except ValueError as exc:
    print(f"Invalid histogram selection: {exc}")
except KeyError as exc:
    print(f"ROOT object not found: {exc}")
```

The CLI wrapper catches these exceptions and converts them into the existing terminal-oriented error messages.

## API design

The refactored application is intentionally divided into four layers.

```text
PlotConfig
    ↓
core.plot_from_root()
    ↓
plotting.py
    ↓
matplotlib.Figure
```

The command line wrapper sits outside this flow:

```text
CLI arguments
    ↓
extract_histograms.py
    ├── cache
    ├── output path
    ├── SVG save
    └── external file opening
    ↓
plot_from_root()
```

This means that code which only needs a figure does not need to know anything about the command line interface or the cache.

## Backwards compatibility and migration

The old implementation stored configuration in the module-level `config_parameters` dictionary and used a global `anapath`. These global states have been removed.

New code should use:

```python
config = PlotConfig(...)
fig = plot_from_root(..., config=config)
```

rather than importing configuration helpers from `utils.py`.

The command line interface remains responsible for preserving the user-facing behavior, including the cache and interactive configuration workflow.

## Development

Run a syntax check over all Python files with:

```bash
python -m compileall src
```

For a local editable installation:

```bash
pip install -e .
```

Then test the public import:

```bash
python -c "from root_extractor import PlotConfig, plot_from_root; print(PlotConfig, plot_from_root)"
```

A real ROOT-file test requires a ROOT file containing one of the supported histogram/object classes.

## Current limitations

Some behavior is intentionally inherited from the original tool rather than redesigned during the refactor. In particular:

- TH2 stacking is not implemented.
- Direct TF1 plotting is limited and the existing code recommends known model overlays where appropriate.
- The CLI still contains compatibility code for running directly from a source checkout.
- Runtime dependencies are not yet listed under `[project.dependencies]` in `pyproject.toml`.
- Some lower-level plotting options exist in the implementation without being exposed as dedicated CLI flags.

These are separate concerns from the architectural refactor and can be addressed independently.
