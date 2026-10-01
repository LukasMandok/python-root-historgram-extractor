import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Dict, Any, Optional, Union, Tuple
import os

# --- Cache Configuration ---
CACHE_FILE = "histograms_cache.json"
COMMON_CACHE_BASE_DIR = os.path.expanduser("~/.cache/python-root-histogram-extractor")

allowed_classes: List[str] = ["TH1", "TH2", "TH3", "TGraph", "TF1"]

# --- Color Palettes ---
color0 = ["#0EA5E9", "#EF4444", "#F7DA47", "#22C55E", "#F79646", "#7E8BD6", "#0369A1"]
color1 = ["#118ab2","#ef476f","#ffd166","#06d6a0","#ff9f1c","#7d5ba6","#073b4c"]
color2 = ["#5AA9E6","#e76f51","#e9c46a","#2a9d8f","#f4a261","#7f5a85","#264653"]
color3 = ["#1982c4","#ff595e","#ffca3a","#8ac926","#ff8c42","#6a4c93","#555555"]
color4 = ["#247ba0","#f25f5c","#ffe066","#70c1b3","#f4975c","#b392ac","#50514f"]
color5 = ['#00E5E5','#FC4902','#FFE74F','#58F37E','#FF8C42','#9C3587','#00486D']
color6 = ['#3AD8FE','#FE7562','#FDD835','#52CE9C','#FF914D','#7E8BD6',"#073b4c"]

color_palettes: Dict[str, List[str]] = {
    "color0": color0,
    "color1": color1,
    "color2": color2,
    "color3": color3,
    "color4": color4,
    "color5": color5,
    "color6": color6
}

color_list: Dict[str, int] = {"blue":0, "red":1, "yellow":2, "green":3, "orange":4, "purple":5, "grey":6}

# --- Colormaps ---
viridis = plt.get_cmap("viridis")
# Sample the viridis colormap skipping the bottom 18%.
viridis_red = ListedColormap(viridis(np.linspace(0.18, 1.0, 256)), name="viridis_red")
if "viridis_red" not in plt.colormaps:
    plt.colormaps.register(cmap=viridis_red)

viridis_inv = ListedColormap(viridis(np.linspace(1.0, 0.18, 256)), name="viridis_inv")
if "viridis_inv" not in plt.colormaps:
    plt.colormaps.register(cmap=viridis_inv)

color_maps = ["viridis_red", "plasma", "inferno", "viridis", "viridis_inv", "cool", "summer", "YlGnBu"]

# --- Type Hint for Histograms ---
HistogramType = Union[
    uproot.behaviors.TH1.TH1,
    uproot.behaviors.TH2.TH2,
    uproot.behaviors.TH3.TH3,
    uproot.behaviors.TGraph.TGraph
]

# --- Terminal Colors ---
reset, red, blue, green = "\033[0m", "\033[91m", "\033[94m", "\033[92m"