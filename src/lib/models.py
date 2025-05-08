import numpy as np
import inspect
import matplotlib.pyplot as plt
from scipy.special import erf
from typing import Callable, Dict, Any, Tuple, List, Optional

# --- Basic Model Definitions ---

def erf_diff_model(x, N, mu, sigma, a):
    """Difference of two error functions."""
    sqrt2 = np.sqrt(2.0)
    term1 = erf((x - mu + a/2) / (sigma * sqrt2))
    term2 = erf((x - mu - a/2) / (sigma * sqrt2))
    return N/2 * (term1 - term2)

def lifetime_model(t, B0, N0, tau):
    """Muon lifetime decay function."""
    return B0 + N0 * np.exp(-t/tau)

def lifetime_oscillation_model(t, B0, N0, tau, A, sigma, omega, phi):
    """Muon lifetime with oscillation."""
    decay = B0 + N0 * np.exp(-t/tau)
    damping = np.exp(-(sigma*t)**2/2)
    oscillation = 1 + A * damping * np.cos(omega*t + phi)
    return decay * oscillation

def asymmetry_model(t, N0, omega, phi):
    """Simple oscillation model."""
    oscillation = np.cos(omega*t + phi)
    return N0 * oscillation

# --- Dictionary of Available Models ---
model_functions: Dict[str, Callable] = {
    "erf_diff": erf_diff_model,
    "lifetime": lifetime_model,
    "lifetime_oscillation": lifetime_oscillation_model,
    "asymmetry": asymmetry_model
}

# --- Predefined Model Functions with Preset Parameters ---

def gps_lifetime_norm_old(t, B0=0.04502, N0=0.8633, tau=2172, A=0.2441, sigma=0.00029, omega=0.005415, phi=0.894):
    return lifetime_oscillation_model(t, B0, N0, tau, A, sigma, omega, phi)

def gps_lifetime_norm(t, B0=0.05508, N0=1.056, tau=2172, A=0.2441, sigma=0.00029, omega=0.005415, phi=0.894):
    return lifetime_oscillation_model(t, B0, N0, tau, A, sigma, omega, phi)

def gps_lifetime_v2(t, B0=20, N0=1000, tau=2200, A=0.3, sigma=0.2, omega=2*np.pi*13.9, phi=0):
    return lifetime_oscillation_model(t, B0, N0, tau, A, sigma, omega, phi)

def upstream_lifetime_norm_old(t, B0=0.00367, N0=0.9075, tau=2218, A=0.2396, sigma=0.00027, omega=2*np.pi*0.000865, phi=-2):
    return lifetime_oscillation_model(t, B0, N0, tau, A, sigma, omega, phi)

def upstream_lifetime_norm(t, B0=0.003977, N0=0.998, tau=2218, A=0.2396, sigma=0.00027, omega=2*np.pi*0.000865, phi=-2):
    return lifetime_oscillation_model(t, B0, N0, tau, A, sigma, omega, phi)

def upstream_lifetime_v2(t, B0=3.67, N0=958, tau=2218, A=0.240, sigma=0.00027, omega=2*np.pi*0.000865, phi=-2):
    return lifetime_oscillation_model(t, B0, N0, tau, A, sigma, omega, phi)

def downstream_lifetime_norm(t, B0=0.094, N0=0.972, tau=2238, A=0.2436, sigma=0.0002915, omega=0.005405, phi=-4.74):
    return lifetime_oscillation_model(t, B0, N0, tau, A, sigma, omega, phi)

def mu_decay(t, B0=2.5, N0=1706, tau=2088):
    return lifetime_model(t, B0, N0, tau)

def feature_1(x, N=0.3203, mu=3.195, sigma=0.5209, a=0.6):
    return erf_diff_model(x, N, mu, sigma, a)

def feature_2(x, N=0.1693, mu=3.744, sigma=0.6494, a=1.1):
    return erf_diff_model(x, N, mu, sigma, a)

def feature_3(x, N=0.1208, mu=4.05, sigma=0.4585, a=1.6):
    return erf_diff_model(x, N, mu, sigma, a)

# Add predefined models to the available models dictionary
model_functions.update({
    'f1': feature_1,
    'f2': feature_2,
    'f3': feature_3,
    "gps_lifetime": gps_lifetime_norm,
    "gps_lifetime_v2": gps_lifetime_v2,
    "upstream_lifetime": upstream_lifetime_norm,
    "upstream_lifetime_v2": upstream_lifetime_v2,
    "downstream_lifetime": downstream_lifetime_norm,
    "mu_decay": mu_decay
})

# --- Default Model Statistics ---
model_statistics = {
    "gps_lifetime": {
        "stats": {
            "params": {"B0": 0.05508, "N0": 1.056, "tau": 2172, "A": 0.2441, "sigma": 0.00029, "omega": 0.005415, "phi": 0.894},
            "errors": {"B0": 0.00012, "N0": 0.0085, "tau": 12, "A": 0.0048, "sigma": 0.000015, "omega": 0.000021, "phi": 0.042},
            "chi2": 105.6,
            "ndf": 98
        }
    },
    "upstream_lifetime": {
        "stats": {
            "params": {"B0": 0.003977, "N0": 0.998, "tau": 2218, "A": 0.2396, "sigma": 0.00027, "omega": 2*np.pi*0.000865, "phi": -2.0},
            "errors": {"B0": 0.00008, "N0": 0.0076, "tau": 14, "A": 0.0052, "sigma": 0.000018, "omega": 0.000019, "phi": 0.057},
            "chi2": 112.8,
            "ndf": 103
        }
    },
    "f1": {
        "stats": {
            "params": {"N": 0.3203, "mu": 3.195, "sigma": 0.5209, "a": 0.6},
            "errors": {"N": 0.0052, "mu": 0.042, "sigma": 0.0163, "a": 0.021},
            "chi2": 24.3,
            "ndf": 21
        }
    },
    "f2": {
        "stats": {
            "params": {"N": 0.1693, "mu": 3.744, "sigma": 0.6494, "a": 1.1},
            "errors": {"N": 0.0048, "mu": 0.057, "sigma": 0.0197, "a": 0.038},
            "chi2": 19.8,
            "ndf": 18
        }
    },
    "f3": {
        "stats": {
            "params": {"N": 0.1208, "mu": 4.05, "sigma": 0.4585, "a": 1.6},
            "errors": {"N": 0.0042, "mu": 0.063, "sigma": 0.0214, "a": 0.046},
            "chi2": 17.5,
            "ndf": 16
        }
    }
}

# --- Model Plotting Helpers ---

def quick_plot_model(ax, model_name, x_range=(0, 10, 1000), **kwargs):
    """Plots a predefined or basic model using default or provided parameters."""
    if model_name not in model_functions:
        print(f"Unknown model: {model_name}. Available models: {list(model_functions.keys())}")
        return None # Return None or raise error

    model_func = model_functions[model_name]
    sig = inspect.signature(model_func)
    default_params = {
        p.name: p.default
        for p in sig.parameters.values()
        if p.default is not inspect.Parameter.empty and p.name != 't' and p.name != 'x'
    }

    # Combine defaults, provided kwargs, and potentially stats from model_statistics
    params = {**default_params, **kwargs}
    if "stats" not in params and model_name in model_statistics:
        # Ensure we don't overwrite existing params with the ones from stats dict
        stats_params = model_statistics[model_name].get("stats", {}).get("params", {})
        params = {**stats_params, **params} # Provided kwargs take precedence
        # Add the full stats dict back if it existed
        if "stats" in model_statistics[model_name]:
             params["stats"] = model_statistics[model_name]["stats"]


    param_names = [p.name for p in sig.parameters.values() if p.name != 't' and p.name != 'x']
    # Ensure all required parameters are present
    param_values = []
    for name in param_names:
        if name in params:
            param_values.append(params[name])
        elif name in default_params:
             param_values.append(default_params[name])
        else:
            print(f"Error: Missing parameter '{name}' for model '{model_name}'")
            return None # Or raise error

    x = np.linspace(*x_range)
    try:
        y = model_func(x, *param_values)
    except Exception as e:
        print(f"Error evaluating model '{model_name}' with params {param_values}: {e}")
        return None

    label = kwargs.get('label', f"{model_name} model") # Default label
    color = kwargs.get('color', 'blue')
    alpha = kwargs.get('alpha', 1.0)
    zorder = kwargs.get('zorder', 10) # Ensure models plot on top

    line, = ax.plot(x, y, color=color, label=label, alpha=alpha, zorder=zorder)
    return line


def plot_model(ax, model_func: Callable, x_range: Tuple[float, float, int], params: List[float],
               color: str = 'blue', label: Optional[str] = None, alpha: float = 1.0):
    """Generic function to plot any model function."""
    x = np.linspace(*x_range)
    y = model_func(x, *params)
    ax.plot(x, y, color=color, label=label, alpha=alpha)
    return ax