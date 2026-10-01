"""Configuration objects for ROOT histogram plotting.

This module contains the configuration state that used to live in the
module-level ``config_parameters`` dictionary.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple


@dataclass
class ConfigParameter:
    """Value and metadata associated with one plotting option."""

    value: Any
    used: bool
    hidden: bool


class PlotConfig:
    """Mutable plotting configuration.

    ``PlotConfig`` intentionally preserves the semantics of the former
    ``config_parameters`` dictionary:

    * every option has a value, a ``used`` flag and a ``hidden`` flag;
    * an option can be updated without changing its metadata;
    * setting a value to ``None`` marks it for default resolution later;
    * histogram-dependent defaults are still resolved by ``refresh_config``
      in the current implementation. That behavior belongs to a later
      refactoring step and is therefore not duplicated here.

    The object is deliberately independent of ROOT, matplotlib and the CLI.
    This makes it safe to instantiate and pass around in notebooks or other
    applications.
    """

    DEFAULT_PARAMETERS: Dict[str, Tuple[Any, bool, bool]] = {
        # key: (initial value, used by default, hidden in the interactive UI)
        "title": (None, True, False),
        "x-label": (None, True, False),
        "y-label": (None, True, False),
        "x-multiplier": (1.0, True, False),
        "y-multiplier": (1.0, True, False),
        "z-label": (None, False, False),
        "x-log": (False, True, True),
        "y-log": (False, True, True),
        "z-log": (False, False, True),
        "limits": (None, True, False),
        "bin_width": (None, True, False),
        "legend": (None, True, False),
        "stats": (False, True, False),
        "grid": (None, True, True),
        "names": (None, False, False),
        "colors": (None, True, False),
        "palette": (None, False, True),
        "alphas": (None, True, True),
        "angles": (None, False, True),
        "thickness": (None, True, True),
        "lines": (False, False, True),
        "cutoff": (False, False, True),
        "flat": (False, False, True),
        "raster": (None, False, True),
        "errors": (False, False, True),
        "fits": (False, True, False),
        "models": ([], False, True),
        "model-params": ({}, False, True),
        "model-stats": (True, False, False),
        "figsize": ((8, 6), False, True),
        "textsize": (12.0, False, True),
        "background": ("white", True, False),
    }

    def __init__(
        self,
        values: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Create a configuration using the existing defaults.

        ``values`` contains explicit overrides. Unknown keys are accepted so
        that currently supported optional plotting settings such as
        ``marker`` or ``linestyle`` can be added without changing this class.
        """
        self._parameters: Dict[str, ConfigParameter] = {
            key: ConfigParameter(
                value=deepcopy(default_value),
                used=used,
                hidden=hidden,
            )
            for key, (default_value, used, hidden) in self.DEFAULT_PARAMETERS.items()
        }

        if values:
            self.update(values)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a configuration value, or ``default`` for an unknown key."""
        parameter = self._parameters.get(key)
        if parameter is None:
            return default
        return parameter.value

    def set(
        self,
        key: str,
        value: Any,
        used: Optional[bool] = None,
        hidden: Optional[bool] = None,
        *,
        hide: Optional[bool] = None,
    ) -> None:
        """Set a configuration value while preserving unspecified metadata.

        ``hide`` is accepted as an alias for ``hidden`` to make migration
        from the old ``set_config(..., hide=...)`` API straightforward.
        Supplying both ``hidden`` and ``hide`` with different values raises a
        ``ValueError`` rather than silently choosing one.
        """
        if hidden is not None and hide is not None and hidden != hide:
            raise ValueError("'hidden' and 'hide' specify conflicting values")

        if hidden is None:
            hidden = hide

        current = self._parameters.get(key)
        if current is None:
            self._parameters[key] = ConfigParameter(
                value=value,
                used=True if used is None else used,
                hidden=False if hidden is None else hidden,
            )
            return

        current.value = value
        if used is not None:
            current.used = used
        if hidden is not None:
            current.hidden = hidden

    def update(
        self,
        values: Mapping[str, Any],
        *,
        used: Optional[bool] = None,
        hidden: Optional[bool] = None,
    ) -> None:
        """Update several configuration values at once."""
        for key, value in values.items():
            self.set(key, value, used=used, hidden=hidden)

    def reset(self) -> None:
        """Reset all configured values to unresolved defaults.

        This mirrors the existing interactive reset behavior, where values
        are set to ``None`` and subsequently resolved from histogram data.
        The parameter metadata is intentionally retained.
        """
        for parameter in self._parameters.values():
            parameter.value = None

    def get_parameter(self, key: str) -> Optional[ConfigParameter]:
        """Return the complete parameter record for a key, if it exists."""
        return self._parameters.get(key)

    def keys(self) -> Iterator[str]:
        """Iterate over configured parameter names."""
        return iter(self._parameters)

    def items(self) -> Iterator[Tuple[str, ConfigParameter]]:
        """Iterate over parameter names and their metadata records."""
        return iter(self._parameters.items())

    def __contains__(self, key: object) -> bool:
        return key in self._parameters

    def __len__(self) -> int:
        return len(self._parameters)
