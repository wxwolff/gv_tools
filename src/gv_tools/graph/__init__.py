"""Plotting and quicklook generation."""

from ..outputs import plot_directory
from ..quicklooks import (
    plot_aio_quicklook,
    plot_aio_wind_rose,
    plot_parsivel_quicklook,
    plot_radar_ppi_quicklook,
    plot_radar_rhi_quicklook,
    plot_radar_bb_zdr_calibration,
    plot_mrr_time_height_quicklook,
    plot_ws800_full_quicklook,
)

__all__ = [
    "plot_aio_quicklook",
    "plot_aio_wind_rose",
    "plot_parsivel_quicklook",
    "plot_radar_ppi_quicklook",
    "plot_radar_rhi_quicklook",
    "plot_radar_bb_zdr_calibration",
    "plot_mrr_time_height_quicklook",
    "plot_ws800_full_quicklook",
    "plot_directory",
]

# Defer Matplotlib imports until a 2DVD plotting function is requested.
__all__ += ["plot_integral_parameters", "plot_dsd"]

def __getattr__(name):
    if name not in ("plot_integral_parameters", "plot_dsd"):
        raise AttributeError(name)
    from importlib import import_module
    value = getattr(import_module(".py_2dvd", __name__), name)
    globals()[name] = value
    return value
