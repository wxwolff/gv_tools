"""Convenience notebook interface for GV Tools 2DVD capabilities.

Typical use::

    from gv_tools import py_2dvd as dvd
    raw_data = dvd.ingest_raw(DATAFILE)
    parameters, dsd = dvd.calculate_products(raw_data)
    rain_figure = dvd.plot_integral_parameters(parameters)
    dsd_figure = dvd.plot_dsd(dsd)

Reading returns a DataFrame. Calculation returns two Xarray Datasets containing
both measured and terminal velocity choices. There is only one literature
implementation; velocity selection occurs when plotting or selecting a Dataset.
Saving data is explicit through save_products. All physical equations and
reference attribution live in core/py_2dvd.py and docs/py_2dvd/methods.md, not in wrappers.

Plot imports are deferred until called: ingestion/calculation alone does not
initialize a graphical backend or change a notebook's plotting configuration.
"""

from .core import calculate_dsd, calculate_integral_parameters, calculate_products, prepare_drops
from .io import ingest_raw, save_products

from . import __version__


def plot_integral_parameters(parameters, **options):
    """Return an integral Figure with eight panels (linear reflectivity omitted).

    Parameters
    ----------
    parameters : xarray.Dataset
        Output of calculate_integral_parameters or the first calculate_products
        result. Keep its time/velocity coordinates, output_valid and attributes.
    **options
        Forwarded unchanged to graph.plot_integral_parameters. Common choices:
        velocity='measured', output_dir=None, figsize=(14, 14), colors={},
        ylimits={}, time_range=None, valid_only=True, linewidth=1.5, dpi=150,
        rc_params={}, show=False. Colors/limits key by physical variable name.

    Returns
    -------
    matplotlib.figure.Figure
        Editable figure; optional output_dir also saves a dated PNG. The function
        never recomputes parameters or edits the input Dataset. See docs/api.md.
    """
    from .graph import plot_integral_parameters as plot

    return plot(parameters, **options)


def plot_dsd(dsd, **options):
    """Return a logarithmic concentration Figure from a calculated DSD Dataset.

    Parameters
    ----------
    dsd : xarray.Dataset
        Output of calculate_dsd or the second calculate_products result.
    **options
        Forwarded unchanged to graph.plot_dsd. Common choices: velocity='measured',
        output_dir=None, diameter_range=(0, 5), concentration_range=(1, 10000),
        time_range=None, valid_only=True, figsize=(12, 6), cmap='dsd', dpi=150,
        rc_params={}, show=False. Concentration limits are linear physical values
        in m^-3 mm^-1; the plotting function applies the logarithmic color scale.

    Returns
    -------
    matplotlib.figure.Figure
        Editable figure; optional output_dir also saves a dated PNG. Raw counts
        cannot be substituted for the normalized dsd variable. See docs/plotting.md.
    """
    from .graph import plot_dsd as plot

    return plot(dsd, **options)


__all__ = [
    "prepare_drops",
    "ingest_raw",
    "save_products",
    "calculate_products",
    "calculate_integral_parameters",
    "calculate_dsd",
    "plot_integral_parameters",
    "plot_dsd",
]
