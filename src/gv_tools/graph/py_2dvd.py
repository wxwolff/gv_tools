"""Readable plots of literature products, with no scientific recalculation.

Integral parameters are displayed in eight panels; linear reflectivity remains
in the data but is intentionally omitted from the figure. The DSD displays
logarithmic concentration, not counts. Date, velocity choice, and method are
always named. Plot styling follows the earlier Parsivel-inspired workflow.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from ..core.py_2dvd import VELOCITIES

DSD_COLOR_STOPS = (
    "#f7fbff",
    "#9ecae1",
    "#3182bd",
    "#31a354",
    "#addd8e",
    "#ffff33",
    "#fd8d3c",
    "#e31a1c",
    "#6a3d9a",
)

DEFAULT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
}
INTEGRAL_PANELS = (
    ("rain_rate",),
    ("reflectivity_dbz",),
    ("liquid_water_content",),
    ("mass_weighted_diameter", "maximum_diameter"),
    ("drop_count",),
    ("number_concentration",),
    ("diameter_std",),
    ("minimum_diameter",),
)
PARAMETER_LABELS = {
    "rain_rate": "Rain rate",
    "reflectivity_dbz": "Reflectivity",
    "liquid_water_content": "Liquid water content",
    "mass_weighted_diameter": "Dm",
    "maximum_diameter": "DMax",
    "drop_count": "Drop count",
    "number_concentration": "Number concentration",
    "diameter_std": "Mass-spectrum spread",
    "minimum_diameter": "DMin",
}


def _heading(product, velocity):
    """Validate the chosen branch and label site/date/velocity without ambiguity.

    Method is always literature. Keeping this heading automatic prevents a saved
    terminal-velocity plot from being mislabeled as a measured-velocity result.
    """
    if velocity not in VELOCITIES:
        raise ValueError("velocity must be 'measured' or 'terminal'")
    return (
        f"{product.attrs['site']} / {product.attrs['instrument_id']} | {product.attrs['date']} | 2DVD\n"
        f"Literature — {velocity} velocity"
    )


def _time_limits(product, time_range):
    """Default to the raw-record window, not only the qualifying rain minutes."""
    if time_range is not None:
        start, end = map(pd.Timestamp, time_range)
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise ValueError("time_range must contain increasing valid timestamps")
        return start, end
    recorded = product.time.values[product.raw_drop_count.values > 0]
    if not len(recorded):
        recorded = product.time.values
    return pd.Timestamp(recorded[0]), pd.Timestamp(recorded[-1]) + pd.Timedelta(
        minutes=1
    )


def _save_figure(figure, product, output_dir, kind, suffix, dpi, velocity):
    """Write the agreed dated PNG name only when saving was explicitly requested.

    output_dir is the root above Plots, not a filename. A second call for the
    same site/day/kind/velocity replaces that PNG. Different velocity choices
    use separate filenames. bbox_inches='tight' includes titles and outer labels.
    """
    if output_dir is None:
        return
    date = pd.Timestamp(product.attrs["date"])
    folder = Path(output_dir).expanduser() / "Plots" / kind / f"{date:%Y/%m}"
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{product.attrs['site']}_{date:%Y_%m%d}_2DVD_{velocity}_velocity_{suffix}.png"
    figure.savefig(folder / filename, dpi=dpi, bbox_inches="tight")


def plot_integral_parameters(
    parameters,
    *,
    velocity="measured",
    output_dir=None,
    figsize=(14, 14),
    colors=None,
    ylimits=None,
    time_range=None,
    valid_only=True,
    linewidth=1.5,
    dpi=150,
    rc_params=None,
    show=False,
):
    """Return an eight-panel Figure, excluding linear reflectivity.

    colors and ylimits are dictionaries keyed by physical parameter name, e.g.
    colors={'mass_weighted_diameter': 'navy', 'maximum_diameter': 'darkorange'}.
    rc_params overrides only this call's Matplotlib style. Optional output_dir
    saves Plots/Rain/YYYY/MM/SITE_YYYY_MMDD_2DVD_measured_velocity_rain.png. Each velocity choice
    has a separate filename. valid_only masks the documented reporting
    threshold; it does not alter the stored scientific results.
    """
    heading = _heading(parameters, velocity)
    palette = {"mass_weighted_diameter": "#0072B2", "maximum_diameter": "#D55E00"}
    palette.update(colors or {})
    # Scoped style defaults keep fonts readable without changing the caller's
    # global Matplotlib rcParams; explicit user values take precedence.
    with mpl.rc_context({**DEFAULT_STYLE, **(rc_params or {})}):
        figure, axes = plt.subplots(
            4, 2, figsize=figsize, sharex=True, layout="constrained"
        )
        for axis, group in zip(axes.flat, INTEGRAL_PANELS):
            for parameter in group:
                values = parameters[parameter].sel(velocity=velocity)
                # Missing values break the line at invalid minutes rather than
                # joining points across periods below the reporting threshold.
                if valid_only:
                    values = values.where(parameters.output_valid.astype(bool))
                axis.plot(
                    values.time,
                    values,
                    color=palette.get(parameter, "black"),
                    linewidth=linewidth,
                    label=PARAMETER_LABELS[parameter],
                )
            axis.set_title(" / ".join(PARAMETER_LABELS[name] for name in group))
            axis.set_ylabel(parameters[group[0]].attrs["units"])
            if group[0] in (ylimits or {}):
                axis.set_ylim(*ylimits[group[0]])
            axis.set_xlim(*_time_limits(parameters, time_range))
            axis.grid(alpha=0.25)
            if len(group) > 1:
                axis.legend(loc="upper left", ncols=2)
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for axis in axes[-1, :]:
            axis.set_xlabel("Time (source timezone unspecified)")
            axis.tick_params(axis="x", labelrotation=30)
        figure.suptitle(heading)
        _save_figure(figure, parameters, output_dir, "Rain", "rain", dpi, velocity)
        if show:
            plt.show()
    return figure


def plot_dsd(
    dsd,
    *,
    velocity="measured",
    output_dir=None,
    diameter_range=(0, 5),
    concentration_range=(1, 10000),
    time_range=None,
    valid_only=True,
    figsize=(12, 6),
    cmap="dsd",
    dpi=150,
    rc_params=None,
    show=False,
):
    """Return a time/diameter DSD Figure with concentration in m^-3 mm^-1.

    Each cell spans an actual minute and its stored diameter edges. Nonpositive
    concentrations and (by default) invalid minutes are masked before applying
    LogNorm. The colorbar is a concentration scale; it is not a logarithm stored
    in the Dataset. Saving is explicit through output_dir; the function never
    calculates a replacement spectrum from plotted integral parameters.
    """
    heading = _heading(dsd, velocity)
    minimum_concentration, maximum_concentration = concentration_range
    if not (0 < minimum_concentration < maximum_concentration < np.inf):
        raise ValueError("concentration_range must be increasing, finite and positive")
    if not (0 <= diameter_range[0] < diameter_range[1] < np.inf):
        raise ValueError("diameter_range must be increasing, finite and nonnegative")
    values = dsd.dsd.sel(velocity=velocity)
    if valid_only:
        values = values.where(dsd.output_valid.astype(bool))
    # pcolormesh expects rows=diameter, columns=time, the transpose of the stored
    # time-by-diameter slice. Mask before LogNorm: log(0) is not a plotted value.
    concentration = np.ma.masked_invalid(values.values.T)
    concentration = np.ma.masked_less_equal(concentration, 0)
    minute_times = pd.DatetimeIndex(dsd.time.values)
    # One more edge than centers is required to draw each complete minute cell.
    time_edges = minute_times.append(
        pd.DatetimeIndex([minute_times[-1] + pd.Timedelta(minutes=1)])
    )
    diameter_edges = np.r_[dsd.bin_lower_mm.values, dsd.bin_upper_mm.values[-1]]
    # Scoped style defaults keep fonts readable without changing the caller's
    # global Matplotlib rcParams; explicit user values take precedence.
    with mpl.rc_context({**DEFAULT_STYLE, **(rc_params or {})}):
        figure, axis = plt.subplots(figsize=figsize, layout="constrained")
        palette = (
            LinearSegmentedColormap.from_list("lite_dsd", DSD_COLOR_STOPS)
            if cmap == "dsd"
            else plt.get_cmap(cmap)
        ).with_extremes(bad="white")
        mesh = axis.pcolormesh(
            time_edges,
            diameter_edges,
            concentration,
            shading="flat",
            cmap=palette,
            norm=LogNorm(minimum_concentration, maximum_concentration),
            rasterized=True,
        )
        colorbar = figure.colorbar(mesh, ax=axis, orientation="horizontal", pad=0.12)
        colorbar.set_label(r"$N(D)$ [m$^{-3}$ mm$^{-1}$]")
        axis.set(
            title=heading,
            ylabel="Drop diameter [mm]",
            xlabel="Time (source timezone unspecified)",
        )
        axis.set_ylim(*diameter_range)
        axis.set_xlim(*_time_limits(dsd, time_range))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        _save_figure(figure, dsd, output_dir, "DSD", "DSD", dpi, velocity)
        if show:
            plt.show()
    return figure
