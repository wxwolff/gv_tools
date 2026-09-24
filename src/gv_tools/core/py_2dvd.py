"""Literature-based drop integration and diameter spectra in explicit units.

Only one physical implementation is present; measured and terminal refer to
velocity choices, not competing algorithms. Instrument context: Kruger &
Krajewski (2002). DSD/integral context: Tokay et al. (2001, 2013). Mass-weighted
mean and spread: Williams et al. (2014), DOI 10.1175/JAMC-D-13-076.1.

The equations below are explicit discrete sampling-volume sums. ``docs/py_2dvd/methods.md``
derives the unit conversions and distinguishes these equations from inherited
FORTRAN preprocessing conventions. No rainnasa integral implementation is used.
"""

import json

from .. import __version__
from datetime import datetime, timezone
from importlib.resources import files

import numpy as np
import pandas as pd
import xarray as xr

from ..io.py_2dvd import _file_hash

SAMPLE_SECONDS = 60.0
MINUTES_PER_DAY = 1440
WATER_DENSITY_KG_M3 = 1000.0
VELOCITIES = ("terminal", "measured")
INTEGRAL_UNITS = {
    "rain_rate": "mm h-1",
    "number_concentration": "m-3",
    "liquid_water_content": "g m-3",
    "reflectivity": "mm6 m-3",
    "reflectivity_dbz": "dBZ",
    "mass_weighted_diameter": "mm",
    "diameter_std": "mm",
    "minimum_diameter": "mm",
    "maximum_diameter": "mm",
    "drop_count": "1",
}
TABLE_NAMES = ("tervel.dat", "oblateness.txt", "2dvd_diameter020.txt")


def _lookup_table(name):
    """Read bundled numerical data, so users never need a --tables argument."""
    resource = files("gv_tools").joinpath("data", "py_2dvd", name)
    with resource.open("r") as stream:
        values = np.loadtxt(stream, ndmin=2)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite lookup table: {name}")
    return values


def _interpolate_three_nearest(diameters_mm, table_diameters_mm, table_values):
    """Reproduce intr.f's quadratic interpolation through three nearest points.

    This is not linear interpolation between neighboring rows. Stable distance
    sorting keeps the earlier table row in a tie; outside the tabulated range
    the same polynomial extrapolates. Blocks bound the temporary distance array.
    """
    interpolated_values = np.empty(len(diameters_mm), dtype=float)
    for start in range(0, len(diameters_mm), 10000):
        query_diameters = np.asarray(diameters_mm[start : start + 10000])
        nearest_indices = np.argsort(
            abs(query_diameters[:, None] - table_diameters_mm), axis=1, kind="stable"
        )[:, :3]
        abscissae = table_diameters_mm[nearest_indices]
        polynomial_values = table_values[nearest_indices].copy()
        # Neville's recurrence evaluates the quadratic without fitting global
        # coefficients, following the arithmetic order used by intr.f.
        for upper in range(1, 3):
            for lower in range(upper):
                polynomial_values[:, upper] = (
                    polynomial_values[:, lower]
                    * (query_diameters - abscissae[:, upper])
                    - polynomial_values[:, upper]
                    * (query_diameters - abscissae[:, lower])
                ) / (abscissae[:, lower] - abscissae[:, upper])
        interpolated_values[start : start + len(query_diameters)] = polynomial_values[
            :, 2
        ]
    return interpolated_values


def prepare_drops(raw_data):
    """Return a copy with derived diameter, terminal velocity and explicit QC.

    These inherited conventions preserve the full package's selected sample:

    * Diameter is (6 V / pi)**0.333, retaining dropbydrop.f's exponent rather
      than silently replacing it with an exact cube root.
    * Terminal velocity is interpolated from tervel.dat (cm/s -> m/s).
    * Strict velocity bounds use 0.5 and 1.5 times that interpolated value.
    * For reported diameter >= 6 mm, terminal velocity is then overridden by
      9.65 - 10.3 exp(-0.6 D) m/s. The QC bounds deliberately remain the earlier
      interpolation bounds, as in the source pipeline.
    * After the first backward hour/minute jump, remaining rows are rejected.
      Seconds alone do not trigger this legacy condition. Input is not sorted.
    * Derived diameter must be <= 10 mm. Axis ratio is retained diagnostically;
      the supplied oblateness table does not impose an additional filter.

    Neither raw_data nor its metadata is mutated. Values are not yet rounded.
    """
    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError("Expected the DataFrame returned by ingest_raw")
    if not {"site", "instrument_id", "date"}.issubset(raw_data.attrs):
        raise ValueError("Missing source metadata; read the file with ingest_raw first")
    drops = raw_data.copy(deep=True)
    velocity_table = _lookup_table("tervel.dat")[:60]
    axis_ratio_table = _lookup_table("oblateness.txt")[:69]
    drops["diameter_mm"] = (6 * drops.drop_volume_mm3 / np.pi) ** 0.333
    interpolated_velocity = (
        _interpolate_three_nearest(
            drops.diameter_mm.to_numpy(), velocity_table[:, 0], velocity_table[:, 1]
        )
        / 100.0
    )
    drops["expected_axis_ratio"] = _interpolate_three_nearest(
        drops.diameter_mm.to_numpy(), axis_ratio_table[:, 0], axis_ratio_table[:, 1]
    )
    drops["velocity_lower_m_s"] = 0.5 * interpolated_velocity
    drops["velocity_upper_m_s"] = 1.5 * interpolated_velocity
    large_drop = drops.reported_diameter_mm.to_numpy() >= 6.0
    interpolated_velocity[large_drop] = 9.65 - 10.3 * np.exp(
        -0.6 * drops.reported_diameter_mm.to_numpy()[large_drop]
    )
    drops["terminal_velocity_m_s"] = interpolated_velocity
    drops["velocity_ok"] = (drops.measured_velocity_m_s > drops.velocity_lower_m_s) & (
        drops.measured_velocity_m_s < drops.velocity_upper_m_s
    )
    clock_minutes = (drops.hour * 60 + drops.minute).to_numpy()
    if len(drops):
        backward_clock_seen = np.maximum.accumulate(
            np.r_[False, np.diff(clock_minutes) < 0]
        )
        drops["time_ok"] = ~backward_clock_seen
    else:
        drops["time_ok"] = pd.Series(dtype=bool)
    drops["accepted"] = drops.time_ok & drops.velocity_ok & (drops.diameter_mm <= 10.0)
    return drops


def _selected_physical_inputs(prepared_drops):
    """Select first, then reproduce the validated intermediate text precision.

    D, both velocities, and area were historically written with three decimals
    and read into single precision. Retaining that convention avoids changing
    drop/bin assignments during this cleanup. Subsequent integration is float64.
    This compatibility convention is NOT a claim that the papers require rounding.
    """
    selected = prepared_drops.loc[prepared_drops.accepted].copy()
    physical_columns = (
        "diameter_mm",
        "measured_velocity_m_s",
        "terminal_velocity_m_s",
        "sampling_area_mm2",
    )
    for column in physical_columns:
        rounded = np.array([float(f"{value:.3f}") for value in selected[column]])
        selected[column] = rounded.astype(np.float32).astype(float)
    values = selected[list(physical_columns)].to_numpy()
    if (
        not np.isfinite(values).all()
        or (values[:, 0] < 0).any()
        or (values[:, 1:] <= 0).any()
    ):
        raise ValueError(
            "Accepted drops require nonnegative diameter and positive area/velocities after rounding"
        )
    return selected


def _diameter_bins():
    """Return nominal centers/width and historical half-open membership edges.

    Nominal centers come from the first 50 rows of 2dvd_diameter020.txt. Membership
    edges retain repeated float32 additions of 0.2 mm (the final edge is slightly
    below 10 mm). Normalization uses exactly 0.2 mm, not the tiny irregular edge
    differences. An upper-edge drop belongs to the next bin; the final upper edge
    is excluded. Keeping these rules preserves the validated drop-count comparison.
    """
    bin_table = _lookup_table("2dvd_diameter020.txt")[:50]
    if (
        bin_table.shape[0] != 50
        or bin_table.shape[1] < 2
        or not np.allclose(bin_table[:, 0], np.arange(50) * 0.2 + 0.1)
        or not np.allclose(bin_table[:, 1], 0.2)
    ):
        raise ValueError("Expected 50 nominal 0.2-mm diameter bins")
    edges_mm = np.zeros(51, dtype=np.float32)
    for index in range(1, 51):
        edges_mm[index] = edges_mm[index - 1] + np.float32(0.2)
    return bin_table[:, 0], edges_mm.astype(float)


def calculate_products(raw_data):
    """Calculate integral parameters and DSD together; return two Xarray Datasets.

    For each selected drop j, A_j is its supplied effective area in m², v_j the
    chosen velocity in m/s, and T=60 s. Its concentration contribution is
    w_j = 1/(A_j v_j T) [m^-3]. Define M_k = sum(D_j**k w_j), with D in mm.

    NT=M0; Z=M6; LWC=(pi/6)*1e-3*M3 [g/m³]; Dm=M4/M3.
    Mass-spectrum variance=sum((D-Dm)**2 D**3 w)/M3 (Williams et al., 2014).
    Rain rate is a per-drop volume flux: sum((pi/6)*(D*1e-3)**3/(A*T)),
    converted from m/s to mm/h by 3.6e6. Velocity cancels from this flux.
    DSD for bin i is sum(w_j in bin i)/0.2 [m^-3 mm^-1].

    Integrals use individual diameters, not bin-center approximations to D³ or
    D⁶. The DSD is a histogram in concentration per diameter, NOT raw counts.
    Both velocity choices are returned; no RainNASA integral code is included.
    """
    prepared = prepare_drops(raw_data)
    drops = _selected_physical_inputs(prepared)
    date = pd.Timestamp(raw_data.attrs["date"])
    minute_index = (drops.hour * 60 + drops.minute).to_numpy(dtype=int)
    diameter_mm = drops.diameter_mm.to_numpy()
    diameter_m = diameter_mm * 1e-3
    sampling_area_m2 = drops.sampling_area_mm2.to_numpy() * 1e-6
    all_minutes = pd.date_range(date, periods=MINUTES_PER_DAY, freq="min")

    def sum_per_minute(contribution):
        """Accumulate one physical contribution per drop into its minute slot.

        Repeated minute indices must add, not overwrite. minlength retains all
        1440 slots, including gaps before the first and after the last record.
        Explicit float output also handles an empty input without integer NaN
        assignment problems when the missing-data policy is applied below.
        """
        return np.bincount(
            minute_index, weights=contribution, minlength=MINUTES_PER_DAY
        ).astype(float)

    accepted_count = np.bincount(minute_index, minlength=MINUTES_PER_DAY)
    has_drops = accepted_count > 0
    minimum_diameter = np.full(MINUTES_PER_DAY, np.inf)
    maximum_diameter = np.full(MINUTES_PER_DAY, -np.inf)
    np.minimum.at(minimum_diameter, minute_index, diameter_mm)
    np.maximum.at(maximum_diameter, minute_index, diameter_mm)
    minimum_diameter[~has_drops] = np.nan
    maximum_diameter[~has_drops] = np.nan

    centers_mm, edges_mm = _diameter_bins()
    bin_index = np.searchsorted(edges_mm, diameter_mm, side="right") - 1
    in_bin = (bin_index >= 0) & (bin_index < len(centers_mm))
    bin_counts = np.zeros((MINUTES_PER_DAY, len(centers_mm)), dtype=np.int32)
    # Indexed accumulation counts every drop, including repeated minute/bin pairs.
    # Ordinary indexed += would lose repeated-index contributions.
    np.add.at(bin_counts, (minute_index[in_bin], bin_index[in_bin]), 1)
    per_drop_volume_m3 = (np.pi / 6.0) * diameter_m**3
    rain_rate = (
        sum_per_minute(per_drop_volume_m3 / (sampling_area_m2 * SAMPLE_SECONDS)) * 3.6e6
    )
    # Keep the established reporting mask separate from calculated values.
    # This is inherited processing policy, not a universal literature threshold.
    output_valid = ((accepted_count > 10) & (rain_rate >= 0.01)).astype(np.int8)
    parameters = xr.Dataset(coords={"time": all_minutes, "velocity": list(VELOCITIES)})
    spectra = []
    branch_parameters = []
    for velocity_name in VELOCITIES:
        velocity_m_s = drops[f"{velocity_name}_velocity_m_s"].to_numpy()
        sample_volume_m3 = sampling_area_m2 * velocity_m_s * SAMPLE_SECONDS
        drop_concentration_per_m3 = 1.0 / sample_volume_m3
        third_moment = sum_per_minute(diameter_mm**3 * drop_concentration_per_m3)
        fourth_moment = sum_per_minute(diameter_mm**4 * drop_concentration_per_m3)
        sixth_moment = sum_per_minute(diameter_mm**6 * drop_concentration_per_m3)
        mean_mass_diameter_mm = np.divide(
            fourth_moment,
            third_moment,
            out=np.full(MINUTES_PER_DAY, np.nan),
            where=third_moment > 0,
        )
        mass_variance_numerator = sum_per_minute(
            (diameter_mm - mean_mass_diameter_mm[minute_index]) ** 2
            * diameter_mm**3
            * drop_concentration_per_m3
        )
        mass_variance = np.divide(
            mass_variance_numerator,
            third_moment,
            out=np.full(MINUTES_PER_DAY, np.nan),
            where=third_moment > 0,
        )
        branch_parameters.append(
            {
                "rain_rate": rain_rate,
                "number_concentration": sum_per_minute(drop_concentration_per_m3),
                "liquid_water_content": sum_per_minute(
                    WATER_DENSITY_KG_M3 * per_drop_volume_m3 * drop_concentration_per_m3
                )
                * 1000.0,  # kg/m³ -> g/m³
                "reflectivity": sixth_moment,
                "reflectivity_dbz": 10
                * np.log10(np.where(sixth_moment > 0, sixth_moment, np.nan)),
                "mass_weighted_diameter": mean_mass_diameter_mm,
                "diameter_std": np.sqrt(mass_variance),
                "minimum_diameter": minimum_diameter,
                "maximum_diameter": maximum_diameter,
                "drop_count": accepted_count,
            }
        )
        spectrum = np.zeros_like(bin_counts, dtype=float)
        np.add.at(
            spectrum,
            (minute_index[in_bin], bin_index[in_bin]),
            drop_concentration_per_m3[in_bin] / 0.2,
        )
        spectra.append(spectrum)

    # Assemble time-by-velocity arrays in the declared velocity order. Counts
    # remain zero in empty minutes; physical bulk values become NaN rather than
    # asserting dry conditions during a period with unknown instrument uptime.
    for name, units in INTEGRAL_UNITS.items():
        values = np.column_stack([branch[name] for branch in branch_parameters])
        if name != "drop_count":
            values[~has_drops, :] = np.nan
        parameters[name] = (("time", "velocity"), values, {"units": units})
    dsd = xr.Dataset(
        {
            "dsd": (
                ("time", "diameter", "velocity"),
                np.stack(spectra, axis=-1),
                {"units": "m-3 mm-1"},
            ),
            "bin_drop_count": (("time", "diameter"), bin_counts, {"units": "1"}),
            "bin_lower_mm": ("diameter", edges_mm[:-1], {"units": "mm"}),
            "bin_upper_mm": ("diameter", edges_mm[1:], {"units": "mm"}),
            "bin_width_mm": ((), 0.2, {"units": "mm"}),
        },
        coords={
            "time": all_minutes,
            "diameter": centers_mm,
            "velocity": list(VELOCITIES),
        },
    )
    dsd.diameter.attrs.update(units="mm", long_name="Nominal drop diameter bin center")
    metadata = dict(raw_data.attrs)
    metadata.update(
        package="gv_tools",
        package_version=__version__,
        source_implementation="py_2dvd_lite 0.1.2",
        method="literature",
        accepted_rows=int(accepted_count.sum()),
        sample_seconds=SAMPLE_SECONDS,
        water_density_kg_m3=WATER_DENSITY_KG_M3,
        processing_conventions="Inherited dropbydrop QC; three-decimal/float32 inputs; float32 bin edges",
        reporting_threshold="accepted_count > 10 and rain_rate >= 0.01 mm/h",
        missing_policy="No accepted drops: bulk NaN except count=0; DSD/counts zero; output_valid=0; no uptime inference",
        reflectivity_assumption="Liquid-water Rayleigh D6 moment, not Mie or polarimetric reflectivity",
        references="Tokay et al. 2001, 2013; Kruger & Krajewski 2002; Williams et al. 2014; docs/py_2dvd/references.md",
        table_sha256=json.dumps(
            {
                name: _file_hash(files("gv_tools").joinpath("data", "py_2dvd", name))
                for name in TABLE_NAMES
            }
        ),
        created_utc=datetime.now(timezone.utc).isoformat(),
        ai_provenance="OpenAI Codex; exact deployed model identifier unavailable",
    )
    raw_minute_index = (raw_data.hour * 60 + raw_data.minute).to_numpy(dtype=int)
    for product in (parameters, dsd):
        product["output_valid"] = ("time", output_valid, {"units": "1"})
        product["raw_drop_count"] = (
            "time",
            np.bincount(raw_minute_index, minlength=MINUTES_PER_DAY),
            {"units": "1"},
        )
        # Independent dictionaries prevent later edits to one product's metadata
        # from changing its sibling. Shared coordinates/masks enable strict export.
        product.attrs = metadata.copy()
    return parameters, dsd


def calculate_integral_parameters(raw_data):
    """Calculate and return literature integral parameters from raw drop records.

    raw_data must be the DataFrame from ingest_raw, including attrs for date,
    site and instrument. Its rows and metadata are not modified. Returns a
    Dataset with time (1440 minutes) and velocity (terminal, measured) dimensions;
    all ten integral variables carry units. output_valid is a reporting mask,
    not a replacement for stored values; raw_drop_count describes acquisition.

    This convenience call computes the shared products and returns the integral
    half. Calling it and calculate_dsd separately repeats that computation. Use
    calculate_products when both are needed. No cache can become stale after
    a caller edits raw data; no output files or figures are produced here.
    """
    return calculate_products(raw_data)[0]


def calculate_dsd(raw_data):
    """Calculate and return literature number concentration per diameter bin.

    raw_data has the same contract as calculate_integral_parameters. Returns
    dsd(time, diameter, velocity) in m^-3 mm^-1, shared bin_drop_count, lower/upper
    edges, nominal width, reporting mask and raw counts. Neither raw_data nor
    its attributes are changed. The diameter coordinate holds nominal centers;
    membership uses stored edges as explained in _diameter_bins.

    Empty bins contain zero; masking and logarithmic display belong to graph.py.
    Use calculate_products to obtain both this Dataset and integral parameters
    in one pass. No files are written and no plotting backend is initialized.
    """
    return calculate_products(raw_data)[1]
