"""Small command-line adapters; all reading, physics and plotting live elsewhere."""

import argparse
import json
import tarfile
import zipfile
from pathlib import Path

from .py_2dvd import __version__, calculate_products, ingest_raw, save_products


def _arguments(description):
    """Construct only options shared by processing and plotting.

    Relative paths resolve from the shell's current directory. The separate
    Output default avoids the full package's default output tree.
    argparse handles syntax/help errors before any scientific work begins.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("datafile", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("Output"))
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def process_main(argv=None):
    """Read one day, calculate both products once, and save both velocities.

    argv=None uses the shell arguments; a list supports testing or programmatic
    invocation. Input is a raw text/ZIP/TGZ path. Site/instrument identify the
    products but do not change the physical equations or sampling geometry.
    Return 0 on success; argparse exits with status 2 for handled input errors.
    The notebook API is preferable when the caller wants in-memory objects.
    """
    parser = _arguments(
        "Ingest one 2DVD text/ZIP/TGZ file; save literature parameters and DSD."
    )
    parser.add_argument("--site", default="WFF")
    parser.add_argument("--instrument", default="sn37")
    arguments = parser.parse_args(argv)
    try:
        raw_data = ingest_raw(
            arguments.datafile, site=arguments.site, instrument=arguments.instrument
        )
        parameters, dsd = calculate_products(raw_data)
        paths = save_products(parameters, dsd, arguments.output_dir)
        print(
            f"Read {len(raw_data):,} drops; accepted {parameters.attrs['accepted_rows']:,}"
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
    except (ValueError, OSError, zipfile.BadZipFile, tarfile.TarError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


def plot_main(argv=None):
    """Render both figures from saved 2DVD products, without rereading raw drops.

    The positional input is NetCDF, not raw data. --velocity selects an existing
    coordinate. --plot-config provides rain/dsd keyword dictionaries and a shared
    rcParams style. --show controls interactive display; figures are saved in
    either case. Return 0 on success or exit 2 for handled errors. A plotting
    failure can leave an earlier saved figure in place; this is not an atomic
    two-file transaction. Re-run after fixing the offending configuration.
    """
    parser = _arguments(
        "Plot literature integral parameters and DSD from a 2DVD NetCDF."
    )
    parser.add_argument(
        "--velocity", choices=("measured", "terminal"), default=None,
        help="Velocity to plot; defaults to the saved branch (measured for legacy combined files)"
    )
    parser.add_argument(
        "--plot-config",
        type=Path,
        help="Optional JSON: rain, dsd and rcParams sections",
    )
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        import matplotlib.pyplot as plt
        import xarray as xr

        from .graph import plot_dsd, plot_integral_parameters

        # Select a noninteractive backend only for the command-line save workflow.
        # Notebook imports and public graph functions leave backend choice alone.
        if not arguments.show:
            plt.switch_backend("Agg")
        configuration = (
            json.loads(arguments.plot_config.read_text())
            if arguments.plot_config
            else {}
        )
        if not isinstance(configuration, dict) or set(configuration) - {
            "rain",
            "dsd",
            "rcParams",
        }:
            raise ValueError(
                "Plot configuration sections must be rain, dsd and rcParams"
            )
        # Load arrays while the file is open, then close it before plotting so a
        # long-lived figure does not keep the NetCDF handle or lazy reads alive.
        with xr.open_dataset(arguments.datafile) as source:
            products = source.load()
        available = list(products.velocity.values)
        velocity = arguments.velocity or (available[0] if len(available) == 1 else "measured")
        if velocity not in available:
            raise ValueError(f"Requested {velocity} velocity is absent; available: {available}")
        # CLI owns velocity/output; JSON supplies visual options, not new science.
        common_options = dict(
            velocity=velocity,
            output_dir=arguments.output_dir,
            rc_params=configuration.get("rcParams", {}),
        )
        rain_figure = plot_integral_parameters(
            products, **common_options, **configuration.get("rain", {})
        )
        dsd_figure = plot_dsd(
            products, **common_options, **configuration.get("dsd", {})
        )
        if arguments.show:
            plt.show()
        else:
            # Release figure memory in batch use after both PNGs have been saved.
            plt.close(rain_figure)
            plt.close(dsd_figure)
        print(
            f"Saved {velocity}-velocity literature plots under {arguments.output_dir}/Plots"
        )
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0
