"""Command-line entry point."""

from __future__ import annotations

import argparse
import platform
import sys


def build_parser():
    parser = argparse.ArgumentParser(prog="gv-tools")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="report runtime dependency availability")
    check.add_argument(
        "--include", choices=("plot", "notebook"), action="append", default=[]
    )
    parsivel = commands.add_parser("parsivel", help="process Parsivel input")
    parsivel.add_argument("input")
    parsivel.add_argument("--instrument-id", required=True)
    parsivel.add_argument("--site-id", required=True)
    parsivel.add_argument(
        "--source-platform",
        choices=("auto", "apu", "piers"),
        default="auto",
        help="raw archive family (default: infer from instrument ID)",
    )
    parsivel.add_argument("--output", required=True)
    parsivel.add_argument("--format", choices=("netcdf", "csv"), action="append")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "check":
        from .environment import dependency_report

        groups = ("required", "parsivel", "netcdf", *args.include)
        print(f"Python: {sys.version.split()[0]}")
        print(f"Platform: {platform.platform()}")
        missing = []
        for item in dependency_report(groups):
            status = "OK" if item["compatible"] else (
                "OLD" if item["available"] else "MISSING"
            )
            print(
                f"{status:7} {item['package']:<18} installed={item['installed']} "
                f"minimum={item['minimum']} ({item['group']})"
            )
            if not item["compatible"]:
                missing.append(item["package"])
        if missing:
            print("Install with: python -m pip install 'gv_tools[parsivel,netcdf]'")
            return 1
        return 0

    from .metadata import InstrumentMetadata
    from .outputs import write_product
    from .registry import create_adapter
    metadata = InstrumentMetadata(
        instrument_id=args.instrument_id,
        instrument_type="disdrometer",
        site_id=args.site_id,
        manufacturer="OTT HydroMet",
        model="Parsivel",
    )
    adapter = create_adapter(
        "parsivel", metadata, source_platform=args.source_platform
    )
    products = adapter.read(args.input)
    formats = tuple(args.format or ["netcdf"])
    for day, dataset in products.items():
        write_product(dataset, args.output, formats=formats, day=day)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
