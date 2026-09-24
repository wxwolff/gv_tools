"""Minimal task-oriented GV Tools example."""

from pathlib import Path

import gv_tools


INPUT_DIR = Path("/path/to/input")
OUTPUT_DIR = Path("/path/to/output")
FILES = [INPUT_DIR / "PIERS0042_Parsivel_20260810_daily.zip"]
parsivel = gv_tools.io.read_parsivel(FILES)

gv_tools.correct.validate_product(parsivel)
gv_tools.io.write_product(parsivel, OUTPUT_DIR, formats=("netcdf",), day="2026-08-10")
