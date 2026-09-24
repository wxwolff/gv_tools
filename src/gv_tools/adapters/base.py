"""Interface implemented by GV instrument adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import xarray as xr

from ..metadata import InstrumentMetadata


class InstrumentAdapter(ABC):
    """Translate instrument-specific input into normalized GV products."""

    name: str

    def __init__(self, metadata: InstrumentMetadata):
        self.metadata = metadata

    @abstractmethod
    def discover(self, input_path: str | Path) -> list[str]:
        """Return the dates available below *input_path* as YYYY-MM-DD."""

    @abstractmethod
    def read(
        self,
        input_path: str | Path,
        **options,
    ) -> dict[str, xr.Dataset]:
        """Read and normalize products keyed by YYYY-MM-DD."""
