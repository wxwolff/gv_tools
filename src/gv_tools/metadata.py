"""Instrument and deployment metadata shared by all GV products."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class InstrumentMetadata:
    """Identity and deployment information attached to a processed product.

    Times in GV Tools products are always UTC. ``timezone`` documents the
    source convention and is retained for provenance.
    """

    instrument_id: str
    instrument_type: str
    site_id: str
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    timezone: str = "UTC"
    campaign: str = "GPM_GV"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("instrument_id", "instrument_type", "site_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.latitude is not None and not -90 <= self.latitude <= 90:
            raise ValueError("latitude must be between -90 and 90 degrees")
        if self.longitude is not None and not -180 <= self.longitude <= 180:
            raise ValueError("longitude must be between -180 and 180 degrees")

    def to_attrs(self) -> dict[str, Any]:
        """Return NetCDF-safe global attributes."""
        values = asdict(self)
        extra = values.pop("extra")
        attrs = {key: value for key, value in values.items() if value is not None}
        attrs.update({key: value for key, value in extra.items() if value is not None})
        return attrs

