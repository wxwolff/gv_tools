import pytest

from gv_tools import InstrumentMetadata


def test_metadata_validates_geographic_coordinates():
    with pytest.raises(ValueError, match="latitude"):
        InstrumentMetadata("apu01", "disdrometer", "WFF", latitude=91)


def test_metadata_attributes_include_extra_values():
    metadata = InstrumentMetadata(
        "apu01", "disdrometer", "WFF", serial_number="123", extra={"height_m": 1.5}
    )
    assert metadata.to_attrs()["height_m"] == 1.5
    assert metadata.to_attrs()["serial_number"] == "123"

