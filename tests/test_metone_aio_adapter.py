from pathlib import Path

import pandas as pd
import pytest

from gv_tools import InstrumentMetadata, available_adapters, create_adapter
from gv_tools.adapters.metone_aio import MetOneAIOAdapter, RawMetOneAIOProduct


HEADER = "01+AIntM  02+.Year  03+..Day  04+.HrMn  05+.WSpd  06+.WDir  07+DirSD  08+.Temp  09+Humid  10+.Press  15+User2\n"
ROW = "01+0001.  02+2025.  03+0001.  04+0000.  05+001.4  06+0070.  07+023.1  08+005.6  09+071.0  10+0993.2  15+23.71\n"


def metadata(**changes):
    values = dict(instrument_id="AIO_GAIL", instrument_type="weather_station", site_id="GAIL", manufacturer="Met One", model="AIO", timezone="UTC")
    values.update(changes)
    return InstrumentMetadata(**values)


def sample(path: Path, text=HEADER + ROW):
    path.write_text(text, encoding="ascii")
    return path


def test_discovers_reads_and_normalizes_monthly_file(tmp_path):
    source = sample(tmp_path / "AIO_GAIL_202501.dat")
    adapter = MetOneAIOAdapter(metadata())
    assert adapter.discover(tmp_path) == ["2025-01-01"]
    raw = adapter.read_raw(source)["2025-01-01"]
    assert isinstance(raw, RawMetOneAIOProduct)
    dataset = adapter.read(source)["2025-01-01"]
    assert dataset.wind_speed.item() == pytest.approx(1.4)
    assert dataset.wind_direction_standard_deviation.item() == pytest.approx(23.1)
    assert dataset.user_channel_2.item() == pytest.approx(23.71)
    assert dataset.attrs["source_platform"] == "metone_aio"


def test_timezone_filter_rejections_duplicates_and_registry(tmp_path):
    text = HEADER + ROW + ROW + "bad row\n"
    source = sample(tmp_path / "AIO_GAIL_202501.dat", text)
    adapter = create_adapter("metone_aio", metadata(timezone="America/New_York"))
    raw = adapter.read_raw(source)["2025-01-01"]
    assert raw.rejected_rows == 1
    assert raw.duplicate_rows == 1
    dataset = adapter.read(source)["2025-01-01"]
    assert pd.Timestamp(dataset.time.values[0]) == pd.Timestamp("2025-01-01 05:00")
    assert "metone_aio" in available_adapters()


def test_rejects_filename_for_another_site(tmp_path):
    source = sample(tmp_path / "AIO_OTHER_202501.dat")
    with pytest.raises(ValueError, match="filename must match"):
        MetOneAIOAdapter(metadata()).read(source)
