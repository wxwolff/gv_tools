"""Tests for the task-oriented public interface."""

import gv_tools


def test_task_namespaces_are_available():
    assert gv_tools.core.InstrumentMetadata is gv_tools.InstrumentMetadata
    assert gv_tools.io.create_adapter is gv_tools.create_adapter
    assert gv_tools.io.read_mrr is gv_tools.read_mrr
    assert gv_tools.correct.validate_product is gv_tools.validate_product
    assert gv_tools.graph.plot_directory is gv_tools.plot_directory
    assert "required" in gv_tools.config.DEPENDENCY_GROUPS


def test_task_namespace_exports_are_deliberate():
    assert "create_adapter" in gv_tools.io.__all__
    assert "read_mrr" in gv_tools.io.__all__
    assert "validate_product" in gv_tools.correct.__all__
    assert "plot_aio_quicklook" in gv_tools.graph.__all__
    assert gv_tools.graph.plot_ws800_full_quicklook is gv_tools.plot_ws800_full_quicklook
    assert gv_tools.graph.plot_radar_ppi_quicklook is gv_tools.plot_radar_ppi_quicklook
    assert gv_tools.graph.plot_radar_rhi_quicklook is gv_tools.plot_radar_rhi_quicklook
    assert gv_tools.graph.plot_radar_bb_zdr_calibration is gv_tools.plot_radar_bb_zdr_calibration
    assert gv_tools.graph.plot_mrr_time_height_quicklook is gv_tools.plot_mrr_time_height_quicklook
    assert gv_tools.graph.plot_parsivel_quicklook is gv_tools.plot_parsivel_quicklook
    assert not hasattr(gv_tools, "plot_aio_overview")
    assert not hasattr(gv_tools, "plot_parsivel_overview")
