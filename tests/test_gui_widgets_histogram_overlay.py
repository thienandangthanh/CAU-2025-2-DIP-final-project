"""Tests for HistogramOverlay widget."""

import pytest
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from gui.widgets.histogram_overlay import HistogramOverlay


@pytest.fixture(scope="module")
def qapp():
    """Provide QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_test_pixmap(color: Qt.GlobalColor = Qt.GlobalColor.red) -> QPixmap:
    pixmap = QPixmap(64, 64)
    pixmap.fill(color)
    return pixmap


def test_histogram_overlay_generates_data(qapp):
    """Histogram overlay should produce cached histogram data."""
    overlay = HistogramOverlay()
    overlay.set_pixmap(_make_test_pixmap())
    overlay.set_histogram_type("grayscale")
    data = overlay.histogram_snapshot()
    assert isinstance(data, list)
    assert len(data) == 256
    assert max(data) <= 1.0


def test_histogram_overlay_rgb_mode(qapp):
    """RGB mode should return per-channel data."""
    overlay = HistogramOverlay()
    overlay.set_pixmap(_make_test_pixmap(Qt.GlobalColor.blue))
    overlay.set_histogram_type("rgb")
    data = overlay.histogram_snapshot()
    assert isinstance(data, dict)
    assert set(data.keys()) == {"r", "g", "b"}
    assert len(data["r"]) == 256


def test_histogram_overlay_draggable_with_parent(qapp):
    """Overlay should remain within parent bounds when moved."""
    parent = QLabel()
    parent.resize(300, 300)
    overlay = HistogramOverlay(parent)
    overlay.set_pixmap(_make_test_pixmap())
    overlay.set_histogram_type("grayscale")
    overlay.move(500, 500)  # Force out of bounds
    overlay.ensure_within_parent()
    assert overlay.x() + overlay.width() <= parent.width()
    assert overlay.y() + overlay.height() <= parent.height()
