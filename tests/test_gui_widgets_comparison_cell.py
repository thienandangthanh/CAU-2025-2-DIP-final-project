"""Tests for ComparisonCell widget.

This module tests the comparison cell widget functionality including:
- Image display
- Status updates
- Timing information
- Click events
"""

import pytest
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from gui.widgets.comparison_cell import ComparisonCell


@pytest.fixture
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_image(tmp_path):
    """Create a test image file."""
    # Create a simple test pixmap
    pixmap = QPixmap(100, 100)
    pixmap.fill(Qt.GlobalColor.blue)
    
    # Save to temp file
    image_path = tmp_path / "test_image.png"
    pixmap.save(str(image_path))
    
    return str(image_path), pixmap


class TestComparisonCell:
    """Tests for ComparisonCell widget."""
    
    def test_init(self, qapp):
        """Test cell initialization."""
        cell = ComparisonCell(
            method_key="test-method",
            method_name="Test Method"
        )
        
        assert cell.method_key == "test-method"
        assert cell.method_name == "Test Method"
        assert cell.is_reference is False
        assert cell.status == "pending"
        assert cell.current_pixmap is None
    
    def test_init_reference_cell(self, qapp):
        """Test reference cell initialization."""
        cell = ComparisonCell(
            method_key="reference",
            method_name="Reference",
            is_reference=True
        )
        
        assert cell.is_reference is True
    
    def test_set_image(self, qapp, test_image):
        """Test setting image."""
        _, pixmap = test_image
        cell = ComparisonCell("test", "Test")
        
        cell.set_image(pixmap)
        
        assert cell.current_pixmap is not None
        assert not cell.current_pixmap.isNull()
    
    def test_set_status_pending(self, qapp):
        """Test setting pending status."""
        cell = ComparisonCell("test", "Test")
        
        cell.set_status("pending")
        
        assert cell.status == "pending"
        assert "Pending" in cell.status_label.text()
    
    def test_set_status_running(self, qapp):
        """Test setting running status."""
        cell = ComparisonCell("test", "Test")
        
        cell.set_status("running")
        
        assert cell.status == "running"
        assert "Processing" in cell.status_label.text()
    
    def test_set_status_done_with_timing(self, qapp):
        """Test setting done status with timing."""
        cell = ComparisonCell("test", "Test")
        
        cell.set_status("done", "2.34s")
        
        assert cell.status == "done"
        assert cell.timing_text == "2.34s"
        assert "2.34s" in cell.status_label.text()
    
    def test_set_status_done_without_timing(self, qapp):
        """Test setting done status without timing."""
        cell = ComparisonCell("test", "Test")
        
        cell.set_status("done")
        
        assert cell.status == "done"
        assert "Completed" in cell.status_label.text()
    
    def test_set_error(self, qapp):
        """Test setting error status."""
        cell = ComparisonCell("test", "Test")
        
        cell.set_error("Test error message")
        
        assert cell.status == "error"
        assert "Error" in cell.status_label.text()
    
    def test_clear(self, qapp, test_image):
        """Test clearing cell."""
        _, pixmap = test_image
        cell = ComparisonCell("test", "Test")
        
        # Set some state
        cell.set_image(pixmap)
        cell.set_status("done", "2.34s")
        
        # Clear
        cell.clear()
        
        # Verify cleared
        assert cell.current_pixmap is None
        assert cell.status == "pending"
        assert cell.timing_text is None
    
    def test_click_signal_emitted(self, qapp, test_image):
        """Test that click signal is emitted when cell is clicked."""
        _, pixmap = test_image
        cell = ComparisonCell("test-method", "Test Method")
        cell.set_image(pixmap)
        
        # Track signal emissions
        signals_received = []
        cell.clicked.connect(lambda key: signals_received.append(key))
        
        # Simulate mouse click
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QMouseEvent
        
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(50.0, 50.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        
        cell.mousePressEvent(event)
        
        # Verify signal was emitted with correct method key
        assert len(signals_received) == 1
        assert signals_received[0] == "test-method"
    
    def test_no_click_signal_without_image(self, qapp):
        """Test that click signal is not emitted when no image is loaded."""
        cell = ComparisonCell("test", "Test")
        
        # Track signal emissions
        signals_received = []
        cell.clicked.connect(lambda key: signals_received.append(key))
        
        # Simulate mouse click
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QMouseEvent
        
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(50.0, 50.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        
        cell.mousePressEvent(event)
        
        # Verify no signal was emitted
        assert len(signals_received) == 0
    
    def test_reference_cell_status_display(self, qapp, test_image):
        """Test that reference cell shows different status text."""
        _, pixmap = test_image
        cell = ComparisonCell("reference", "Reference", is_reference=True)
        
        # Before image
        assert "Waiting" in cell.status_label.text()
        
        # After image - need to set status to update the label
        cell.set_image(pixmap)
        cell.set_status("done")  # This triggers the status update
        assert "Original" in cell.status_label.text()

    def test_histogram_configuration(self, qapp, test_image):
        """Test histogram overlay can be toggled on a cell."""
        _, pixmap = test_image
        cell = ComparisonCell("test-method", "Test Method")
        cell.set_image(pixmap)
        cell.set_histogram_type("grayscale")
        cell.set_histogram_enabled(True)
        assert cell.histogram_visible is True
        cell.set_histogram_type("rgb")
        assert cell.histogram_visible is True
        cell.set_histogram_enabled(False)
        assert cell.histogram_visible is False


@pytest.mark.gui
class TestComparisonCellIntegration:
    """Integration tests for ComparisonCell widget."""
    
    def test_cell_displays_correctly(self, qapp, test_image):
        """Test that cell displays correctly with all components."""
        _, pixmap = test_image
        cell = ComparisonCell("test-method", "Test Method")
        
        # Set image and status
        cell.set_image(pixmap)
        cell.set_status("done", "1.23s")
        
        # Verify all components are visible and correct
        assert cell.name_label.text() == "Test Method"
        assert "1.23s" in cell.status_label.text()
        assert not cell.image_label.pixmap().isNull()
    
    def test_status_updates_border_color(self, qapp, test_image):
        """Test that status updates change border color."""
        _, pixmap = test_image
        cell = ComparisonCell("test", "Test")
        cell.set_image(pixmap)
        
        # Test different statuses
        cell.set_status("running")
        assert "#2196F3" in cell.image_label.styleSheet()  # Blue
        
        cell.set_status("done")
        assert "#4CAF50" in cell.image_label.styleSheet()  # Green
        
        cell.set_error("Error")
        assert "#F44336" in cell.image_label.styleSheet()  # Red
