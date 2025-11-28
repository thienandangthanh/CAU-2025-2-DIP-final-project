"""Tests for ImagePanel widget.

Tests the image display widget functionality including:
- Empty state display
- Image loading from file
- Image loading from pixmap
- Drag & drop support
- Click events
- Info overlay display
"""

import pytest
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from gui.widgets.image_panel import ImagePanel


@pytest.fixture(scope="module")
def qapp():
    """Provide a QApplication instance for all tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def image_panel(qapp):
    """Provide a fresh ImagePanel instance for each test."""
    panel = ImagePanel(title="Test Panel", placeholder_text="Test placeholder")
    return panel


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample image file for testing."""
    from PIL import Image
    import numpy as np

    # Create a simple test image (100x100 red square)
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)
    image_array[:, :, 0] = 255  # Red channel
    image = Image.fromarray(image_array, mode="RGB")

    # Save to temp file
    image_path = tmp_path / "test_image.png"
    image.save(image_path)

    return str(image_path)


class TestImagePanelInitialization:
    """Tests for ImagePanel initialization."""

    def test_init_creates_widget(self, qapp):
        """Test that ImagePanel initializes correctly."""
        panel = ImagePanel()
        assert panel is not None
        assert panel.title == "Image"
        assert panel.placeholder_text == "Click to open an image"

    def test_init_with_custom_title(self, qapp):
        """Test initialization with custom title."""
        panel = ImagePanel(title="Custom Title")
        assert panel.title == "Custom Title"

    def test_init_with_custom_placeholder(self, qapp):
        """Test initialization with custom placeholder text."""
        panel = ImagePanel(placeholder_text="Custom placeholder")
        assert panel.placeholder_text == "Custom placeholder"

    def test_initial_state_has_no_image(self, image_panel):
        """Test that panel starts with no image."""
        assert not image_panel.has_image()
        assert image_panel.get_pixmap() is None
        assert image_panel.get_image_path() is None


class TestImagePanelImageLoading:
    """Tests for image loading functionality."""

    def test_set_image_from_file(self, image_panel, sample_image_path):
        """Test loading image from file."""
        success = image_panel.set_image(sample_image_path)
        assert success
        assert image_panel.has_image()
        assert image_panel.get_pixmap() is not None
        assert image_panel.get_image_path() == sample_image_path

    def test_set_image_from_invalid_file(self, image_panel, tmp_path):
        """Test loading from invalid file fails gracefully."""
        invalid_path = str(tmp_path / "nonexistent.png")
        success = image_panel.set_image(invalid_path)
        assert not success
        assert not image_panel.has_image()

    def test_set_image_from_pixmap(self, image_panel):
        """Test loading image from QPixmap."""
        # Create a test pixmap
        pixmap = QPixmap(100, 100)
        pixmap.fill(Qt.GlobalColor.red)

        image_panel.set_image_from_pixmap(pixmap, "test_path.png")
        assert image_panel.has_image()
        assert image_panel.get_pixmap() is not None
        assert image_panel.get_image_path() == "test_path.png"

    def test_clear_removes_image(self, image_panel, sample_image_path):
        """Test that clear() removes the current image."""
        image_panel.set_image(sample_image_path)
        assert image_panel.has_image()

        image_panel.clear()
        assert not image_panel.has_image()
        assert image_panel.get_pixmap() is None
        assert image_panel.get_image_path() is None


class TestImagePanelStates:
    """Tests for different panel states."""

    def test_processing_state(self, image_panel):
        """Test setting processing state."""
        image_panel.set_processing(True, "Processing test...")
        assert image_panel.is_processing

        image_panel.set_processing(False)
        assert not image_panel.is_processing

    def test_info_overlay_visibility(self, image_panel, sample_image_path):
        """Test info overlay visibility toggle."""
        image_panel.set_image(sample_image_path)

        # Show overlay
        image_panel.set_info_overlay_visible(True)
        # Note: We can't easily test visibility without rendering,
        # but we can verify the method doesn't crash

        # Hide overlay
        image_panel.set_info_overlay_visible(False)

    def test_histogram_toggle(self, image_panel, sample_image_path):
        """Test enabling and disabling histogram overlay."""
        image_panel.set_image(sample_image_path)
        image_panel.set_histogram_type("grayscale")
        image_panel.set_histogram_enabled(True)
        assert image_panel.histogram_visible is True

        # Switch type while visible
        image_panel.set_histogram_type("rgb")
        assert image_panel.histogram_visible is True

        image_panel.set_histogram_enabled(False)
        assert image_panel.histogram_visible is False


class TestImagePanelFileValidation:
    """Tests for file validation."""

    def test_is_image_file_valid_extensions(self, image_panel):
        """Test that valid image extensions are recognized."""
        valid_files = [
            "/path/to/image.jpg",
            "/path/to/image.jpeg",
            "/path/to/image.png",
            "/path/to/image.bmp",
        ]

        for filepath in valid_files:
            assert image_panel._is_image_file(filepath)

    def test_is_image_file_invalid_extensions(self, image_panel):
        """Test that invalid extensions are rejected."""
        invalid_files = [
            "/path/to/file.txt",
            "/path/to/file.pdf",
            "/path/to/file.doc",
        ]

        for filepath in invalid_files:
            assert not image_panel._is_image_file(filepath)

    def test_is_image_file_case_insensitive(self, image_panel):
        """Test that extension check is case-insensitive."""
        assert image_panel._is_image_file("/path/to/image.JPG")
        assert image_panel._is_image_file("/path/to/image.PNG")


class TestImagePanelSignals:
    """Tests for PyQt signals."""

    def test_image_clicked_signal(self, image_panel, qtbot):
        """Test that clicking emits image_clicked signal."""
        with qtbot.waitSignal(image_panel.image_clicked, timeout=1000):
            # Simulate mouse click
            qtbot.mouseClick(image_panel, Qt.MouseButton.LeftButton)

    def test_image_clicked_not_emitted_during_processing(self, image_panel, qtbot):
        """Test that click is ignored during processing."""
        image_panel.set_processing(True)

        # Click should not emit signal during processing
        with qtbot.assertNotEmitted(image_panel.image_clicked):
            qtbot.mouseClick(image_panel, Qt.MouseButton.LeftButton)


@pytest.mark.gui
class TestImagePanelIntegration:
    """Integration tests for ImagePanel."""

    def test_full_workflow(self, image_panel, sample_image_path):
        """Test complete workflow: load, display, clear."""
        # Start with no image
        assert not image_panel.has_image()

        # Load image
        success = image_panel.set_image(sample_image_path)
        assert success
        assert image_panel.has_image()

        # Clear image
        image_panel.clear()
        assert not image_panel.has_image()
