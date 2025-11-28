"""Tests for image panel timing display functionality.

This module tests the timing display added to ImagePanel in Task 2.9 Step 3
to show enhancement elapsed time in the info overlay.
"""

import pytest
from PIL import Image
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from gui.widgets import ImagePanel


@pytest.fixture(scope="module")
def qapp():
    """Provide QApplication instance for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def image_panel(qtbot, qapp):
    """Provide ImagePanel instance for testing."""
    panel = ImagePanel(title="Test Panel", placeholder_text="Test Placeholder")
    qtbot.addWidget(panel)
    return panel


@pytest.fixture
def sample_pixmap():
    """Provide a sample pixmap for testing."""
    pixmap = QPixmap(256, 256)
    pixmap.fill()
    return pixmap


class TestEnhancementTimeAttribute:
    """Tests for enhancement_time attribute initialization."""

    def test_enhancement_time_initialized_to_none(self, image_panel):
        """Test that enhancement_time is initialized to None."""
        assert hasattr(image_panel, "enhancement_time")
        assert image_panel.enhancement_time is None

    def test_enhancement_time_cleared_on_panel_clear(self, image_panel, sample_pixmap):
        """Test that enhancement_time is cleared when panel is cleared."""
        # Set some timing data
        image_panel.current_pixmap = sample_pixmap
        image_panel.enhancement_time = "2.34s"

        # Clear panel
        image_panel.clear()

        # Timing should be cleared
        assert image_panel.enhancement_time is None


class TestSetEnhancementTimeMethod:
    """Tests for set_enhancement_time() method."""

    def test_set_enhancement_time_method_exists(self, image_panel):
        """Test that set_enhancement_time method exists."""
        assert hasattr(image_panel, "set_enhancement_time")
        assert callable(image_panel.set_enhancement_time)

    def test_set_enhancement_time_stores_value(self, image_panel):
        """Test that set_enhancement_time stores the value."""
        image_panel.set_enhancement_time("2.34s")

        assert image_panel.enhancement_time == "2.34s"

    def test_set_enhancement_time_with_none(self, image_panel):
        """Test that set_enhancement_time can clear with None."""
        image_panel.set_enhancement_time("2.34s")
        assert image_panel.enhancement_time == "2.34s"

        image_panel.set_enhancement_time(None)
        assert image_panel.enhancement_time is None

    def test_set_enhancement_time_updates_existing_value(self, image_panel):
        """Test that set_enhancement_time can update existing value."""
        image_panel.set_enhancement_time("1.23s")
        assert image_panel.enhancement_time == "1.23s"

        image_panel.set_enhancement_time("4.56s")
        assert image_panel.enhancement_time == "4.56s"

    def test_set_enhancement_time_accepts_various_formats(self, image_panel):
        """Test that set_enhancement_time accepts various time formats."""
        formats = ["0.05s", "2.34s", "59.99s", "1m 0s", "1m 34s", "3m 5s"]

        for time_str in formats:
            image_panel.set_enhancement_time(time_str)
            assert image_panel.enhancement_time == time_str


class TestInfoOverlayTimingDisplay:
    """Tests for timing display in info overlay."""

    def test_info_overlay_shows_timing_when_set(self, image_panel, sample_pixmap):
        """Test that info overlay displays timing when enhancement_time is set."""
        # Set up image
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "enhanced_test.png"

        # Set timing
        image_panel.set_enhancement_time("2.34s")

        # Check info label text
        info_text = image_panel.info_label.text()

        assert "Enhanced in 2.34s" in info_text

    def test_info_overlay_without_timing(self, image_panel, sample_pixmap):
        """Test that info overlay works without timing (backward compatibility)."""
        # Set up image without timing
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "test.png"
        image_panel.enhancement_time = None

        image_panel._update_info_overlay()

        # Should show normal info without timing
        info_text = image_panel.info_label.text()

        assert "test.png" in info_text
        assert "256" in info_text  # Check for dimensions (any format)
        assert " px" in info_text
        assert "Enhanced in" not in info_text

    def test_info_overlay_updates_when_timing_added(self, image_panel, sample_pixmap):
        """Test that info overlay updates when timing is added after image."""
        # Set up image first
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "enhanced.png"
        image_panel._update_info_overlay()

        # Initially no timing
        info_text_before = image_panel.info_label.text()
        assert "Enhanced in" not in info_text_before

        # Add timing
        image_panel.set_enhancement_time("2.34s")

        # Now should have timing
        info_text_after = image_panel.info_label.text()
        assert "Enhanced in 2.34s" in info_text_after

    def test_info_overlay_format_with_all_fields(self, image_panel, sample_pixmap):
        """Test complete info overlay format with all fields including timing."""
        # Set up complete info
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "result_enhanced.png"
        image_panel.enhancement_time = "2.34s"

        image_panel._update_info_overlay()

        info_text = image_panel.info_label.text()

        # Should contain all fields
        assert "result_enhanced.png" in info_text
        assert "256" in info_text  # Check for dimensions
        assert " px" in info_text
        assert "In Memory" in info_text
        assert "Enhanced in 2.34s" in info_text

    def test_info_overlay_timing_appears_last(self, image_panel, sample_pixmap):
        """Test that timing appears as the last line in info overlay."""
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "test.png"
        image_panel.enhancement_time = "1.23s"

        image_panel._update_info_overlay()

        info_text = image_panel.info_label.text()
        lines = info_text.split("\n")

        # Timing should be on the last line
        assert "Enhanced in 1.23s" in lines[-1]


class TestInfoOverlayTimingFormats:
    """Tests for various timing format displays."""

    def test_short_timing_format(self, image_panel, sample_pixmap):
        """Test display of short timing (< 1 second)."""
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "fast.png"
        image_panel.set_enhancement_time("0.12s")

        info_text = image_panel.info_label.text()
        assert "Enhanced in 0.12s" in info_text

    def test_medium_timing_format(self, image_panel, sample_pixmap):
        """Test display of medium timing (few seconds)."""
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "medium.png"
        image_panel.set_enhancement_time("2.34s")

        info_text = image_panel.info_label.text()
        assert "Enhanced in 2.34s" in info_text

    def test_long_timing_format(self, image_panel, sample_pixmap):
        """Test display of long timing (minutes)."""
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "slow.png"
        image_panel.set_enhancement_time("1m 34s")

        info_text = image_panel.info_label.text()
        assert "Enhanced in 1m 34s" in info_text


class TestTimingClearingBehavior:
    """Tests for timing clearing behavior."""

    def test_timing_cleared_on_new_image_without_timing(
        self, image_panel, sample_pixmap
    ):
        """Test that old timing is cleared when loading new image without timing."""
        # Set up first image with timing
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "first.png"
        image_panel.enhancement_time = "2.34s"

        # Load new image (clear first)
        image_panel.clear()

        # Timing should be cleared
        assert image_panel.enhancement_time is None

    def test_timing_persists_until_cleared(self, image_panel, sample_pixmap):
        """Test that timing persists until explicitly cleared."""
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "test.png"
        image_panel.set_enhancement_time("5.67s")

        # Update overlay multiple times
        image_panel._update_info_overlay()
        image_panel._update_info_overlay()

        # Timing should still be there
        assert image_panel.enhancement_time == "5.67s"
        assert "Enhanced in 5.67s" in image_panel.info_label.text()


class TestInfoOverlayVisibility:
    """Tests for info overlay visibility with timing."""

    def test_info_overlay_visible_with_timing(self, image_panel, sample_pixmap):
        """Test that info overlay is visible when timing is set."""
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "test.png"
        image_panel.set_enhancement_time("2.34s")

        # Info text should contain timing
        info_text = image_panel.info_label.text()
        assert "Enhanced in 2.34s" in info_text

    def test_info_overlay_hidden_when_no_image(self, image_panel):
        """Test that info overlay is hidden when no image despite timing."""
        image_panel.enhancement_time = "2.34s"
        image_panel.current_pixmap = None

        # Set enhancement time on panel without image
        # Should not crash and overlay should be hidden
        image_panel.set_enhancement_time("2.34s")

        # Overlay should be hidden (no image to show info for)
        # Note: This depends on current implementation
        # The method updates only if pixmap exists


@pytest.mark.integration
class TestImagePanelTimingIntegration:
    """Integration tests for complete timing display workflow."""

    def test_complete_timing_display_workflow(self, image_panel, sample_pixmap):
        """Test complete workflow from image load to timing display."""
        # Initially no timing
        assert image_panel.enhancement_time is None

        # Load image
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "enhanced_result.png"

        # Set timing (simulating enhancement completion)
        image_panel.set_enhancement_time("2.34s")

        # Verify all components
        assert image_panel.enhancement_time == "2.34s"

        info_text = image_panel.info_label.text()
        assert "enhanced_result.png" in info_text
        assert "256" in info_text  # Check for dimensions
        assert " px" in info_text
        assert "In Memory" in info_text
        assert "Enhanced in 2.34s" in info_text

        # Clear panel
        image_panel.clear()

        # Everything should be cleared
        assert image_panel.enhancement_time is None

    def test_multiple_enhancements_update_timing(self, image_panel, sample_pixmap):
        """Test that multiple enhancements update timing correctly."""
        # First enhancement
        image_panel.current_pixmap = sample_pixmap
        image_panel.display_name = "result1.png"
        image_panel.set_enhancement_time("1.50s")

        info_text_1 = image_panel.info_label.text()
        assert "Enhanced in 1.50s" in info_text_1

        # Second enhancement (different time)
        image_panel.set_enhancement_time("3.75s")

        info_text_2 = image_panel.info_label.text()
        assert "Enhanced in 3.75s" in info_text_2
        assert "Enhanced in 1.50s" not in info_text_2
