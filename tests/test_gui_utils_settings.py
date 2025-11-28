"""Tests for AppSettings utility module.

Tests application settings management including:
- Model settings
- Recent files
- Window geometry
- Display settings
- Performance settings
"""

from pathlib import Path

import pytest
from PyQt6.QtCore import QPoint, QSettings, QSize

from gui.utils.settings import AppSettings


@pytest.fixture
def settings():
    """Provide a fresh AppSettings instance for each test.

    Note: Uses a test organization name to avoid polluting real settings.
    """
    # Use test settings to avoid overwriting real app settings
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    settings = AppSettings()
    settings.settings = QSettings("ZeroDCE_Test", "GUI_Test")

    # Clear any existing settings
    settings.reset_to_defaults()

    yield settings

    # Cleanup
    settings.reset_to_defaults()


class TestModelSettings:
    """Tests for model-related settings."""

    def test_default_weights_directory(self, settings):
        """Test default weights directory."""
        directory = settings.get_weights_directory()
        assert directory == "./weights"

    def test_set_weights_directory(self, settings):
        """Test setting custom weights directory."""
        settings.set_weights_directory("/custom/path/weights")
        assert settings.get_weights_directory() == "/custom/path/weights"

    def test_default_model_file(self, settings):
        """Test default model filename."""
        filename = settings.get_default_model_file()
        assert filename == "zero_dce.weights.h5"

    def test_set_default_model_file(self, settings):
        """Test setting custom model filename."""
        settings.set_default_model_file("custom_model.h5")
        assert settings.get_default_model_file() == "custom_model.h5"

    def test_get_full_model_path(self, settings):
        """Test constructing full model path."""
        path = settings.get_full_model_path()
        assert "weights" in path
        assert "zero_dce.weights.h5" in path

    def test_auto_load_model_default(self, settings):
        """Test that auto-load is enabled by default."""
        assert settings.get_auto_load_model() is True

    def test_set_auto_load_model(self, settings):
        """Test enabling auto-load."""
        settings.set_auto_load_model(True)
        assert settings.get_auto_load_model() is True

    def test_enhancement_iterations_default(self, settings):
        """Test default enhancement iterations."""
        iterations = settings.get_enhancement_iterations()
        assert iterations == 8

    def test_set_enhancement_iterations(self, settings):
        """Test setting custom iterations."""
        settings.set_enhancement_iterations(16)
        assert settings.get_enhancement_iterations() == 16


class TestRecentFiles:
    """Tests for recent files management."""

    def test_get_recent_files_empty(self, settings):
        """Test that recent files is empty initially."""
        recent = settings.get_recent_files()
        assert recent == []

    def test_add_recent_file(self, settings):
        """Test adding a file to recent files."""
        settings.add_recent_file("/path/to/file1.png")
        recent = settings.get_recent_files()
        assert len(recent) == 1
        assert recent[0] == "/path/to/file1.png"

    def test_add_multiple_recent_files(self, settings):
        """Test adding multiple files."""
        settings.add_recent_file("/path/to/file1.png")
        settings.add_recent_file("/path/to/file2.png")
        settings.add_recent_file("/path/to/file3.png")

        recent = settings.get_recent_files()
        assert len(recent) == 3
        # Most recent should be first
        assert recent[0] == "/path/to/file3.png"

    def test_recent_files_max_limit(self, settings):
        """Test that recent files respects max limit."""
        # Add more than MAX_RECENT_FILES (10)
        for i in range(15):
            settings.add_recent_file(f"/path/to/file{i}.png")

        recent = settings.get_recent_files()
        assert len(recent) == settings.MAX_RECENT_FILES

    def test_recent_file_moves_to_top(self, settings):
        """Test that reopening a file moves it to top."""
        settings.add_recent_file("/path/to/file1.png")
        settings.add_recent_file("/path/to/file2.png")
        settings.add_recent_file("/path/to/file1.png")  # Add again

        recent = settings.get_recent_files()
        assert recent[0] == "/path/to/file1.png"
        assert len(recent) == 2  # Should not duplicate

    def test_clear_recent_files(self, settings):
        """Test clearing recent files."""
        settings.add_recent_file("/path/to/file1.png")
        settings.add_recent_file("/path/to/file2.png")

        settings.clear_recent_files()
        recent = settings.get_recent_files()
        assert recent == []


class TestWindowGeometry:
    """Tests for window geometry settings."""

    def test_get_window_geometry_none_initially(self, settings):
        """Test that window geometry is None initially."""
        pos, size = settings.get_window_geometry()
        assert pos is None
        assert size is None

    def test_set_window_geometry(self, settings):
        """Test setting window geometry."""
        test_pos = QPoint(100, 100)
        test_size = QSize(800, 600)

        settings.set_window_geometry(test_pos, test_size)

        pos, size = settings.get_window_geometry()
        assert pos == test_pos
        assert size == test_size


class TestDisplaySettings:
    """Tests for display settings."""

    def test_default_zoom_mode(self, settings):
        """Test default zoom mode."""
        mode = settings.get_default_zoom_mode()
        assert mode == "fit"

    def test_set_zoom_mode(self, settings):
        """Test setting zoom mode."""
        settings.set_default_zoom_mode("actual")
        assert settings.get_default_zoom_mode() == "actual"

    def test_set_invalid_zoom_mode(self, settings):
        """Test that invalid zoom mode raises ValueError."""
        with pytest.raises(ValueError):
            settings.set_default_zoom_mode("invalid")

    def test_sync_zoom_default(self, settings):
        """Test default sync zoom setting."""
        assert settings.get_sync_zoom() is True

    def test_set_sync_zoom(self, settings):
        """Test setting sync zoom."""
        settings.set_sync_zoom(False)
        assert settings.get_sync_zoom() is False

    def test_show_info_overlay_default(self, settings):
        """Test default info overlay setting."""
        assert settings.get_show_info_overlay() is True

    def test_set_show_info_overlay(self, settings):
        """Test setting info overlay visibility."""
        settings.set_show_info_overlay(False)
        assert settings.get_show_info_overlay() is False


class TestPerformanceSettings:
    """Tests for performance settings."""

    def test_default_gpu_mode(self, settings):
        """Test default GPU mode."""
        mode = settings.get_gpu_mode()
        assert mode == "auto"

    def test_set_gpu_mode(self, settings):
        """Test setting GPU mode."""
        settings.set_gpu_mode("enable")
        assert settings.get_gpu_mode() == "enable"

    def test_set_invalid_gpu_mode(self, settings):
        """Test that invalid GPU mode raises ValueError."""
        with pytest.raises(ValueError):
            settings.set_gpu_mode("invalid")

    def test_default_max_image_dimension(self, settings):
        """Test default max image dimension."""
        dimension = settings.get_max_image_dimension()
        assert dimension == 4096

    def test_set_max_image_dimension(self, settings):
        """Test setting max image dimension."""
        settings.set_max_image_dimension(2048)
        assert settings.get_max_image_dimension() == 2048


class TestOutputSettings:
    """Tests for output settings."""

    def test_default_output_format(self, settings):
        """Test default output format."""
        format = settings.get_output_format()
        assert format == "PNG"

    def test_set_output_format(self, settings):
        """Test setting output format."""
        settings.set_output_format("JPEG")
        assert settings.get_output_format() == "JPEG"

    def test_set_invalid_output_format(self, settings):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            settings.set_output_format("BMP")

    def test_default_jpeg_quality(self, settings):
        """Test default JPEG quality."""
        quality = settings.get_jpeg_quality()
        assert quality == 95

    def test_set_jpeg_quality(self, settings):
        """Test setting JPEG quality."""
        settings.set_jpeg_quality(80)
        assert settings.get_jpeg_quality() == 80

    def test_set_invalid_jpeg_quality(self, settings):
        """Test that invalid quality raises ValueError."""
        with pytest.raises(ValueError):
            settings.set_jpeg_quality(150)  # > 100

        with pytest.raises(ValueError):
            settings.set_jpeg_quality(-10)  # < 0


class TestAdvancedSettings:
    """Tests for advanced settings."""

    def test_get_cache_directory(self, settings):
        """Test getting cache directory."""
        cache_dir = settings.get_cache_directory()
        assert "zero-dce-gui" in cache_dir.lower()

    def test_set_cache_directory(self, settings):
        """Test setting cache directory."""
        settings.set_cache_directory("/custom/cache")
        assert settings.get_cache_directory() == "/custom/cache"

    def test_default_auto_clear_cache(self, settings):
        """Test default auto-clear cache setting."""
        assert settings.get_auto_clear_cache() is True

    def test_set_auto_clear_cache(self, settings):
        """Test setting auto-clear cache."""
        settings.set_auto_clear_cache(False)
        assert settings.get_auto_clear_cache() is False

    def test_default_debug_logging(self, settings):
        """Test default debug logging setting."""
        assert settings.get_debug_logging() is False

    def test_set_debug_logging(self, settings):
        """Test setting debug logging."""
        settings.set_debug_logging(True)
        assert settings.get_debug_logging() is True

    def test_get_log_directory(self, settings):
        """Test getting log directory."""
        log_dir = settings.get_log_directory()
        assert "zero-dce-gui" in log_dir.lower()

    def test_set_log_directory(self, settings):
        """Test setting log directory."""
        settings.set_log_directory("/custom/logs")
        assert settings.get_log_directory() == "/custom/logs"


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_reset_to_defaults(self, settings):
        """Test resetting all settings to defaults."""
        # Change some settings
        settings.set_weights_directory("/custom/path")
        settings.add_recent_file("/test/file.png")
        settings.set_jpeg_quality(80)

        # Reset
        settings.reset_to_defaults()

        # Verify defaults restored
        assert settings.get_weights_directory() == "./weights"
        assert settings.get_recent_files() == []
        assert settings.get_jpeg_quality() == 95

    def test_sync(self, settings):
        """Test that sync doesn't crash."""
        settings.set_weights_directory("/test")
        settings.sync()  # Should not raise


@pytest.mark.unit
class TestSettingsPersistence:
    """Tests for settings persistence."""

    def test_settings_persist_across_instances(self):
        """Test that settings persist across AppSettings instances."""
        # Create first instance and set value
        settings1 = AppSettings()
        settings1.settings = QSettings("ZeroDCE_PersistTest", "GUI_PersistTest")
        settings1.set_weights_directory("/test/persist")
        settings1.sync()

        # Create second instance and check value
        settings2 = AppSettings()
        settings2.settings = QSettings("ZeroDCE_PersistTest", "GUI_PersistTest")
        assert settings2.get_weights_directory() == "/test/persist"

        # Cleanup
        settings2.reset_to_defaults()
