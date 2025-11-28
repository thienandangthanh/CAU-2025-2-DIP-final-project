"""Smoke tests for GUI modules.

These tests verify that all GUI modules can be imported without errors.
This is useful for catching basic import and syntax errors.
"""

import pytest


@pytest.mark.unit
class TestGUIImports:
    """Test that all GUI modules can be imported."""

    def test_import_gui_utils(self):
        """Test importing GUI utils modules."""
        from gui.utils import AppSettings, ImageProcessor, ModelLoader

        assert AppSettings is not None
        assert ModelLoader is not None
        assert ImageProcessor is not None

    def test_import_gui_widgets(self):
        """Test importing GUI widgets."""
        from gui.widgets import EnhanceButton, ImagePanel

        assert ImagePanel is not None
        assert EnhanceButton is not None

    def test_import_gui_dialogs(self):
        """Test importing GUI dialogs."""
        from gui.dialogs import ErrorDialog

        assert ErrorDialog is not None

    def test_import_main_window(self):
        """Test importing main window."""
        from gui.main_window import MainWindow

        assert MainWindow is not None


@pytest.mark.unit
class TestGUIConstants:
    """Test that GUI modules have expected constants and attributes."""

    def test_image_processor_constants(self):
        """Test ImageProcessor has required constants."""
        from gui.utils import ImageProcessor

        assert hasattr(ImageProcessor, "MIN_DIMENSION")
        assert hasattr(ImageProcessor, "MAX_DIMENSION")
        assert hasattr(ImageProcessor, "SUPPORTED_FORMATS")

    def test_app_settings_defaults(self):
        """Test AppSettings has default values."""
        from gui.utils import AppSettings

        assert hasattr(AppSettings, "DEFAULT_WEIGHTS_DIR")
        assert hasattr(AppSettings, "DEFAULT_MODEL_FILE")
        assert hasattr(AppSettings, "MAX_RECENT_FILES")


@pytest.mark.unit
class TestModuleDocstrings:
    """Test that all modules have proper docstrings."""

    def test_utils_docstrings(self):
        """Test that utils modules have docstrings."""
        from gui.utils import image_processor, model_loader, settings

        assert model_loader.__doc__ is not None
        assert settings.__doc__ is not None
        assert image_processor.__doc__ is not None

    def test_widgets_docstrings(self):
        """Test that widget modules have docstrings."""
        from gui.widgets import enhance_button, image_panel

        assert image_panel.__doc__ is not None
        assert enhance_button.__doc__ is not None

    def test_dialogs_docstrings(self):
        """Test that dialog modules have docstrings."""
        from gui.dialogs import error_dialog

        assert error_dialog.__doc__ is not None

    def test_main_window_docstring(self):
        """Test that main window has docstring."""
        from gui import main_window

        assert main_window.__doc__ is not None
