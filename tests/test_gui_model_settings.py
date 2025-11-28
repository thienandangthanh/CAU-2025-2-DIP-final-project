"""Tests for model settings persistence.

Tests cover:
- Default model settings persistence using AppSettings
- Loading from Default Weights submenu saves as default
- Settings are properly synced
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PyQt6.QtCore import QSettings

from gui.main_window import MainWindow
from gui.utils.model_loader import ModelLoader
from gui.utils.settings import AppSettings


@pytest.fixture
def app_settings():
    """Provide a fresh AppSettings instance with unique organization name."""
    # Use a unique name to avoid conflicts with other tests
    settings = AppSettings()
    settings.settings = QSettings("ZeroDCE-Test", "GUI-ModelSettings")
    settings.settings.clear()
    yield settings
    settings.settings.clear()


@pytest.fixture
def main_window(qtbot, app_settings):
    """Provide a MainWindow instance with test settings."""
    with patch.object(MainWindow, "_auto_load_model"):
        window = MainWindow()
        # Replace settings with test settings
        window.settings = app_settings
        qtbot.addWidget(window)
        return window


class TestModelSettingsPersistence:
    """Tests for model settings persistence."""

    def test_get_full_model_path_constructs_correct_path(self, app_settings):
        """Test that get_full_model_path constructs the correct path."""
        app_settings.set_weights_directory("./weights")
        app_settings.set_default_model_file("model.h5")

        full_path = app_settings.get_full_model_path()
        expected_path = str(Path("./weights") / "model.h5")
        assert full_path == expected_path

    def test_set_default_model_file_persists(self, app_settings):
        """Test that set_default_model_file persists the setting."""
        app_settings.set_default_model_file("custom_model.h5")
        app_settings.sync()

        # Get the value back
        filename = app_settings.get_default_model_file()
        assert filename == "custom_model.h5"

    def test_set_weights_directory_persists(self, app_settings):
        """Test that set_weights_directory persists the setting."""
        app_settings.set_weights_directory("/custom/path/weights")
        app_settings.sync()

        # Get the value back
        directory = app_settings.get_weights_directory()
        assert directory == "/custom/path/weights"

    def test_default_values_returned_when_not_set(self, app_settings):
        """Test that default values are returned when settings not set."""
        # Clear all settings
        app_settings.reset_to_defaults()

        # Check defaults
        assert app_settings.get_weights_directory() == "./weights"
        assert app_settings.get_default_model_file() == "zero_dce.weights.h5"
        assert app_settings.get_full_model_path() == str(
            Path("./weights") / "zero_dce.weights.h5"
        )


class TestLoadModelWeightsWithDefaultSave:
    """Tests for loading model weights with save_as_default parameter."""

    def test_load_model_weights_without_save_default(self, main_window, tmp_path):
        """Test that loading model without save_as_default doesn't change settings."""
        # Create a test weight file
        weight_file = tmp_path / "test_model.h5"
        weight_file.write_bytes(b"dummy")

        # Set initial default
        main_window.settings.set_default_model_file("original.h5")
        main_window.settings.set_weights_directory("./weights")
        main_window.settings.sync()

        # Mock the model loader
        with patch.object(main_window.model_loader, "load_model"):
            with patch.object(
                ModelLoader, "validate_weights_file", return_value=(True, "")
            ):
                # Load without save_as_default
                main_window._load_model_weights(str(weight_file), save_as_default=False)

                # Settings should not change
                assert main_window.settings.get_default_model_file() == "original.h5"
                assert main_window.settings.get_weights_directory() == "./weights"

    def test_load_model_weights_with_save_default(self, main_window, tmp_path):
        """Test that loading model with save_as_default=True updates settings."""
        # Create a test weight file
        weights_dir = tmp_path / "custom_weights"
        weights_dir.mkdir()
        weight_file = weights_dir / "new_model.h5"
        weight_file.write_bytes(b"dummy")

        # Set initial default
        main_window.settings.set_default_model_file("original.h5")
        main_window.settings.set_weights_directory("./weights")
        main_window.settings.sync()

        # Mock the model loader
        with patch.object(main_window.model_loader, "load_model"):
            with patch.object(
                ModelLoader, "validate_weights_file", return_value=(True, "")
            ):
                # Load with save_as_default=True
                main_window._load_model_weights(str(weight_file), save_as_default=True)

                # Settings should update
                assert main_window.settings.get_default_model_file() == "new_model.h5"
                assert main_window.settings.get_weights_directory() == str(weights_dir)

    def test_load_from_default_weights_menu_saves_as_default(
        self, main_window, tmp_path, qtbot
    ):
        """Test that loading from Default Weights submenu saves as default."""
        # Create test weight files
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        weight_file = weights_dir / "menu_model.h5"
        weight_file.write_bytes(b"dummy")

        # Set different initial default
        main_window.settings.set_default_model_file("original.h5")
        main_window.settings.set_weights_directory("./weights")

        # Mock model loading
        with patch.object(main_window.model_loader, "load_model"):
            with patch.object(
                ModelLoader, "validate_weights_file", return_value=(True, "")
            ):
                with patch.object(
                    ModelLoader,
                    "list_available_weights",
                    return_value=[str(weight_file)],
                ):
                    # Update menu
                    main_window._update_default_weights_menu()

                    # Get the menu action
                    actions = main_window.default_weights_menu.actions()
                    assert len(actions) == 1

                    # Trigger the action (simulates clicking in menu)
                    actions[0].trigger()

                    # Wait for any queued events to process
                    qtbot.wait(100)

                    # Verify settings were updated
                    assert (
                        main_window.settings.get_default_model_file() == "menu_model.h5"
                    )
                    assert main_window.settings.get_weights_directory() == str(
                        weights_dir
                    )


class TestSettingsSync:
    """Tests for settings synchronization."""

    def test_settings_sync_called_after_save_default(self, main_window, tmp_path):
        """Test that settings.sync() is called after saving default."""
        weight_file = tmp_path / "model.h5"
        weight_file.write_bytes(b"dummy")

        # Mock settings.sync to track calls
        with patch.object(main_window.settings, "sync") as mock_sync:
            with patch.object(main_window.model_loader, "load_model"):
                with patch.object(
                    ModelLoader, "validate_weights_file", return_value=(True, "")
                ):
                    main_window._load_model_weights(
                        str(weight_file), save_as_default=True
                    )

                    # Verify sync was called
                    mock_sync.assert_called_once()

    def test_settings_not_synced_when_not_saving_default(self, main_window, tmp_path):
        """Test that settings.sync() is not called when not saving as default."""
        weight_file = tmp_path / "model.h5"
        weight_file.write_bytes(b"dummy")

        # Track initial sync call count
        with patch.object(main_window.settings, "sync") as mock_sync:
            with patch.object(main_window.model_loader, "load_model"):
                with patch.object(
                    ModelLoader, "validate_weights_file", return_value=(True, "")
                ):
                    main_window._load_model_weights(
                        str(weight_file), save_as_default=False
                    )

                    # Verify sync was not called
                    mock_sync.assert_not_called()


# Mark tests appropriately
pytestmark = [pytest.mark.gui, pytest.mark.unit]
