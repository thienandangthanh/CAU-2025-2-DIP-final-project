"""Tests for Model Menu.

Tests cover:
- Default Weights submenu population
- Loading models from Default Weights submenu
- Epoch extraction from filenames
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from gui.main_window import MainWindow
from gui.utils.model_loader import ModelLoader


@pytest.fixture
def app(qapp):
    """Provide QApplication instance (qapp fixture from pytest-qt)."""
    return qapp


@pytest.fixture
def main_window(qtbot):
    """Provide a MainWindow instance for testing."""
    # Mock auto-load to prevent loading model on startup
    with patch.object(MainWindow, "_auto_load_model"):
        window = MainWindow()
        qtbot.addWidget(window)
        return window


@pytest.fixture
def sample_model_info():
    """Provide sample model information for testing."""
    return {
        "loaded": True,
        "weights_path": "/path/to/weights/zero_dce_epoch100.weights.h5",
        "filename": "zero_dce_epoch100.weights.h5",
        "file_size": 1024 * 1024 * 2,  # 2 MB
    }


class TestDefaultWeightsSubmenu:
    """Tests for Default Weights submenu functionality."""

    def test_default_weights_menu_exists(self, main_window):
        """Test that Default Weights submenu is created in Model menu."""
        # Find Model menu
        menubar = main_window.menuBar()
        model_menu = None
        for action in menubar.actions():
            if action.text() == "&Model":
                model_menu = action.menu()
                break

        assert model_menu is not None, "Model menu not found"

        # Find Default Weights submenu
        default_weights_menu = None
        for action in model_menu.actions():
            if hasattr(action, "menu") and action.menu() is not None:
                if "Default Weights" in action.text():
                    default_weights_menu = action.menu()
                    break

        assert default_weights_menu is not None, "Default Weights submenu not found"
        assert main_window.default_weights_menu is not None

    def test_empty_weights_directory_shows_message(self, main_window):
        """Test that empty weights directory shows '(No weights found)' message."""
        # Mock list_available_weights to return empty list
        with patch.object(ModelLoader, "list_available_weights", return_value=[]):
            main_window._update_default_weights_menu()

            # Check menu has one disabled action with text "(No weights found)"
            actions = main_window.default_weights_menu.actions()
            assert len(actions) == 1
            assert actions[0].text() == "(No weights found)"
            assert not actions[0].isEnabled()

    def test_weights_directory_with_files_populates_menu(self, main_window, tmp_path):
        """Test that available weights are listed in the submenu."""
        # Create mock weight files
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()

        weight_files = [
            weights_dir / "zero_dce_epoch100.weights.h5",
            weights_dir / "zero_dce_epoch200.weights.h5",
            weights_dir / "model.h5",
        ]

        for wf in weight_files:
            wf.write_bytes(b"dummy")

        # Mock list_available_weights to return these files
        with patch.object(
            ModelLoader,
            "list_available_weights",
            return_value=[str(f) for f in weight_files],
        ):
            main_window._update_default_weights_menu()

            # Check menu has actions for each weight file
            actions = main_window.default_weights_menu.actions()
            assert len(actions) == len(weight_files)

            # Check filenames are in menu
            action_texts = [a.text() for a in actions]
            assert any("zero_dce_epoch100.weights.h5" in text for text in action_texts)
            assert any("zero_dce_epoch200.weights.h5" in text for text in action_texts)
            assert any("model.h5" in text for text in action_texts)

    def test_epoch_displayed_in_menu_item(self, main_window, tmp_path):
        """Test that training epoch is extracted and displayed in menu item."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        weight_file = weights_dir / "zero_dce_epoch100.weights.h5"
        weight_file.write_bytes(b"dummy")

        with patch.object(
            ModelLoader, "list_available_weights", return_value=[str(weight_file)]
        ):
            main_window._update_default_weights_menu()

            actions = main_window.default_weights_menu.actions()
            assert len(actions) == 1
            # Should include "Epoch 100" in the text
            assert "Epoch 100" in actions[0].text()

    def test_currently_loaded_model_is_marked(self, main_window, tmp_path):
        """Test that the currently loaded model has a checkmark."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        weight_file1 = weights_dir / "model1.h5"
        weight_file2 = weights_dir / "model2.h5"
        weight_file1.write_bytes(b"dummy")
        weight_file2.write_bytes(b"dummy")

        # Mock model loader to have model1 loaded with absolute path
        main_window.model_loader.current_weights_path = str(weight_file1.resolve())
        main_window.model_loader.current_model = Mock()

        with patch.object(
            ModelLoader,
            "list_available_weights",
            return_value=[str(weight_file1), str(weight_file2)],
        ):
            main_window._update_default_weights_menu()

            actions = main_window.default_weights_menu.actions()
            assert len(actions) == 2

            # First action (model1) should be checked
            assert actions[0].isCheckable()
            assert actions[0].isChecked()

            # Second action (model2) should not be checked
            assert not actions[1].isCheckable() or not actions[1].isChecked()

    def test_clicking_weight_loads_model(self, main_window, tmp_path, qtbot):
        """Test that clicking a weight file in submenu loads the model and saves as default."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        weight_file = weights_dir / "model.h5"
        weight_file.write_bytes(b"dummy")

        # Mock _load_model_weights to track if it was called
        with patch.object(main_window, "_load_model_weights") as mock_load:
            with patch.object(
                ModelLoader, "list_available_weights", return_value=[str(weight_file)]
            ):
                main_window._update_default_weights_menu()

                actions = main_window.default_weights_menu.actions()
                assert len(actions) == 1

                # Trigger the action
                actions[0].trigger()

                # Verify _load_model_weights was called with save_as_default=True
                mock_load.assert_called_once_with(str(weight_file), save_as_default=True)

    def test_checkmark_with_relative_vs_absolute_paths(self, main_window, tmp_path):
        """Test that checkmark works even with relative vs absolute path differences."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        weight_file = weights_dir / "model.h5"
        weight_file.write_bytes(b"dummy")

        # Mock model loader with absolute path (as it would be stored after loading)
        absolute_path = str(weight_file.resolve())
        main_window.model_loader.current_weights_path = absolute_path
        main_window.model_loader.current_model = Mock()

        # list_available_weights might return relative or absolute paths
        # Test both cases
        for path_to_use in [str(weight_file), absolute_path]:
            with patch.object(
                ModelLoader, "list_available_weights", return_value=[path_to_use]
            ):
                main_window._update_default_weights_menu()

                actions = main_window.default_weights_menu.actions()
                assert len(actions) == 1

                # Action should be checked regardless of relative/absolute path
                assert actions[0].isCheckable(), f"Failed for path: {path_to_use}"
                assert actions[0].isChecked(), f"Failed for path: {path_to_use}"


class TestEpochExtraction:
    """Tests for epoch extraction from filenames."""

    def test_extract_epoch_from_filename_various_formats(self, main_window):
        """Test epoch extraction from various filename formats."""
        test_cases = [
            ("zero_dce_epoch100.weights.h5", "Epoch 100"),
            ("model_epoch_200.h5", "Epoch 200"),
            ("model_epoch-50.weights.h5", "Epoch 50"),
            ("modele100.h5", "Epoch 100"),
            ("model_E200.h5", "Epoch 200"),
            ("model.h5", ""),  # No epoch
            ("", ""),  # Empty filename
        ]

        for filename, expected_epoch in test_cases:
            result = main_window._extract_epoch_from_filename(filename)
            assert result == expected_epoch, (
                f"Failed for {filename}: expected '{expected_epoch}', got '{result}'"
            )


class TestModelInfoIntegration:
    """Integration tests for Model Info with MainWindow."""

    def test_show_model_info_with_no_model_loaded(self, main_window, qtbot):
        """Test that showing model info with no model shows appropriate message."""
        # Ensure no model is loaded
        main_window.model_loader.current_model = None

        # Patch QMessageBox to capture the call
        with patch("gui.main_window.QMessageBox.information") as mock_msgbox:
            main_window._show_model_info()

            # Should show message box saying no model is loaded
            mock_msgbox.assert_called_once()
            args = mock_msgbox.call_args[0]
            assert "No model is currently loaded" in args[2]

    def test_show_model_info_with_model_loaded(
        self, main_window, qtbot, sample_model_info
    ):
        """Test that showing model info with loaded model shows message box."""
        # Mock model loader to return sample info
        main_window.model_loader.current_model = Mock()
        with patch.object(
            main_window.model_loader, "get_model_info", return_value=sample_model_info
        ):
            # Patch QMessageBox to track creation
            with patch("gui.main_window.QMessageBox.information") as mock_msgbox:
                main_window._show_model_info()

                # Message box should be called
                mock_msgbox.assert_called_once()
                args = mock_msgbox.call_args[0]
                # Should contain model info
                assert "Model Information" in args[1] or "Model Info" in args[1]
                assert sample_model_info["filename"] in args[2]


class TestModelMenuUpdates:
    """Tests for model menu updates after loading models."""

    def test_default_weights_menu_updated_after_model_load(self, main_window, tmp_path):
        """Test that Default Weights menu is updated after loading a model."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        weight_file = weights_dir / "model.h5"
        weight_file.write_bytes(b"dummy")

        # Mock settings and list_available_weights
        with patch.object(
            main_window.settings, "get_weights_directory", return_value=str(weights_dir)
        ):
            with patch.object(
                ModelLoader, "list_available_weights", return_value=[str(weight_file)]
            ):
                with patch.object(main_window.model_loader, "load_model") as mock_load:
                    with patch.object(
                        ModelLoader, "validate_weights_file", return_value=(True, "")
                    ):
                        # Track _update_default_weights_menu calls
                        with patch.object(
                            main_window, "_update_default_weights_menu"
                        ) as mock_update:
                            main_window._load_model_weights(str(weight_file))

                            # Should call _update_default_weights_menu
                            mock_update.assert_called()


# Mark tests appropriately
pytestmark = [pytest.mark.gui, pytest.mark.unit]
