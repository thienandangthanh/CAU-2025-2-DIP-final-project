"""Tests for MethodSelectionDialog.

This module tests the method selection dialog functionality including:
- Method checkbox creation
- Quick selection buttons
- Reference image selection
- Method selection persistence
"""

from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from gui.dialogs.method_selection_dialog import MethodSelectionDialog
from gui.utils.enhancement_methods import get_registry


@pytest.fixture
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_reference_image(tmp_path):
    """Create a test reference image file."""
    from PyQt6.QtGui import QPixmap

    pixmap = QPixmap(100, 100)
    pixmap.fill(Qt.GlobalColor.white)

    image_path = tmp_path / "reference.png"
    pixmap.save(str(image_path))

    return str(image_path)


class TestMethodSelectionDialog:
    """Tests for MethodSelectionDialog."""

    def test_init_without_model(self, qapp):
        """Test dialog initialization without model loaded."""
        dialog = MethodSelectionDialog(model_loaded=False)

        assert dialog.model_loaded is False
        assert dialog.selected_methods == []
        assert dialog.reference_path is None
        assert len(dialog.method_checkboxes) > 0

    def test_init_with_model(self, qapp):
        """Test dialog initialization with model loaded."""
        dialog = MethodSelectionDialog(model_loaded=True)

        assert dialog.model_loaded is True

    def test_init_with_current_selection(self, qapp):
        """Test dialog initialization with current selection."""
        current_selection = ["clahe", "gamma"]
        dialog = MethodSelectionDialog(
            model_loaded=False, current_selection=current_selection
        )

        assert dialog.selected_methods == current_selection

        # Check that checkboxes are checked
        assert dialog.method_checkboxes["clahe"].isChecked()
        assert dialog.method_checkboxes["gamma"].isChecked()

    def test_init_with_reference_path(self, qapp, test_reference_image):
        """Test dialog initialization with reference path."""
        dialog = MethodSelectionDialog(
            model_loaded=False, reference_path=test_reference_image
        )

        assert dialog.reference_path == test_reference_image
        assert test_reference_image in dialog.reference_line_edit.text()

    def test_all_methods_have_checkboxes(self, qapp):
        """Test that all registered methods have checkboxes."""
        dialog = MethodSelectionDialog(model_loaded=True)

        registry = get_registry()
        available_methods = registry.get_available_methods()

        # All methods should have checkboxes
        for method_key in available_methods:
            assert method_key in dialog.method_checkboxes

    def test_zero_dce_disabled_without_model(self, qapp):
        """Test that Zero-DCE is disabled when model is not loaded."""
        dialog = MethodSelectionDialog(model_loaded=False)

        # Zero-DCE checkbox should be disabled
        if "zero-dce" in dialog.method_checkboxes:
            assert not dialog.method_checkboxes["zero-dce"].isEnabled()

    def test_zero_dce_enabled_with_model(self, qapp):
        """Test that Zero-DCE is enabled when model is loaded."""
        dialog = MethodSelectionDialog(model_loaded=True)

        # Zero-DCE checkbox should be enabled
        if "zero-dce" in dialog.method_checkboxes:
            assert dialog.method_checkboxes["zero-dce"].isEnabled()

    def test_classical_methods_always_enabled(self, qapp):
        """Test that classical methods are always enabled."""
        dialog = MethodSelectionDialog(model_loaded=False)

        registry = get_registry()
        classical_methods = registry.get_methods_by_category("classical")

        for method_key in classical_methods:
            assert dialog.method_checkboxes[method_key].isEnabled()

    def test_select_all_button(self, qapp):
        """Test Select All button."""
        dialog = MethodSelectionDialog(model_loaded=True)

        # Click Select All
        dialog._select_all()

        # All enabled checkboxes should be checked
        for checkbox in dialog.method_checkboxes.values():
            if checkbox.isEnabled():
                assert checkbox.isChecked()

    def test_select_none_button(self, qapp):
        """Test Select None button."""
        dialog = MethodSelectionDialog(
            model_loaded=True, current_selection=["clahe", "gamma"]
        )

        # Click Select None
        dialog._select_none()

        # All checkboxes should be unchecked
        for checkbox in dialog.method_checkboxes.values():
            assert not checkbox.isChecked()

    def test_select_classical_only_button(self, qapp):
        """Test Classical Only button."""
        dialog = MethodSelectionDialog(model_loaded=True)

        # Click Classical Only
        dialog._select_classical_only()

        registry = get_registry()
        classical_methods = registry.get_methods_by_category("classical")
        dl_methods = registry.get_methods_by_category("deep_learning")

        # Classical methods should be checked
        for method_key in classical_methods:
            if dialog.method_checkboxes[method_key].isEnabled():
                assert dialog.method_checkboxes[method_key].isChecked()

        # Deep learning methods should not be checked
        for method_key in dl_methods:
            assert not dialog.method_checkboxes[method_key].isChecked()

    def test_select_fast_only_button(self, qapp):
        """Test Fast Methods button."""
        dialog = MethodSelectionDialog(model_loaded=True)

        # Click Fast Only
        dialog._select_fast_only()

        from gui.utils.enhancement_methods import ExecutionSpeed

        registry = get_registry()
        fast_methods = registry.get_methods_by_speed(ExecutionSpeed.FAST)

        # Fast methods should be checked
        for method_key in fast_methods:
            if dialog.method_checkboxes[method_key].isEnabled():
                assert dialog.method_checkboxes[method_key].isChecked()

    def test_clear_reference_button(self, qapp, test_reference_image):
        """Test clearing reference image."""
        dialog = MethodSelectionDialog(
            model_loaded=False, reference_path=test_reference_image
        )

        # Clear reference
        dialog._clear_reference()

        assert dialog.reference_path is None
        assert dialog.reference_line_edit.text() == ""

    def test_ok_with_no_selection_shows_warning(self, qapp, monkeypatch):
        """Test that OK with no selection shows warning."""
        dialog = MethodSelectionDialog(model_loaded=True)

        # Deselect all
        dialog._select_none()

        # Mock QMessageBox to prevent actual dialog from showing
        warning_called = []

        def mock_warning(*args, **kwargs):
            warning_called.append(True)

        monkeypatch.setattr(QMessageBox, "warning", mock_warning)

        # Try to click OK (should show warning)
        dialog._on_ok()

        # Verify warning was called
        assert len(warning_called) == 1

        # Dialog should not be accepted (result should not be Accepted)
        assert dialog.result() != QDialog.DialogCode.Accepted

    def test_get_selected_methods(self, qapp):
        """Test getting selected methods."""
        dialog = MethodSelectionDialog(
            model_loaded=False, current_selection=["clahe", "gamma"]
        )

        selected = dialog.get_selected_methods()

        assert "clahe" in selected
        assert "gamma" in selected

    def test_get_reference_path(self, qapp, test_reference_image):
        """Test getting reference path."""
        dialog = MethodSelectionDialog(
            model_loaded=False, reference_path=test_reference_image
        )

        ref_path = dialog.get_reference_path()

        assert ref_path == test_reference_image


@pytest.mark.gui
class TestMethodSelectionDialogIntegration:
    """Integration tests for MethodSelectionDialog."""

    def test_dialog_displays_correctly(self, qapp):
        """Test that dialog displays all UI elements correctly."""
        dialog = MethodSelectionDialog(model_loaded=True)

        # Check that dialog has required elements
        assert dialog.windowTitle() == "Select Enhancement Methods"
        assert dialog.isModal()

        # Check that all method groups are created
        registry = get_registry()
        dl_methods = registry.get_methods_by_category("deep_learning")
        classical_methods = registry.get_methods_by_category("classical")

        # Should have checkboxes for all methods
        assert len(dialog.method_checkboxes) == len(dl_methods) + len(classical_methods)

    def test_selection_persists_through_quick_buttons(self, qapp):
        """Test that manual selection works with quick buttons."""
        dialog = MethodSelectionDialog(model_loaded=True)

        # Manually check one method
        if "clahe" in dialog.method_checkboxes:
            dialog.method_checkboxes["clahe"].setChecked(True)

        # Click Classical Only
        dialog._select_classical_only()

        # CLAHE should still be checked (it's classical)
        if "clahe" in dialog.method_checkboxes:
            assert dialog.method_checkboxes["clahe"].isChecked()

    def test_reference_image_workflow(self, qapp, test_reference_image):
        """Test complete reference image selection workflow."""
        dialog = MethodSelectionDialog(model_loaded=False)

        # Initially no reference
        assert dialog.reference_path is None

        # Simulate setting reference (would normally be from file dialog)
        dialog.reference_path = test_reference_image
        dialog.reference_line_edit.setText(test_reference_image)

        # Verify reference is set
        assert dialog.get_reference_path() == test_reference_image

        # Clear reference
        dialog._clear_reference()

        # Verify cleared
        assert dialog.get_reference_path() is None

    def test_method_tooltips(self, qapp):
        """Test that method checkboxes have tooltips with descriptions."""
        dialog = MethodSelectionDialog(model_loaded=True)

        registry = get_registry()

        for method_key, checkbox in dialog.method_checkboxes.items():
            method_info = registry.get_method_info(method_key)

            # Tooltip should contain description
            assert method_info.description in checkbox.toolTip()


@pytest.mark.unit
class TestMethodSelectionDialogEdgeCases:
    """Edge case tests for MethodSelectionDialog."""

    def test_dialog_with_empty_current_selection(self, qapp):
        """Test dialog with empty current selection list."""
        dialog = MethodSelectionDialog(model_loaded=False, current_selection=[])

        assert dialog.selected_methods == []

        # No checkboxes should be checked
        for checkbox in dialog.method_checkboxes.values():
            assert not checkbox.isChecked()

    def test_dialog_with_invalid_method_in_selection(self, qapp):
        """Test dialog with invalid method key in current selection."""
        # This should not crash, just ignore invalid keys
        dialog = MethodSelectionDialog(
            model_loaded=False,
            current_selection=["clahe", "invalid-method-key", "gamma"],
        )

        # Valid methods should be checked
        if "clahe" in dialog.method_checkboxes:
            assert dialog.method_checkboxes["clahe"].isChecked()
        if "gamma" in dialog.method_checkboxes:
            assert dialog.method_checkboxes["gamma"].isChecked()

    def test_dialog_with_nonexistent_reference_path(self, qapp):
        """Test dialog with nonexistent reference path."""
        dialog = MethodSelectionDialog(
            model_loaded=False, reference_path="/nonexistent/path/to/image.png"
        )

        # Should still set the path (validation happens elsewhere)
        assert dialog.reference_path == "/nonexistent/path/to/image.png"
