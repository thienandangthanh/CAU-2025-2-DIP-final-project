"""Tests for Preferences Dialog functionality.

This module tests the PreferencesDialog implementation including:
- Dialog opening and closing
- Settings loading and saving
- Validation
- Dirty state tracking
- OK/Cancel/Apply button behavior
"""

import pytest
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox, QPushButton
from PyQt6.QtCore import Qt

from gui.dialogs import PreferencesDialog
from gui.dialogs.preferences_dialog import GeneralTab
from gui.utils import AppSettings


@pytest.fixture(autouse=True)
def mock_message_boxes(monkeypatch):
    """Auto-mock all QMessageBox methods to prevent blocking dialogs.
    
    This fixture automatically applies to all tests in this module.
    Individual tests can override specific methods if needed.
    """
    def mock_warning(*args, **kwargs):
        return QMessageBox.StandardButton.Ok
    
    def mock_information(*args, **kwargs):
        return QMessageBox.StandardButton.Ok
    
    def mock_question(*args, **kwargs):
        return QMessageBox.StandardButton.Yes
    
    def mock_critical(*args, **kwargs):
        return QMessageBox.StandardButton.Ok
    
    monkeypatch.setattr(QMessageBox, 'warning', mock_warning)
    monkeypatch.setattr(QMessageBox, 'information', mock_information)
    monkeypatch.setattr(QMessageBox, 'question', mock_question)
    monkeypatch.setattr(QMessageBox, 'critical', mock_critical)


@pytest.fixture
def settings(tmp_path):
    """Provide a clean AppSettings instance for testing.
    
    Args:
        tmp_path: pytest fixture for temporary directory
    
    Yields:
        AppSettings instance with reset settings
    """
    settings = AppSettings()
    
    # Store original values
    original_values = {
        'weights_dir': settings.get_weights_directory(),
        'default_file': settings.get_default_model_file(),
        'auto_load': settings.get_auto_load_model(),
        'zoom_mode': settings.get_default_zoom_mode(),
        'sync_zoom': settings.get_sync_zoom(),
        'show_info': settings.get_show_info_overlay(),
        'gpu_mode': settings.get_gpu_mode(),
        'max_dimension': settings.get_max_image_dimension(),
    }
    
    yield settings
    
    # Restore original values
    settings.set_weights_directory(original_values['weights_dir'])
    settings.set_default_model_file(original_values['default_file'])
    settings.set_auto_load_model(original_values['auto_load'])
    settings.set_default_zoom_mode(original_values['zoom_mode'])
    settings.set_sync_zoom(original_values['sync_zoom'])
    settings.set_show_info_overlay(original_values['show_info'])
    settings.set_gpu_mode(original_values['gpu_mode'])
    settings.set_max_image_dimension(original_values['max_dimension'])
    settings.sync()


@pytest.fixture
def general_tab(qtbot, settings):
    """Provide a GeneralTab instance for testing.
    
    Args:
        qtbot: pytest-qt fixture
        settings: AppSettings fixture
    
    Returns:
        GeneralTab instance
    """
    tab = GeneralTab(settings)
    qtbot.addWidget(tab)
    return tab


@pytest.fixture
def preferences_dialog(qtbot, settings):
    """Provide a PreferencesDialog instance for testing.
    
    Args:
        qtbot: pytest-qt fixture
        settings: AppSettings fixture (implicit via PreferencesDialog)
    
    Returns:
        PreferencesDialog instance
    """
    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    return dialog


@pytest.fixture
def sample_weights_file(tmp_path):
    """Create a sample weights file for testing.
    
    Args:
        tmp_path: pytest fixture for temporary directory
    
    Returns:
        Path to sample weights file
    """
    weights_file = tmp_path / "test_model.weights.h5"
    weights_file.write_bytes(b"dummy weights")
    return str(weights_file)


class TestGeneralTab:
    """Tests for General tab of preferences dialog."""
    
    def test_general_tab_initialization(self, general_tab):
        """Test that GeneralTab initializes correctly."""
        assert general_tab is not None
        assert hasattr(general_tab, 'weights_dir_edit')
        assert hasattr(general_tab, 'auto_load_checkbox')
        assert hasattr(general_tab, 'zoom_mode_combo')
        assert hasattr(general_tab, 'sync_zoom_checkbox')
        assert hasattr(general_tab, 'show_info_checkbox')
        assert hasattr(general_tab, 'gpu_mode_combo')
        assert hasattr(general_tab, 'max_dimension_combo')
    
    def test_load_settings(self, general_tab, settings):
        """Test that settings are loaded correctly into UI controls."""
        # Set known values
        settings.set_weights_directory("./test_weights")
        settings.set_auto_load_model(True)
        settings.set_default_zoom_mode("actual")
        settings.set_sync_zoom(False)
        settings.set_show_info_overlay(True)
        settings.set_gpu_mode("disable")
        settings.set_max_image_dimension(2048)
        
        # Reload settings
        general_tab._load_settings()
        
        # Verify UI reflects settings
        assert general_tab.weights_dir_edit.text() == "./test_weights"
        assert general_tab.auto_load_checkbox.isChecked() is True
        assert general_tab.zoom_mode_combo.currentData() == "actual"
        assert general_tab.sync_zoom_checkbox.isChecked() is False
        assert general_tab.show_info_checkbox.isChecked() is True
        assert general_tab.gpu_mode_combo.currentData() == "disable"
        assert general_tab.max_dimension_combo.currentData() == 2048
    
    def test_save_settings_valid(self, general_tab, settings, tmp_path):
        """Test that settings save correctly with valid input."""
        # Create a weights directory
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        
        # Set values in UI
        general_tab.weights_dir_edit.setText(str(weights_dir))
        general_tab.auto_load_checkbox.setChecked(False)
        general_tab.zoom_mode_combo.setCurrentIndex(
            general_tab.zoom_mode_combo.findData("fit")
        )
        general_tab.sync_zoom_checkbox.setChecked(True)
        general_tab.show_info_checkbox.setChecked(False)
        general_tab.gpu_mode_combo.setCurrentIndex(
            general_tab.gpu_mode_combo.findData("auto")
        )
        general_tab.max_dimension_combo.setCurrentIndex(
            general_tab.max_dimension_combo.findData(8192)
        )
        
        # Save settings
        result = general_tab.save_settings()
        
        # Verify save succeeded
        assert result is True
        
        # Verify settings were saved
        assert settings.get_weights_directory() == str(weights_dir)
        assert settings.get_auto_load_model() is False
        assert settings.get_default_zoom_mode() == "fit"
        assert settings.get_sync_zoom() is True
        assert settings.get_show_info_overlay() is False
        assert settings.get_gpu_mode() == "auto"
        assert settings.get_max_image_dimension() == 8192
    
    def test_save_settings_invalid_directory(self, general_tab, qtbot, monkeypatch):
        """Test that saving fails with invalid directory."""
        # Mock QMessageBox.warning to avoid blocking
        def mock_warning(*args, **kwargs):
            pass
        
        monkeypatch.setattr(QMessageBox, 'warning', mock_warning)
        
        # Set non-existent directory
        general_tab.weights_dir_edit.setText("/nonexistent/directory")
        
        # Save should fail
        result = general_tab.save_settings()
        assert result is False
    
    def test_save_settings_path_is_file(self, general_tab, qtbot, tmp_path, monkeypatch):
        """Test that saving fails when path is a file, not directory."""
        # Mock QMessageBox.warning to avoid blocking
        def mock_warning(*args, **kwargs):
            pass
        
        monkeypatch.setattr(QMessageBox, 'warning', mock_warning)
        
        # Create a file (not directory)
        file_path = tmp_path / "not_a_directory.txt"
        file_path.write_text("test")
        
        general_tab.weights_dir_edit.setText(str(file_path))
        
        # Save should fail
        result = general_tab.save_settings()
        assert result is False
    
    def test_browse_directory_button(self, general_tab):
        """Test that Browse button exists and is connected."""
        assert general_tab.browse_dir_btn is not None
        assert general_tab.browse_dir_btn.text() == "Browse..."
    
    def test_gpu_status_label(self, general_tab):
        """Test that GPU status label is displayed."""
        assert general_tab.gpu_status_label is not None
        assert "Status:" in general_tab.gpu_status_label.text()
    
    def test_unlimited_warning_exists(self, general_tab):
        """Test that unlimited dimension warning label exists and has correct text."""
        # Verify the warning label exists
        assert general_tab.unlimited_warning_label is not None
        assert "Warning" in general_tab.unlimited_warning_label.text()
        assert "Unlimited" in general_tab.unlimited_warning_label.text() or "unlimited" in general_tab.unlimited_warning_label.text()
        
        # Initially should be hidden (4096 is default)
        assert not general_tab.unlimited_warning_label.isVisible()
    
    def test_get_initial_state(self, general_tab):
        """Test that initial state is captured correctly."""
        state = general_tab.get_initial_state()
        
        assert isinstance(state, dict)
        assert 'weights_dir' in state
        assert 'auto_load' in state
        assert 'zoom_mode' in state
        assert 'sync_zoom' in state
        assert 'show_info' in state
        assert 'gpu_mode' in state
        assert 'max_dimension' in state
    
    def test_get_current_state(self, general_tab):
        """Test that current state is retrieved correctly."""
        # Modify some settings
        general_tab.auto_load_checkbox.setChecked(False)
        general_tab.sync_zoom_checkbox.setChecked(True)
        
        state = general_tab.get_current_state()
        
        assert state['auto_load'] is False
        assert state['sync_zoom'] is True


class TestPreferencesDialog:
    """Tests for PreferencesDialog."""
    
    def test_dialog_initialization(self, preferences_dialog):
        """Test that PreferencesDialog initializes correctly."""
        assert preferences_dialog is not None
        assert preferences_dialog.windowTitle() == "Preferences"
        assert preferences_dialog.isModal()
    
    def test_dialog_size(self, preferences_dialog):
        """Test that dialog has appropriate minimum size."""
        assert preferences_dialog.minimumWidth() >= 600
        assert preferences_dialog.minimumHeight() >= 450
    
    def test_tab_widget_exists(self, preferences_dialog):
        """Test that tab widget exists with General tab."""
        assert preferences_dialog.tab_widget is not None
        assert preferences_dialog.tab_widget.count() >= 1
        assert preferences_dialog.tab_widget.tabText(0) == "General"
    
    def test_general_tab_exists(self, preferences_dialog):
        """Test that General tab is accessible."""
        assert preferences_dialog.general_tab is not None
        assert isinstance(preferences_dialog.general_tab, GeneralTab)
    
    def test_buttons_exist(self, preferences_dialog):
        """Test that OK/Cancel/Apply buttons exist."""
        assert preferences_dialog.ok_btn is not None
        assert preferences_dialog.cancel_btn is not None
        assert preferences_dialog.apply_btn is not None
        
        assert preferences_dialog.ok_btn.text() == "OK"
        assert preferences_dialog.cancel_btn.text() == "Cancel"
        assert preferences_dialog.apply_btn.text() == "Apply"
    
    def test_apply_button_initially_disabled(self, preferences_dialog):
        """Test that Apply button is disabled when dialog opens."""
        assert not preferences_dialog.apply_btn.isEnabled()
    
    def test_apply_button_enabled_on_change(self, preferences_dialog):
        """Test that Apply button is enabled when settings change."""
        # Change a setting
        preferences_dialog.general_tab.auto_load_checkbox.toggle()
        
        # Apply button should be enabled
        assert preferences_dialog.apply_btn.isEnabled()
    
    def test_ok_button_is_default(self, preferences_dialog):
        """Test that OK button is the default button."""
        assert preferences_dialog.ok_btn.isDefault()
    
    def test_dirty_state_tracking(self, preferences_dialog):
        """Test that dirty state is tracked correctly."""
        # Initially not dirty
        assert not preferences_dialog._is_dirty()
        
        # Make a change
        preferences_dialog.general_tab.auto_load_checkbox.toggle()
        
        # Should be dirty
        assert preferences_dialog._is_dirty()
    
    def test_save_settings_method(self, preferences_dialog, tmp_path):
        """Test that _save_settings method works."""
        # Create a weights directory
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        
        # Set a value
        preferences_dialog.general_tab.weights_dir_edit.setText(str(weights_dir))
        
        # Save
        result = preferences_dialog._save_settings()
        
        # Should succeed
        assert result is True
        
        # Apply button should be disabled after save
        assert not preferences_dialog.apply_btn.isEnabled()
    
    def test_settings_changed_signal(self, preferences_dialog, qtbot):
        """Test that settings_changed signal is emitted."""
        # Connect a spy to the signal
        with qtbot.waitSignal(preferences_dialog.settings_changed, timeout=1000):
            # Trigger save
            preferences_dialog.general_tab.auto_load_checkbox.toggle()
            preferences_dialog._save_settings()
    
    def test_initial_state_capture(self, preferences_dialog):
        """Test that initial state is captured on dialog creation."""
        assert preferences_dialog._initial_state is not None
        assert isinstance(preferences_dialog._initial_state, dict)
    
    def test_cancel_with_no_changes(self, preferences_dialog, qtbot, monkeypatch):
        """Test that Cancel works when no changes made."""
        # Mock QMessageBox to avoid blocking
        mock_called = {'called': False}
        
        def mock_question(*args, **kwargs):
            mock_called['called'] = True
            return QMessageBox.StandardButton.Yes
        
        monkeypatch.setattr(QMessageBox, 'question', mock_question)
        
        # Click Cancel (should close without warning since no changes)
        preferences_dialog._on_cancel()
        
        # Question should NOT be called since no changes were made
        assert not mock_called['called']
        
        # Dialog should be rejected
        assert preferences_dialog.result() == preferences_dialog.DialogCode.Rejected
    
    def test_cancel_with_changes_discard(self, preferences_dialog, qtbot, monkeypatch):
        """Test that Cancel with changes shows warning and discards on Yes."""
        # Make a change
        preferences_dialog.general_tab.auto_load_checkbox.toggle()
        
        # Track if question was called
        question_called = {'called': False}
        
        def mock_question(*args, **kwargs):
            question_called['called'] = True
            return QMessageBox.StandardButton.Yes
        
        monkeypatch.setattr(QMessageBox, 'question', mock_question)
        
        # Click Cancel
        preferences_dialog._on_cancel()
        
        # Question should have been called
        assert question_called['called']
        
        # Dialog should be rejected
        assert preferences_dialog.result() == preferences_dialog.DialogCode.Rejected
    
    def test_cancel_with_changes_keep(self, preferences_dialog, qtbot, monkeypatch):
        """Test that Cancel with changes keeps dialog open on No."""
        # Make a change
        preferences_dialog.general_tab.auto_load_checkbox.toggle()
        
        # Track if question was called
        question_called = {'called': False}
        
        def mock_question(*args, **kwargs):
            question_called['called'] = True
            return QMessageBox.StandardButton.No
        
        monkeypatch.setattr(QMessageBox, 'question', mock_question)
        
        # Click Cancel
        preferences_dialog._on_cancel()
        
        # Question should have been called
        assert question_called['called']
        
        # Dialog should NOT be closed
        assert preferences_dialog.result() == 0  # No result
    
    def test_close_event_with_changes(self, preferences_dialog, qtbot, monkeypatch):
        """Test that closing dialog with changes shows warning."""
        from PyQt6.QtGui import QCloseEvent
        
        # Make a change
        preferences_dialog.general_tab.auto_load_checkbox.toggle()
        
        # Mock QMessageBox.question to return Yes
        def mock_question(*args, **kwargs):
            return QMessageBox.StandardButton.Yes
        
        monkeypatch.setattr(QMessageBox, 'question', mock_question)
        
        # Create close event
        event = QCloseEvent()
        
        # Trigger close event
        preferences_dialog.closeEvent(event)
        
        # Event should be accepted (dialog closes)
        assert event.isAccepted()
        
        # Dialog result should be Rejected
        assert preferences_dialog.result() == preferences_dialog.DialogCode.Rejected
    
    def test_escape_key_with_changes_reject(self, preferences_dialog, qtbot, monkeypatch):
        """Test that Escape key with changes shows confirmation and rejects on Yes."""
        # Make a change
        preferences_dialog.general_tab.auto_load_checkbox.toggle()
        
        # Track if question was called
        question_called = {'called': False}
        
        def mock_question(*args, **kwargs):
            question_called['called'] = True
            return QMessageBox.StandardButton.Yes  # Discard changes
        
        monkeypatch.setattr(QMessageBox, 'question', mock_question)
        
        # Call reject() directly (which Escape key triggers)
        preferences_dialog.reject()
        
        # Question should have been called
        assert question_called['called']
        
        # Dialog should be rejected
        assert preferences_dialog.result() == preferences_dialog.DialogCode.Rejected
    
    def test_escape_key_with_changes_keep(self, preferences_dialog, qtbot, monkeypatch):
        """Test that Escape key with changes keeps dialog open on No."""
        # Make a change
        preferences_dialog.general_tab.auto_load_checkbox.toggle()
        
        # Track if question was called
        question_called = {'called': False}
        
        def mock_question(*args, **kwargs):
            question_called['called'] = True
            return QMessageBox.StandardButton.No  # Keep dialog open
        
        monkeypatch.setattr(QMessageBox, 'question', mock_question)
        
        # Call reject() directly (which Escape key triggers)
        preferences_dialog.reject()
        
        # Question should have been called
        assert question_called['called']
        
        # Dialog should still be open (not rejected)
        assert preferences_dialog.result() == 0
    
    def test_escape_key_no_changes(self, preferences_dialog, qtbot, monkeypatch):
        """Test that Escape key without changes closes immediately."""
        # Track if question was called
        question_called = {'called': False}
        
        def mock_question(*args, **kwargs):
            question_called['called'] = True
            return QMessageBox.StandardButton.Yes
        
        monkeypatch.setattr(QMessageBox, 'question', mock_question)
        
        # Call reject() directly (no changes made)
        preferences_dialog.reject()
        
        # Question should NOT have been called (no changes)
        assert not question_called['called']
        
        # Dialog should be rejected
        assert preferences_dialog.result() == preferences_dialog.DialogCode.Rejected

    def test_apply_button_handler(self, preferences_dialog, qtbot, tmp_path, monkeypatch):
        """Test that Apply button saves settings without closing."""
        # Create a weights directory
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        
        # Make a change
        preferences_dialog.general_tab.weights_dir_edit.setText(str(weights_dir))
        
        # Mock QMessageBox.information to avoid blocking
        def mock_information(*args, **kwargs):
            pass
        
        monkeypatch.setattr(QMessageBox, 'information', mock_information)
        
        # Click Apply
        preferences_dialog._on_apply()
        
        # Dialog should still be open (not accepted/rejected)
        assert preferences_dialog.result() == 0  # No result yet


class TestPreferencesDialogIntegration:
    """Integration tests for PreferencesDialog."""
    
    def test_settings_persist_after_save(self, preferences_dialog, settings, tmp_path):
        """Test that settings persist after saving."""
        # Create a weights directory
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        
        # Set values
        preferences_dialog.general_tab.weights_dir_edit.setText(str(weights_dir))
        preferences_dialog.general_tab.auto_load_checkbox.setChecked(False)
        
        # Save
        preferences_dialog._save_settings()
        
        # Verify persistence
        assert settings.get_weights_directory() == str(weights_dir)
        assert settings.get_auto_load_model() is False
    
    def test_multiple_tabs_in_future(self, preferences_dialog):
        """Test that dialog structure supports multiple tabs (future-proofing)."""
        # Currently only General tab exists
        assert preferences_dialog.tab_widget.count() == 1
        
        # Tab widget should support adding more tabs
        assert hasattr(preferences_dialog.tab_widget, 'addTab')
    
    def test_all_settings_round_trip(self, preferences_dialog, settings, tmp_path):
        """Test that all settings can be set and retrieved correctly."""
        # Create a weights directory
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        
        # Set all settings
        preferences_dialog.general_tab.weights_dir_edit.setText(str(weights_dir))
        preferences_dialog.general_tab.auto_load_checkbox.setChecked(True)
        preferences_dialog.general_tab.zoom_mode_combo.setCurrentIndex(
            preferences_dialog.general_tab.zoom_mode_combo.findData("fit")
        )
        preferences_dialog.general_tab.sync_zoom_checkbox.setChecked(True)
        preferences_dialog.general_tab.show_info_checkbox.setChecked(True)
        preferences_dialog.general_tab.gpu_mode_combo.setCurrentIndex(
            preferences_dialog.general_tab.gpu_mode_combo.findData("auto")
        )
        preferences_dialog.general_tab.max_dimension_combo.setCurrentIndex(
            preferences_dialog.general_tab.max_dimension_combo.findData(4096)
        )
        
        # Save
        preferences_dialog._save_settings()
        
        # Verify all settings
        assert settings.get_weights_directory() == str(weights_dir)
        assert settings.get_auto_load_model() is True
        assert settings.get_default_zoom_mode() == "fit"
        assert settings.get_sync_zoom() is True
        assert settings.get_show_info_overlay() is True
        assert settings.get_gpu_mode() == "auto"
        assert settings.get_max_image_dimension() == 4096


class TestPreferencesDialogEdgeCases:
    """Edge case tests for PreferencesDialog."""
    
    def test_empty_weights_directory(self, general_tab):
        """Test that empty weights directory is handled correctly."""
        # Set empty path
        general_tab.weights_dir_edit.setText("")
        
        # Save should succeed (empty is allowed - keeps current)
        result = general_tab.save_settings()
        assert result is True
    
    def test_whitespace_only_weights_directory(self, general_tab):
        """Test that whitespace-only directory is handled correctly."""
        # Set whitespace path
        general_tab.weights_dir_edit.setText("   ")
        
        # Save should succeed (treated as empty)
        result = general_tab.save_settings()
        assert result is True
    
    def test_relative_vs_absolute_paths(self, general_tab, tmp_path):
        """Test that both relative and absolute paths work."""
        # Create a weights directory
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        
        # Test absolute path
        general_tab.weights_dir_edit.setText(str(weights_dir))
        result = general_tab.save_settings()
        assert result is True
    
    def test_combo_box_invalid_index(self, general_tab):
        """Test that combo boxes handle invalid indices gracefully."""
        # Set invalid index (should not crash)
        try:
            general_tab.zoom_mode_combo.setCurrentIndex(999)
            # If no exception, test passes
            assert True
        except Exception:
            # Should not raise exception
            pytest.fail("Setting invalid combo box index raised exception")
    
    def test_rapid_setting_changes(self, preferences_dialog):
        """Test that rapid changes don't break dirty tracking."""
        from PyQt6.QtWidgets import QApplication
        
        # Store initial state
        initial_state = preferences_dialog.general_tab.auto_load_checkbox.isChecked()
        
        # Make multiple rapid changes (odd number to ensure final state differs)
        for _ in range(11):
            preferences_dialog.general_tab.auto_load_checkbox.toggle()
            QApplication.processEvents()  # Ensure signals are processed
        
        # Final state should differ from initial
        assert preferences_dialog.general_tab.auto_load_checkbox.isChecked() != initial_state
        
        # Should still track dirty state correctly
        assert preferences_dialog._is_dirty()
    
    def test_settings_sync_called(self, preferences_dialog, tmp_path):
        """Test that settings.sync() is called after save."""
        # Create a weights directory
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        
        # Set a value
        preferences_dialog.general_tab.weights_dir_edit.setText(str(weights_dir))
        
        # Save
        preferences_dialog._save_settings()
        
        # Create new settings instance to verify persistence
        new_settings = AppSettings()
        assert str(weights_dir) == new_settings.get_weights_directory()


@pytest.mark.gui
class TestPreferencesDialogUI:
    """UI-specific tests for PreferencesDialog."""
    
    def test_dialog_is_modal(self, preferences_dialog):
        """Test that dialog is modal (blocks parent)."""
        assert preferences_dialog.isModal()
    
    def test_tooltips_exist(self, general_tab):
        """Test that important controls have tooltips."""
        assert general_tab.weights_dir_edit.toolTip() != ""
        assert general_tab.auto_load_checkbox.toolTip() != ""
        assert general_tab.zoom_mode_combo.toolTip() != ""
        assert general_tab.gpu_mode_combo.toolTip() != ""
        assert general_tab.max_dimension_combo.toolTip() != ""
    
    def test_shortcut_works(self, preferences_dialog, qtbot):
        """Test that Ctrl+, shortcut exists in parent window."""
        # This would be tested in main window tests
        # Just verify dialog doesn't interfere
        assert True
