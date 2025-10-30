"""Preferences dialog for application settings.

This module provides a comprehensive preferences dialog with tabbed interface
for managing all application settings including model configuration, display
options, and performance settings.
"""

from typing import Optional
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QPushButton,
    QWidget,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence

from gui.utils import AppSettings


class GeneralTab(QWidget):
    """General settings tab for preferences dialog.
    
    Provides settings for:
    - Model configuration (default weights, auto-load)
    - Display options (zoom mode, info overlay, sync zoom)
    - Performance settings (GPU mode, max image dimension)
    """
    
    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None):
        """Initialize the General tab.
        
        Args:
            settings: Application settings instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.settings = settings
        self._init_ui()
        self._load_settings()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Model Settings Group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        # Default model weights path
        weights_layout = QHBoxLayout()
        self.weights_path_edit = QLineEdit()
        self.weights_path_edit.setPlaceholderText("Path to model weights file")
        self.weights_path_edit.setToolTip(
            "Default model weights file to load on startup.\n"
            "Must be a .h5 or .weights.h5 file."
        )
        weights_layout.addWidget(self.weights_path_edit, 1)
        
        self.browse_weights_btn = QPushButton("Browse...")
        self.browse_weights_btn.clicked.connect(self._browse_weights)
        self.browse_weights_btn.setToolTip("Select a model weights file")
        weights_layout.addWidget(self.browse_weights_btn)
        
        model_layout.addRow("Default Model Weights:", weights_layout)
        
        # Auto-load model on startup
        self.auto_load_checkbox = QCheckBox("Auto-load model on startup")
        self.auto_load_checkbox.setToolTip(
            "Automatically load the default model when the application starts"
        )
        model_layout.addRow("", self.auto_load_checkbox)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Display Settings Group
        display_group = QGroupBox("Display Settings")
        display_layout = QFormLayout()
        
        # Default zoom level
        self.zoom_mode_combo = QComboBox()
        self.zoom_mode_combo.addItem("Fit to Window", "fit")
        self.zoom_mode_combo.addItem("Actual Size (100%)", "actual")
        self.zoom_mode_combo.setToolTip(
            "Default zoom mode for newly loaded images"
        )
        display_layout.addRow("Default Zoom Level:", self.zoom_mode_combo)
        
        # Sync zoom between panels
        self.sync_zoom_checkbox = QCheckBox("Keep zoom synchronized between panels")
        self.sync_zoom_checkbox.setToolTip(
            "When enabled, zoom changes apply to both input and output panels"
        )
        display_layout.addRow("", self.sync_zoom_checkbox)
        
        # Show info overlay
        self.show_info_checkbox = QCheckBox("Show image info overlay")
        self.show_info_checkbox.setToolTip(
            "Display image dimensions and other information on the image panels"
        )
        display_layout.addRow("", self.show_info_checkbox)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Performance Settings Group
        performance_group = QGroupBox("Performance Settings")
        performance_layout = QFormLayout()
        
        # GPU acceleration
        gpu_layout = QVBoxLayout()
        self.gpu_mode_combo = QComboBox()
        self.gpu_mode_combo.addItem("Auto (Recommended)", "auto")
        self.gpu_mode_combo.addItem("Enable", "enable")
        self.gpu_mode_combo.addItem("Disable (CPU Only)", "disable")
        self.gpu_mode_combo.setToolTip(
            "GPU acceleration mode:\n"
            "- Auto: Use GPU if available\n"
            "- Enable: Force GPU usage (may fail if unavailable)\n"
            "- Disable: Use CPU only"
        )
        gpu_layout.addWidget(self.gpu_mode_combo)
        
        # GPU status label
        self.gpu_status_label = QLabel()
        self.gpu_status_label.setStyleSheet("color: #666666; font-size: 11px;")
        self._update_gpu_status()
        gpu_layout.addWidget(self.gpu_status_label)
        
        performance_layout.addRow("GPU Acceleration:", gpu_layout)
        
        # Max image dimension
        max_dim_layout = QVBoxLayout()
        self.max_dimension_combo = QComboBox()
        self.max_dimension_combo.addItem("2048 pixels", 2048)
        self.max_dimension_combo.addItem("4096 pixels (Recommended)", 4096)
        self.max_dimension_combo.addItem("8192 pixels", 8192)
        self.max_dimension_combo.addItem("Unlimited (May use lots of memory)", -1)
        self.max_dimension_combo.setToolTip(
            "Maximum dimension for loaded images.\n"
            "Larger images will be downscaled to fit this limit."
        )
        self.max_dimension_combo.currentIndexChanged.connect(self._on_max_dimension_changed)
        max_dim_layout.addWidget(self.max_dimension_combo)
        
        # Warning label for unlimited
        self.unlimited_warning_label = QLabel(
            "⚠ Warning: Unlimited may cause out-of-memory errors"
        )
        self.unlimited_warning_label.setStyleSheet("color: #FF6B6B; font-size: 11px;")
        self.unlimited_warning_label.hide()
        max_dim_layout.addWidget(self.unlimited_warning_label)
        
        performance_layout.addRow("Max Image Dimension:", max_dim_layout)
        
        performance_group.setLayout(performance_layout)
        layout.addWidget(performance_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def _browse_weights(self):
        """Open file dialog to browse for model weights."""
        current_path = self.weights_path_edit.text()
        if current_path and Path(current_path).exists():
            start_dir = str(Path(current_path).parent)
        else:
            start_dir = self.settings.get_weights_directory()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Weights File",
            start_dir,
            "Model Weights (*.h5 *.weights.h5);;All Files (*)"
        )
        
        if file_path:
            self.weights_path_edit.setText(file_path)
    
    def _update_gpu_status(self):
        """Update GPU status label."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.gpu_status_label.setText(f"Status: {len(gpus)} GPU(s) detected")
                self.gpu_status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
            else:
                self.gpu_status_label.setText("Status: No GPU detected (CPU mode)")
                self.gpu_status_label.setStyleSheet("color: #FF9800; font-size: 11px;")
        except Exception:
            self.gpu_status_label.setText("Status: Unable to detect GPU")
            self.gpu_status_label.setStyleSheet("color: #666666; font-size: 11px;")
    
    def _on_max_dimension_changed(self, index: int):
        """Handle max dimension combo box change.
        
        Args:
            index: Selected combo box index
        """
        max_dim = self.max_dimension_combo.itemData(index)
        if max_dim == -1:  # Unlimited
            self.unlimited_warning_label.show()
        else:
            self.unlimited_warning_label.hide()
    
    def _load_settings(self):
        """Load current settings into UI controls."""
        # Model settings
        full_path = self.settings.get_full_model_path()
        self.weights_path_edit.setText(full_path)
        self.auto_load_checkbox.setChecked(self.settings.get_auto_load_model())
        
        # Display settings
        zoom_mode = self.settings.get_default_zoom_mode()
        index = self.zoom_mode_combo.findData(zoom_mode)
        if index >= 0:
            self.zoom_mode_combo.setCurrentIndex(index)
        
        self.sync_zoom_checkbox.setChecked(self.settings.get_sync_zoom())
        self.show_info_checkbox.setChecked(self.settings.get_show_info_overlay())
        
        # Performance settings
        gpu_mode = self.settings.get_gpu_mode()
        index = self.gpu_mode_combo.findData(gpu_mode)
        if index >= 0:
            self.gpu_mode_combo.setCurrentIndex(index)
        
        max_dim = self.settings.get_max_image_dimension()
        index = self.max_dimension_combo.findData(max_dim)
        if index >= 0:
            self.max_dimension_combo.setCurrentIndex(index)
        else:
            # Default to 4096 if not found
            index = self.max_dimension_combo.findData(4096)
            if index >= 0:
                self.max_dimension_combo.setCurrentIndex(index)
    
    def save_settings(self) -> bool:
        """Save settings from UI controls.
        
        Returns:
            True if settings saved successfully, False otherwise
        """
        # Validate model weights path
        weights_path = self.weights_path_edit.text().strip()
        if weights_path:
            path = Path(weights_path)
            if not path.exists():
                QMessageBox.warning(
                    self,
                    "Invalid Model Path",
                    f"The model weights file does not exist:\n{weights_path}\n\n"
                    "Please select a valid file or leave empty to use default."
                )
                return False
            
            if not (path.suffix == '.h5' or path.name.endswith('.weights.h5')):
                QMessageBox.warning(
                    self,
                    "Invalid File Type",
                    "Model weights file must be a .h5 or .weights.h5 file."
                )
                return False
            
            # Split into directory and filename
            self.settings.set_weights_directory(str(path.parent))
            self.settings.set_default_model_file(path.name)
        
        # Model settings
        self.settings.set_auto_load_model(self.auto_load_checkbox.isChecked())
        
        # Display settings
        zoom_mode = self.zoom_mode_combo.currentData()
        self.settings.set_default_zoom_mode(zoom_mode)
        self.settings.set_sync_zoom(self.sync_zoom_checkbox.isChecked())
        self.settings.set_show_info_overlay(self.show_info_checkbox.isChecked())
        
        # Performance settings
        gpu_mode = self.gpu_mode_combo.currentData()
        self.settings.set_gpu_mode(gpu_mode)
        
        max_dim = self.max_dimension_combo.currentData()
        self.settings.set_max_image_dimension(max_dim)
        
        return True
    
    def get_initial_state(self) -> dict:
        """Get initial state of all settings for dirty tracking.
        
        Returns:
            Dictionary of setting name to value
        """
        return {
            'weights_path': self.weights_path_edit.text(),
            'auto_load': self.auto_load_checkbox.isChecked(),
            'zoom_mode': self.zoom_mode_combo.currentData(),
            'sync_zoom': self.sync_zoom_checkbox.isChecked(),
            'show_info': self.show_info_checkbox.isChecked(),
            'gpu_mode': self.gpu_mode_combo.currentData(),
            'max_dimension': self.max_dimension_combo.currentData(),
        }
    
    def get_current_state(self) -> dict:
        """Get current state of all settings for dirty tracking.
        
        Returns:
            Dictionary of setting name to value
        """
        return self.get_initial_state()


class PreferencesDialog(QDialog):
    """Preferences dialog with tabbed interface for application settings.
    
    Features:
    - Tabbed interface for organizing settings
    - OK/Cancel/Apply buttons
    - Dirty state tracking (warns about unsaved changes)
    - Settings validation
    - Restart warning if needed
    
    Signals:
        settings_changed: Emitted when settings are saved
    """
    
    settings_changed = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the preferences dialog.
        
        Args:
            parent: Parent widget (typically MainWindow)
        """
        super().__init__(parent)
        
        self.settings = AppSettings()
        self._initial_state = {}
        self._settings_require_restart = False
        
        self._init_ui()
        self._capture_initial_state()
    
    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Preferences")
        self.setMinimumSize(600, 450)
        self.setModal(True)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # General tab
        self.general_tab = GeneralTab(self.settings)
        self.tab_widget.addTab(self.general_tab, "General")
        
        # Future tabs can be added here
        # self.advanced_tab = AdvancedTab(self.settings)
        # self.tab_widget.addTab(self.advanced_tab, "Advanced")
        
        layout.addWidget(self.tab_widget)
        
        # Button box
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Apply button
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        self.apply_btn.setToolTip("Save settings without closing")
        self.apply_btn.setEnabled(False)  # Disabled until changes made
        button_layout.addWidget(self.apply_btn)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setToolTip("Discard changes and close")
        button_layout.addWidget(self.cancel_btn)
        
        # OK button
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self._on_ok)
        self.ok_btn.setDefault(True)
        self.ok_btn.setToolTip("Save settings and close")
        button_layout.addWidget(self.ok_btn)
        
        layout.addLayout(button_layout)
        
        # Connect signals for dirty state tracking
        self._connect_change_signals()
        
        # Center on parent
        if self.parent():
            self._center_on_parent()
    
    def _connect_change_signals(self):
        """Connect signals to track changes (dirty state)."""
        # General tab signals
        self.general_tab.weights_path_edit.textChanged.connect(self._on_settings_changed)
        self.general_tab.auto_load_checkbox.stateChanged.connect(self._on_settings_changed)
        self.general_tab.zoom_mode_combo.currentIndexChanged.connect(self._on_settings_changed)
        self.general_tab.sync_zoom_checkbox.stateChanged.connect(self._on_settings_changed)
        self.general_tab.show_info_checkbox.stateChanged.connect(self._on_settings_changed)
        self.general_tab.gpu_mode_combo.currentIndexChanged.connect(self._on_settings_changed)
        self.general_tab.max_dimension_combo.currentIndexChanged.connect(self._on_settings_changed)
    
    def _on_settings_changed(self):
        """Handle settings change (enable Apply button)."""
        self.apply_btn.setEnabled(True)
    
    def _capture_initial_state(self):
        """Capture initial state for dirty tracking."""
        self._initial_state = self.general_tab.get_initial_state()
    
    def _is_dirty(self) -> bool:
        """Check if settings have been modified.
        
        Returns:
            True if settings differ from initial state
        """
        current_state = self.general_tab.get_current_state()
        return current_state != self._initial_state
    
    def _save_settings(self) -> bool:
        """Save all settings from tabs.
        
        Returns:
            True if all settings saved successfully
        """
        # Save general tab settings
        if not self.general_tab.save_settings():
            return False
        
        # Sync settings to disk
        self.settings.sync()
        
        # Emit signal
        self.settings_changed.emit()
        
        # Disable Apply button after successful save
        self.apply_btn.setEnabled(False)
        
        # Update initial state
        self._capture_initial_state()
        
        return True
    
    def _on_ok(self):
        """Handle OK button click (save and close)."""
        if self._save_settings():
            self.accept()
    
    def _on_cancel(self):
        """Handle Cancel button click (discard and close)."""
        # Call reject() which we've overridden to handle unsaved changes
        self.reject()
    
    def reject(self):
        """Override reject() to handle Escape key and Cancel button consistently.
        
        This method is called by:
        - Escape key press (Qt built-in behavior)
        - Cancel button (via _on_cancel)
        """
        if self._is_dirty():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Discard them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # User confirmed, close the dialog
                super().reject()
            # If No, do nothing (dialog stays open)
        else:
            # No changes, close normally
            super().reject()
    
    def _on_apply(self):
        """Handle Apply button click (save without closing)."""
        if self._save_settings():
            QMessageBox.information(
                self,
                "Settings Saved",
                "Settings have been saved successfully."
            )
    
    def _center_on_parent(self):
        """Center dialog on parent window."""
        parent_geometry = self.parent().geometry()
        dialog_geometry = self.geometry()
        
        x = parent_geometry.x() + (parent_geometry.width() - dialog_geometry.width()) // 2
        y = parent_geometry.y() + (parent_geometry.height() - dialog_geometry.height()) // 2
        
        self.move(x, y)
    
    def closeEvent(self, event):
        """Handle dialog close event (warn about unsaved changes).
        
        This handles:
        - Alt+F4 key combination
        - Window manager close actions
        
        Note: Escape key and Cancel button go through reject() override instead.
        
        Args:
            event: Close event
        """
        if self._is_dirty():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Discard them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Set result to Rejected since user is discarding changes
                self.setResult(QDialog.DialogCode.Rejected)
                event.accept()
            else:
                event.ignore()
        else:
            # No changes, close normally with Rejected result
            self.setResult(QDialog.DialogCode.Rejected)
            event.accept()
