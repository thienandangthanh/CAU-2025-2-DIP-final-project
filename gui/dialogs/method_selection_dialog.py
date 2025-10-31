"""Method selection dialog for comparison feature.

This dialog allows users to select which enhancement methods to compare,
including options for selecting reference images.
"""

from typing import List, Optional, Dict
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from gui.utils.enhancement_methods import get_registry


class MethodSelectionDialog(QDialog):
    """Dialog for selecting enhancement methods and reference image.
    
    This dialog provides:
    - Checkboxes for each available enhancement method
    - Method descriptions
    - Disabled state for methods that can't run (e.g., Zero-DCE without model)
    - Quick selection buttons (Select All, Select None, Classical Only, etc.)
    - Optional reference image picker
    """
    
    def __init__(
        self,
        model_loaded: bool,
        current_selection: Optional[List[str]] = None,
        reference_path: Optional[str] = None,
        parent: Optional[QWidget] = None
    ):
        """Initialize the method selection dialog.
        
        Args:
            model_loaded: Whether a model is currently loaded
            current_selection: List of currently selected method keys
            reference_path: Path to current reference image (if any)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.model_loaded = model_loaded
        self.selected_methods: List[str] = current_selection or []
        self.reference_path = reference_path
        self.registry = get_registry()
        
        self.method_checkboxes: Dict[str, QCheckBox] = {}
        
        # Set dialog properties
        self.setWindowTitle("Select Enhancement Methods")
        self.setMinimumSize(500, 600)
        self.setModal(True)
        
        # Initialize UI
        self._init_ui()
        
        # Restore current selection
        self._restore_selection()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title label
        title_label = QLabel("Select methods to compare:")
        title_label.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333333;
            }
            """
        )
        layout.addWidget(title_label)
        
        # Quick selection buttons
        quick_buttons_layout = QHBoxLayout()
        quick_buttons_layout.setSpacing(10)
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        quick_buttons_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self._select_none)
        quick_buttons_layout.addWidget(select_none_btn)
        
        classical_only_btn = QPushButton("Classical Only")
        classical_only_btn.clicked.connect(self._select_classical_only)
        quick_buttons_layout.addWidget(classical_only_btn)
        
        fast_only_btn = QPushButton("Fast Methods")
        fast_only_btn.clicked.connect(self._select_fast_only)
        quick_buttons_layout.addWidget(fast_only_btn)
        
        quick_buttons_layout.addStretch()
        layout.addLayout(quick_buttons_layout)
        
        # Method selection scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        
        # Group methods by category
        dl_methods = self.registry.get_methods_by_category("deep_learning")
        classical_methods = self.registry.get_methods_by_category("classical")
        
        # Deep Learning Methods group
        if dl_methods:
            dl_group = QGroupBox("Deep Learning Methods")
            dl_group_layout = QVBoxLayout(dl_group)
            
            for method_key in dl_methods:
                method_info = self.registry.get_method_info(method_key)
                checkbox = self._create_method_checkbox(method_info)
                dl_group_layout.addWidget(checkbox)
            
            scroll_layout.addWidget(dl_group)
        
        # Classical Methods group
        if classical_methods:
            classical_group = QGroupBox("Classical Methods")
            classical_group_layout = QVBoxLayout(classical_group)
            
            for method_key in classical_methods:
                method_info = self.registry.get_method_info(method_key)
                checkbox = self._create_method_checkbox(method_info)
                classical_group_layout.addWidget(checkbox)
            
            scroll_layout.addWidget(classical_group)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        
        # Reference image section
        reference_group = QGroupBox("Reference Image (Optional)")
        reference_layout = QVBoxLayout(reference_group)
        
        ref_info_label = QLabel(
            "Select a high-light reference image for comparison.\n"
            "This is optional and helps visualize the enhancement effect."
        )
        ref_info_label.setWordWrap(True)
        ref_info_label.setStyleSheet("color: #666666; font-size: 11px;")
        reference_layout.addWidget(ref_info_label)
        
        # Reference path display and browse button
        ref_path_layout = QHBoxLayout()
        
        self.reference_line_edit = QLineEdit()
        self.reference_line_edit.setPlaceholderText("No reference image selected")
        self.reference_line_edit.setReadOnly(True)
        if self.reference_path:
            self.reference_line_edit.setText(self.reference_path)
        ref_path_layout.addWidget(self.reference_line_edit)
        
        browse_ref_btn = QPushButton("Browse...")
        browse_ref_btn.clicked.connect(self._browse_reference)
        ref_path_layout.addWidget(browse_ref_btn)
        
        clear_ref_btn = QPushButton("Clear")
        clear_ref_btn.clicked.connect(self._clear_reference)
        ref_path_layout.addWidget(clear_ref_btn)
        
        reference_layout.addLayout(ref_path_layout)
        layout.addWidget(reference_group)
        
        # Dialog buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        buttons_layout.addWidget(ok_btn)
        
        layout.addLayout(buttons_layout)
    
    def _create_method_checkbox(self, method_info) -> QCheckBox:
        """Create a checkbox for a method with description.
        
        Args:
            method_info: EnhancementMethod object
            
        Returns:
            QCheckBox widget
        """
        # Create checkbox
        checkbox = QCheckBox(method_info.name)
        checkbox.setProperty("method_key", method_info.key)
        
        # Check if method can run
        can_run = self.registry.can_run_method(method_info.key, self.model_loaded)
        
        if not can_run:
            checkbox.setEnabled(False)
            tooltip = f"{method_info.description}\n\n(Requires model to be loaded)"
        else:
            tooltip = method_info.description
        
        checkbox.setToolTip(tooltip)
        
        # Style
        checkbox.setStyleSheet(
            """
            QCheckBox {
                font-size: 12px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            """
        )
        
        # Store checkbox
        self.method_checkboxes[method_info.key] = checkbox
        
        return checkbox
    
    def _restore_selection(self):
        """Restore the current selection state."""
        for method_key in self.selected_methods:
            if method_key in self.method_checkboxes:
                self.method_checkboxes[method_key].setChecked(True)
    
    def _select_all(self):
        """Select all available methods."""
        for method_key, checkbox in self.method_checkboxes.items():
            if checkbox.isEnabled():
                checkbox.setChecked(True)
    
    def _select_none(self):
        """Deselect all methods."""
        for checkbox in self.method_checkboxes.values():
            checkbox.setChecked(False)
    
    def _select_classical_only(self):
        """Select only classical methods."""
        classical_methods = self.registry.get_methods_by_category("classical")
        
        for method_key, checkbox in self.method_checkboxes.items():
            if method_key in classical_methods and checkbox.isEnabled():
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(False)
    
    def _select_fast_only(self):
        """Select only fast methods."""
        from gui.utils.enhancement_methods import ExecutionSpeed
        fast_methods = self.registry.get_methods_by_speed(ExecutionSpeed.FAST)
        
        for method_key, checkbox in self.method_checkboxes.items():
            if method_key in fast_methods and checkbox.isEnabled():
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(False)
    
    def _browse_reference(self):
        """Browse for a reference image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if file_path:
            self.reference_path = file_path
            self.reference_line_edit.setText(file_path)
    
    def _clear_reference(self):
        """Clear the reference image selection."""
        self.reference_path = None
        self.reference_line_edit.clear()
    
    def _on_ok(self):
        """Handle OK button click."""
        # Get selected methods
        selected = []
        for method_key, checkbox in self.method_checkboxes.items():
            if checkbox.isChecked():
                selected.append(method_key)
        
        # Validate selection
        if not selected:
            QMessageBox.warning(
                self,
                "No Methods Selected",
                "Please select at least one enhancement method to compare."
            )
            return
        
        self.selected_methods = selected
        self.accept()
    
    def get_selected_methods(self) -> List[str]:
        """Get the list of selected method keys.
        
        Returns:
            List of selected method keys
        """
        return self.selected_methods
    
    def get_reference_path(self) -> Optional[str]:
        """Get the selected reference image path.
        
        Returns:
            Reference image path, or None if not selected
        """
        return self.reference_path
