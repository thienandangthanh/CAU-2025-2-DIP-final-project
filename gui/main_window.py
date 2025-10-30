"""Main window for Zero-DCE GUI application.

This module provides the main application window with:
- Image display panels (input and output)
- Enhancement button
- Menu bar with File, Model, and Help menus
- Status bar
- File dialogs
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import time
from pathlib import Path
from typing import Optional, Dict
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QMenuBar,
    QMenu,
    QStatusBar,
    QFileDialog,
    QLabel,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence

from gui.widgets import ImagePanel, EnhanceButton
from gui.dialogs import ErrorDialog, PreferencesDialog
from gui.utils import ModelLoader, ImageProcessor, AppSettings, EnhancementResult


class EnhancementWorker(QThread):
    """Worker thread for image enhancement to avoid UI freezing.

    Signals:
        finished: Emitted when enhancement completes successfully (QPixmap)
        error: Emitted when enhancement fails (Exception)
        progress: Emitted to update progress (int, 0-100)
    """

    finished = pyqtSignal(object)  # Enhanced PIL Image
    error = pyqtSignal(Exception)
    progress = pyqtSignal(int)

    def __init__(self, image, model, original_size):
        """Initialize the worker.

        Args:
            image: PIL Image to enhance
            model: ZeroDCE model instance
            original_size: Original image size (width, height)
        """
        super().__init__()
        self.image = image
        self.model = model
        self.original_size = original_size

    def run(self):
        """Run the enhancement process."""
        try:
            self.progress.emit(0)

            # Enhance image
            enhanced_image = ImageProcessor.enhance_image(
                self.image, self.model, self.original_size
            )

            self.progress.emit(100)
            self.finished.emit(enhanced_image)

        except Exception as e:
            self.error.emit(e)


class MainWindow(QMainWindow):
    """Main application window for Zero-DCE GUI.

    Provides a complete interface for:
    - Loading input images
    - Loading model weights
    - Enhancing images
    - Saving enhanced results
    - Managing application settings
    """

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        # Initialize utilities
        self.model_loader = ModelLoader()
        self.settings = AppSettings()

        # Current state
        self.current_input_image = None  # PIL Image
        self.current_enhanced_image = None  # PIL Image
        self.enhancement_worker = None

        # Enhancement timing and results (future-proof for multi-method comparison)
        self._enhancement_results: Dict[
            str, EnhancementResult
        ] = {}  # Stores all enhancement results
        self._enhancement_start_time: Optional[float] = (
            None  # Track timing for current operation
        )
        self._current_enhancement_method: str = "Zero-DCE"  # Current method name

        # Initialize UI
        self._init_ui()

        # Load model if auto-load is enabled
        if self.settings.get_auto_load_model():
            self._auto_load_model()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Zero-DCE Image Enhancement")
        self.setMinimumSize(900, 600)

        # Restore window geometry if saved
        pos, size = self.settings.get_window_geometry()
        if pos is not None and size is not None:
            self.move(pos)
            self.resize(size)
        else:
            self.resize(1200, 700)

        # Create menu bar
        self._create_menu_bar()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Image panels layout (horizontal)
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(20)

        # Input panel
        self.input_panel = ImagePanel(
            title="Input Image",
            placeholder_text="Click to open an image\nor drag & drop here",
        )
        self.input_panel.image_clicked.connect(self._open_image)
        self.input_panel.image_dropped.connect(self._load_dropped_image)
        self.input_panel.cleared.connect(self._on_input_panel_cleared)
        panels_layout.addWidget(self.input_panel, 1)

        # Enhance button (centered between panels)
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.enhance_button = EnhanceButton()
        self.enhance_button.enhance_clicked.connect(self._enhance_image)
        button_layout.addWidget(self.enhance_button)

        panels_layout.addWidget(button_container)

        # Output panel
        self.output_panel = ImagePanel(
            title="Enhanced Image", placeholder_text="Enhanced image will appear here"
        )
        self.output_panel.image_clicked.connect(self._save_enhanced_image)
        self.output_panel.cleared.connect(self._on_output_panel_cleared)
        panels_layout.addWidget(self.output_panel, 1)

        main_layout.addLayout(panels_layout)

        # Create status bar
        self._create_status_bar()
        
        # Initialize status display with default model
        self._update_model_status_display()

        # Update UI state
        self._update_ui_state()

    def _create_menu_bar(self):
        """Create the menu bar with File, Model, and Help menus."""
        menubar = self.menuBar()

        # ==================== File Menu ====================
        file_menu = menubar.addMenu("&File")

        # Open Image
        open_action = QAction("&Open Image...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip("Open an image file")
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)

        # Recent Files submenu
        self.recent_files_menu = QMenu("Recent &Files", self)
        file_menu.addMenu(self.recent_files_menu)
        self._update_recent_files_menu()

        file_menu.addSeparator()

        # Save Enhanced Image
        self.save_action = QAction("&Save Enhanced Image...", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.setStatusTip("Save the enhanced image")
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self._save_enhanced_image)
        file_menu.addAction(self.save_action)

        file_menu.addSeparator()

        # Exit
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ==================== Edit Menu ====================
        edit_menu = menubar.addMenu("&Edit")

        # Clear Input
        clear_action = QAction("&Clear Input", self)
        clear_action.setStatusTip("Clear the input image")
        clear_action.triggered.connect(self._clear_input)
        edit_menu.addAction(clear_action)

        edit_menu.addSeparator()

        # Preferences
        preferences_action = QAction("&Preferences...", self)
        preferences_action.setShortcut(QKeySequence("Ctrl+,"))
        preferences_action.setStatusTip("Open application preferences")
        preferences_action.triggered.connect(self._show_preferences)
        edit_menu.addAction(preferences_action)

        # ==================== Model Menu ====================
        model_menu = menubar.addMenu("&Model")

        # Load Model Weights
        load_model_action = QAction("&Load Model Weights...", self)
        load_model_action.setStatusTip("Load Zero-DCE model weights")
        load_model_action.triggered.connect(self._load_model)
        model_menu.addAction(load_model_action)

        # Default Weights submenu
        self.default_weights_menu = QMenu("&Default Weights", self)
        model_menu.addMenu(self.default_weights_menu)
        self._update_default_weights_menu()

        model_menu.addSeparator()

        # Model Info
        model_info_action = QAction("Model &Info", self)
        model_info_action.setStatusTip("Show current model information")
        model_info_action.triggered.connect(self._show_model_info)
        model_menu.addAction(model_info_action)

        # ==================== Help Menu ====================
        help_menu = menubar.addMenu("&Help")

        # About
        about_action = QAction("&About", self)
        about_action.setStatusTip("About Zero-DCE GUI")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_status_bar(self):
        """Create the status bar with model and image info."""
        self.statusBar().showMessage("Ready")

        # Model info label (right side)
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setStyleSheet("QLabel { padding: 0 10px; }")
        self.statusBar().addPermanentWidget(self.model_status_label)
    
    def _update_model_status_display(self):
        """Update the model status label in status bar.
        
        This reflects the currently loaded model.
        Called when:
        - A model is loaded
        - Model is cleared
        """
        if self.model_loader.is_model_loaded():
            # Show currently loaded model
            filename = Path(self.model_loader.get_weights_path()).name
            self.model_status_label.setText(f"Model: {filename}")
        else:
            # No model loaded
            self.model_status_label.setText("No model loaded")

    # ==================== Menu Actions ====================

    def _update_recent_files_menu(self):
        """Update the Recent Files submenu with current list of recent files."""
        # Clear existing menu items
        self.recent_files_menu.clear()

        # Get recent files list
        recent_files = self.settings.get_recent_files()

        if not recent_files:
            # Show empty state
            empty_action = QAction("(Empty)", self)
            empty_action.setEnabled(False)
            self.recent_files_menu.addAction(empty_action)
        else:
            # Add action for each recent file
            for filepath in recent_files:
                # Check if file still exists
                file_exists = Path(filepath).exists()

                # Get filename for display (truncate if too long)
                filename = Path(filepath).name
                if len(filename) > 50:
                    filename = filename[:47] + "..."

                # Create action
                action = QAction(filename, self)
                action.setStatusTip(filepath)

                # Disable if file no longer exists
                if not file_exists:
                    action.setEnabled(False)
                    action.setText(f"{filename} (not found)")
                else:
                    # Connect to open the file
                    action.triggered.connect(
                        lambda checked=False, path=filepath: self._load_image(path)
                    )

                self.recent_files_menu.addAction(action)

            # Add separator and Clear action
            self.recent_files_menu.addSeparator()
            clear_action = QAction("&Clear Recent Files", self)
            clear_action.triggered.connect(self._clear_recent_files)
            self.recent_files_menu.addAction(clear_action)

    def _clear_recent_files(self):
        """Clear the recent files list."""
        self.settings.clear_recent_files()
        self._update_recent_files_menu()
        self.statusBar().showMessage("Recent files cleared", 2000)

    def _update_default_weights_menu(self):
        """Update the Default Weights submenu with available weight files."""
        # Clear existing menu items
        self.default_weights_menu.clear()

        # Get available weights
        weights_dir = self.settings.get_weights_directory()
        available_weights = ModelLoader.list_available_weights(weights_dir)

        if not available_weights:
            # Show empty state
            empty_action = QAction("(No weights found)", self)
            empty_action.setEnabled(False)
            self.default_weights_menu.addAction(empty_action)
        else:
            # Add action for each weight file
            for weights_path in available_weights:
                weights_file = Path(weights_path)

                # Extract epoch if available
                epoch_str = self._extract_epoch_from_filename(weights_file.name)
                display_name = weights_file.name
                if epoch_str:
                    display_name = f"{weights_file.name} ({epoch_str})"

                # Create action
                action = QAction(display_name, self)
                action.setStatusTip(weights_path)

                # Mark currently loaded model with checkmark
                # Normalize paths for comparison (resolve to absolute paths)
                if self.model_loader.is_model_loaded():
                    current_path = Path(self.model_loader.get_weights_path()).resolve()
                    menu_path = Path(weights_path).resolve()
                    if current_path == menu_path:
                        action.setCheckable(True)
                        action.setChecked(True)

                # Connect to load the weights and save as default
                action.triggered.connect(
                    lambda checked=False, path=weights_path: self._load_model_weights(
                        path, save_as_default=True
                    )
                )

                self.default_weights_menu.addAction(action)

    def _extract_epoch_from_filename(self, filename: str) -> str:
        """Extract training epoch from filename.

        Args:
            filename: Model filename

        Returns:
            Epoch string (e.g., "Epoch 100") or empty string if not found
        """
        if not filename:
            return ""

        import re

        # Try different patterns
        patterns = [
            r"epoch[_-]?(\d+)",  # epoch100, epoch_100, epoch-100
            r"e(\d+)",  # e100
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                epoch_num = match.group(1)
                return f"Epoch {epoch_num}"

        return ""

    def _open_image(self):
        """Open file dialog to select an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
        )

        if file_path:
            self._load_image(file_path)

    def _load_image(self, filepath: str):
        """Load an image from file path.

        Args:
            filepath: Path to image file
        """
        try:
            # Validate image file
            is_valid, error_msg = ImageProcessor.validate_image_file(filepath)
            if not is_valid:
                ErrorDialog.show_error(
                    title="Invalid Image",
                    message="Cannot load the selected image.",
                    details=error_msg,
                    parent=self,
                )
                return

            # Load image
            max_dimension = self.settings.get_max_image_dimension()
            image = ImageProcessor.load_image(filepath, max_dimension)

            # Store image
            self.current_input_image = image

            # Display in input panel
            pixmap = ImageProcessor.pil_to_pixmap(image)
            self.input_panel.set_image_from_pixmap(pixmap, filepath)

            # Clear output panel
            self.output_panel.clear()
            self.current_enhanced_image = None

            # Add to recent files
            self.settings.add_recent_file(filepath)

            # Update recent files menu
            self._update_recent_files_menu()

            # Update UI state
            self._update_ui_state()

            self.statusBar().showMessage(f"Loaded: {Path(filepath).name}", 3000)

        except Exception as e:
            ErrorDialog.show_image_error(e, self)

    def _load_dropped_image(self, filepath: str):
        """Handle image dropped onto input panel.

        Args:
            filepath: Path to dropped image file
        """
        self._load_image(filepath)

    def _save_enhanced_image(self):
        """Save the enhanced image to file."""
        if self.current_enhanced_image is None:
            return

        # Get default filename
        if self.input_panel.get_image_path():
            input_path = Path(self.input_panel.get_image_path())
            default_name = f"{input_path.stem}_enhanced{input_path.suffix}"
        else:
            default_name = "enhanced.png"

        # Open save dialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Enhanced Image",
            default_name,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )

        if file_path:
            try:
                # Determine format from filter or extension
                if "PNG" in selected_filter or file_path.lower().endswith(".png"):
                    format = "PNG"
                else:
                    format = "JPEG"

                # Get quality setting
                quality = self.settings.get_jpeg_quality()

                # Save image
                ImageProcessor.save_image(
                    self.current_enhanced_image, file_path, format, quality
                )

                self.statusBar().showMessage(f"Saved: {Path(file_path).name}", 3000)

            except Exception as e:
                ErrorDialog.show_file_error(e, self)

    def _clear_input(self):
        """Clear the input image and reset panels."""
        self.input_panel.clear()
        self.output_panel.clear()
        self.current_input_image = None
        self.current_enhanced_image = None
        self._update_ui_state()
        self.statusBar().showMessage("Cleared", 2000)

    def _show_preferences(self):
        """Show the preferences dialog."""
        dialog = PreferencesDialog(self)
        dialog.settings_changed.connect(self._on_settings_changed)
        dialog.exec()

    def _on_settings_changed(self):
        """Handle settings changed signal from preferences dialog."""
        # Reload settings to get the latest values
        self.settings = AppSettings()
        
        # Update UI elements that depend on settings
        self._update_model_status_display()
        self._update_default_weights_menu()
        
        # Show confirmation
        self.statusBar().showMessage("Preferences saved and applied", 2000)

    def _on_input_panel_cleared(self):
        """Handle input panel cleared signal.

        Clear the input image buffer and also clear output since
        there's no input to enhance anymore.
        """
        self.current_input_image = None
        self.output_panel.clear()
        self.current_enhanced_image = None
        self._update_ui_state()
        self.statusBar().showMessage("Input cleared", 2000)

    def _on_output_panel_cleared(self):
        """Handle output panel cleared signal.

        Clear the enhanced image buffer.
        """
        self.current_enhanced_image = None
        self._update_ui_state()
        self.statusBar().showMessage("Enhanced image cleared", 2000)

    def _load_model(self):
        """Open file dialog to load model weights."""
        # Get default weights directory
        weights_dir = self.settings.get_weights_directory()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model Weights",
            weights_dir,
            "Model Weights (*.h5 *.weights.h5);;All Files (*)",
        )

        if file_path:
            self._load_model_weights(file_path)

    def _load_model_weights(self, filepath: str, save_as_default: bool = False):
        """Load model weights from file.

        Args:
            filepath: Path to weights file
            save_as_default: If True, save this as the default model in settings
        """
        try:
            self.statusBar().showMessage("Loading model...", 0)

            # Validate weights file
            is_valid, error_msg = ModelLoader.validate_weights_file(filepath)
            if not is_valid:
                ErrorDialog.show_error(
                    title="Invalid Weights File",
                    message="Cannot load the selected weights file.",
                    details=error_msg,
                    parent=self,
                )
                self.statusBar().showMessage("Ready", 2000)
                return

            # Load model
            self.model_loader.load_model(filepath)

            # Save directory as default if requested
            if save_as_default:
                filepath_obj = Path(filepath)
                # Save only the directory
                self.settings.set_weights_directory(str(filepath_obj.parent))
                self.settings.sync()

            # Update status bar and menu
            self._update_model_status_display()
            self._update_default_weights_menu()
            self.statusBar().showMessage("Model loaded successfully", 3000)

            # Update UI state
            self._update_ui_state()

        except Exception as e:
            ErrorDialog.show_model_error(e, self)
            self.statusBar().showMessage("Failed to load model", 3000)

    def _auto_load_model(self):
        """Automatically load default model weights on startup."""
        model_path = self.settings.get_full_model_path()

        # Check if model file exists
        if not Path(model_path).exists():
            ErrorDialog.show_error(
                title="Model Not Found",
                message="Default model weights file not found at startup.",
                details=f"Expected path: {model_path}",
                solution="Please check the model file path in settings or train the model first. "
                "You can also load a model manually via Model -> Load Model Weights.",
                parent=self,
            )
            self.statusBar().showMessage("Model not found - please load manually", 5000)
            return

        # Load the model
        self._load_model_weights(model_path)

    def _show_model_info(self):
        """Show information about the currently loaded model."""
        info = self.model_loader.get_model_info()

        if not info["loaded"]:
            QMessageBox.information(self, "Model Info", "No model is currently loaded.")
            return

        # Format file size
        size_mb = info["file_size"] / (1024 * 1024)

        info_text = f"""
        <b>Model Information</b><br><br>
        <b>Filename:</b> {info["filename"]}<br>
        <b>Path:</b> {info["weights_path"]}<br>
        <b>File Size:</b> {size_mb:.2f} MB<br>
        """

        QMessageBox.information(self, "Model Info", info_text)

    def _show_about(self):
        """Show about dialog."""
        about_text = """
        <h2>Zero-DCE Image Enhancement</h2>
        <p><b>Version:</b> 1.0.0</p>
        <p>A GUI application for enhancing low-light images using the 
        Zero-Reference Deep Curve Estimation (Zero-DCE) method.</p>
        <p><b>Paper:</b> Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (CVPR 2020)</p>
        <p><b>Paper Authors:</b> Chunle Guo, Chongyi Li, et al.</p>
        <p><b>App Authors:</b> Thanh Thien-An Dang, Hao Hoang Tran</p>
        <p>This is a graduate-level Digital Image Processing course final project in semester 2025-2, Chung-Ang University.</p>
        """

        QMessageBox.about(self, "About Zero-DCE GUI", about_text)

    # ==================== Enhancement ====================

    def _enhance_image(self):
        """Enhance the current input image."""
        if self.current_input_image is None or not self.model_loader.is_model_loaded():
            return

        try:
            # Capture start time for timing measurement
            self._enhancement_start_time = time.perf_counter()

            # Set processing state
            self.enhance_button.set_processing(True)
            self.output_panel.set_processing(True, "Enhancing...")
            self.statusBar().showMessage("Enhancing image...", 0)

            # Get original size
            original_size = self.current_input_image.size

            # Create and start worker thread
            model = self.model_loader.get_model()
            self.enhancement_worker = EnhancementWorker(
                self.current_input_image, model, original_size
            )
            self.enhancement_worker.finished.connect(self._on_enhancement_finished)
            self.enhancement_worker.error.connect(self._on_enhancement_error)
            self.enhancement_worker.start()

        except Exception as e:
            self._on_enhancement_error(e)

    def _on_enhancement_finished(self, enhanced_image):
        """Handle successful enhancement completion.

        Args:
            enhanced_image: Enhanced PIL Image
        """
        try:
            # Calculate elapsed time and create result object
            elapsed = time.perf_counter() - self._enhancement_start_time
            result = EnhancementResult(
                enhanced_image, self._current_enhancement_method, elapsed
            )

            # Store result (future-proof for multi-method comparison)
            self._enhancement_results[self._current_enhancement_method] = result

            # Store enhanced image
            self.current_enhanced_image = enhanced_image

            # Display in output panel with metadata
            pixmap = ImageProcessor.pil_to_pixmap(enhanced_image)

            # Generate display name based on input image
            if self.input_panel.get_image_path():
                input_path = Path(self.input_panel.get_image_path())
                display_name = f"{input_path.stem}_enhanced{input_path.suffix}"
            else:
                display_name = "enhanced_image.png"

            # Pass display_name - will show "In Memory" for size
            self.output_panel.set_image_from_pixmap(
                pixmap, image_path=None, display_name=display_name
            )

            # Update UI state
            self.enhance_button.set_completed()
            self.output_panel.set_processing(False)

            # Get formatted time string
            time_str = result.format_time()

            # Update output panel info overlay with timing
            self.output_panel.set_enhancement_time(time_str)

            # Update status bar with timing information
            self.statusBar().showMessage(f"Enhanced successfully in {time_str}", 5000)

            # Enable save action
            self._update_ui_state()

        except Exception as e:
            self._on_enhancement_error(e)

    def _on_enhancement_error(self, error: Exception):
        """Handle enhancement error.

        Args:
            error: Exception that occurred
        """
        self.enhance_button.set_processing(False)
        self.output_panel.set_processing(False)
        self.statusBar().showMessage("Enhancement failed", 3000)

        ErrorDialog.show_error(
            title="Enhancement Error",
            message="Failed to enhance the image.",
            details=str(error),
            solution="Check that the model is loaded correctly and the image is valid.",
            parent=self,
        )

    # ==================== UI State Management ====================

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        has_image = self.current_input_image is not None
        has_model = self.model_loader.is_model_loaded()
        has_enhanced = self.current_enhanced_image is not None

        # Update enhance button
        self.enhance_button.set_ready(has_image and has_model)

        # Update save action
        self.save_action.setEnabled(has_enhanced)

    # ==================== Window Events ====================

    def closeEvent(self, event):
        """Handle window close event.

        Args:
            event: Close event
        """
        # Save window geometry
        self.settings.set_window_geometry(self.pos(), self.size())
        self.settings.sync()

        # Unload model to free memory
        if self.model_loader.is_model_loaded():
            self.model_loader.unload_model()

        event.accept()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts.

        Args:
            event: Key press event
        """
        # Ctrl+E to enhance
        if (
            event.modifiers() == Qt.KeyboardModifier.ControlModifier
            and event.key() == Qt.Key.Key_E
        ):
            if self.enhance_button.is_ready:
                self._enhance_image()
        else:
            super().keyPressEvent(event)
