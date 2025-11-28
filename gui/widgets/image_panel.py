"""Image display panel widget for Zero-DCE GUI.

This widget provides image display functionality with multiple states:
- Empty state (placeholder with clickable area)
- Image loaded state (displays image fitted to panel)
- Processing state (loading indicator)

Supports drag & drop, click to open, and right-click context menu.
"""

from pathlib import Path
from typing import Optional, Callable
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QSizePolicy,
    QMenu,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QRect
from PyQt6.QtGui import QPixmap, QPainter, QColor, QDragEnterEvent, QDropEvent, QPen

from .histogram_overlay import HistogramOverlay


class ImagePanel(QWidget):
    """Widget for displaying images with multiple states.

    This panel can display:
    - Placeholder text when no image is loaded
    - An image fitted to the panel while maintaining aspect ratio
    - Loading indicator during processing

    Signals:
        image_clicked: Emitted when panel is clicked (for opening/saving)
        image_dropped: Emitted when image file is dropped (str: filepath)
        cleared: Emitted when panel is cleared (to notify parent to clear buffers)
    """

    # Signals
    image_clicked = pyqtSignal()
    image_dropped = pyqtSignal(str)
    cleared = pyqtSignal()  # Emitted when panel is cleared

    def __init__(
        self,
        title: str = "Image",
        placeholder_text: str = "Click to open an image",
        parent: Optional[QWidget] = None,
    ):
        """Initialize the image panel.

        Args:
            title: Title label for the panel
            placeholder_text: Text to show when no image is loaded
            parent: Parent widget
        """
        super().__init__(parent)

        self.title = title
        self.placeholder_text = placeholder_text
        self.current_pixmap: Optional[QPixmap] = None
        self.image_path: Optional[str] = None
        self.display_name: Optional[str] = None  # For enhanced images
        self.is_processing = False
        self.enhancement_time: Optional[str] = None  # For displaying enhancement timing

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Set minimum size
        self.setMinimumSize(300, 300)

        # Initialize UI
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Title label
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333333;
                padding: 5px;
            }
            """
        )
        layout.addWidget(self.title_label)

        # Image display label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image_label.setStyleSheet(
            """
            QLabel {
                background-color: #F5F5F5;
                border: 2px dashed #CCCCCC;
                border-radius: 8px;
            }
            """
        )
        layout.addWidget(self.image_label)

        # Info overlay label (filename, dimensions, size)
        # Note: Parent will be set to image_label after it's created
        self.info_label = QLabel(self.image_label)  # Set parent to image_label
        self.info_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom
        )
        self.info_label.setStyleSheet(
            """
            QLabel {
                background-color: rgba(0, 0, 0, 150);
                color: white;
                font-size: 10px;
                padding: 5px;
                border-radius: 4px;
            }
            """
        )
        self.info_label.hide()  # Hidden by default

        # Raise to ensure it's on top of the image
        self.info_label.raise_()

        # Histogram overlay (hidden by default, global toggle controls visibility)
        self.histogram_overlay = HistogramOverlay(self.image_label)
        self.histogram_overlay.hide()
        self.histogram_visible = False
        self.histogram_type = "grayscale"

        # Set placeholder text
        self._show_placeholder()

    def _show_placeholder(self):
        """Show placeholder text when no image is loaded."""
        self.image_label.setText(self.placeholder_text)
        self.image_label.setStyleSheet(
            """
            QLabel {
                background-color: #F5F5F5;
                border: 2px dashed #CCCCCC;
                border-radius: 8px;
                color: #999999;
                font-size: 14px;
            }
            """
        )

    def set_image(self, image_path: str) -> bool:
        """Load and display an image from file.

        Args:
            image_path: Path to image file

        Returns:
            True if image loaded successfully, False otherwise
        """
        try:
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                return False

            self.current_pixmap = pixmap
            self.image_path = image_path
            self.is_processing = False

            # Update display
            self._update_image_display()
            self._update_info_overlay()
            self._refresh_histogram_overlay()

            # Change border style to solid when image is loaded
            self.image_label.setStyleSheet(
                """
                QLabel {
                    background-color: #FFFFFF;
                    border: 2px solid #CCCCCC;
                    border-radius: 8px;
                }
                """
            )

            return True

        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def set_image_from_pixmap(
        self,
        pixmap: QPixmap,
        image_path: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        """Set image from a QPixmap object.

        Args:
            pixmap: QPixmap to display
            image_path: Optional path associated with the image
            display_name: Optional display name (for enhanced images without path)
        """
        if pixmap.isNull():
            return

        self.current_pixmap = pixmap
        self.image_path = image_path
        self.display_name = display_name  # Store display name
        self.is_processing = False

        self._update_image_display()
        # Show overlay if we have a path OR display_name
        if image_path or display_name:
            self._update_info_overlay()
        self._refresh_histogram_overlay()

        self.image_label.setStyleSheet(
            """
            QLabel {
                background-color: #FFFFFF;
                border: 2px solid #CCCCCC;
                border-radius: 8px;
            }
            """
        )

    def _update_image_display(self):
        """Update the image label to show the current pixmap fitted to the panel."""
        if self.current_pixmap is None:
            return

        # Get available size (accounting for margins)
        available_size = self.image_label.size()

        # Scale pixmap to fit while maintaining aspect ratio
        scaled_pixmap = self.current_pixmap.scaled(
            available_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.image_label.setPixmap(scaled_pixmap)

    def _update_info_overlay(self):
        """Update the info overlay with image information."""
        # If no path and no display name, hide overlay
        if self.image_path is None and not hasattr(self, "display_name"):
            self.info_label.hide()
            return

        if self.image_path is None and self.display_name is None:
            self.info_label.hide()
            return

        # Get image dimensions
        width = self.current_pixmap.width()
        height = self.current_pixmap.height()

        # Determine filename and size
        if self.image_path:
            # Real file - use path info
            path = Path(self.image_path)
            filename = path.name
            size_bytes = path.stat().st_size if path.exists() else 0
            size_kb = size_bytes / 1024
            size_text = f"{size_kb:.1f} KB"
        else:
            # Enhanced image - use display info
            filename = (
                self.display_name if hasattr(self, "display_name") else "Enhanced"
            )
            # Always show "In Memory" for unsaved images
            size_text = "In Memory"

        # Format info text
        info_text = f"{filename}\n{width}x{height} px\n{size_text}"

        # Add enhancement timing if available
        if self.enhancement_time is not None:
            info_text += f"\nEnhanced in {self.enhancement_time}"

        self.info_label.setText(info_text)

        # Adjust size to fit content
        self.info_label.adjustSize()

        # Position in bottom-left corner of image_label
        # Add small margin from edges
        x_pos = 10
        y_pos = self.image_label.height() - self.info_label.height() - 10

        # Ensure position is valid (not negative)
        if y_pos < 0:
            y_pos = 10  # Fallback to top if not enough space

        self.info_label.move(x_pos, y_pos)
        self.info_label.show()
        self.info_label.raise_()  # Ensure it's on top

    def clear(self):
        """Clear the current image and return to placeholder state."""
        self.current_pixmap = None
        self.image_path = None
        self.display_name = None
        self.is_processing = False
        self.enhancement_time = None
        self.info_label.hide()
        self.histogram_overlay.hide()
        self._show_placeholder()

        # Emit cleared signal to notify parent
        self.cleared.emit()

    def set_processing(self, processing: bool, message: str = "Processing..."):
        """Set processing state (show loading indicator).

        Args:
            processing: True to show processing state, False to hide
            message: Message to display during processing
        """
        self.is_processing = processing

        if processing:
            self.image_label.setText(f"{message}")
            self.image_label.setStyleSheet(
                """
                QLabel {
                    background-color: #F5F5F5;
                    border: 2px solid #2196F3;
                    border-radius: 8px;
                    color: #2196F3;
                    font-size: 14px;
                }
                """
            )
        else:
            # Return to previous state
            if self.current_pixmap is not None:
                self._update_image_display()
            else:
                self._show_placeholder()

    def get_image_path(self) -> Optional[str]:
        """Get the path of the currently displayed image.

        Returns:
            Image path, or None if no image is loaded
        """
        return self.image_path

    def get_pixmap(self) -> Optional[QPixmap]:
        """Get the current pixmap.

        Returns:
            Current QPixmap, or None if no image is loaded
        """
        return self.current_pixmap

    def has_image(self) -> bool:
        """Check if an image is currently loaded.

        Returns:
            True if an image is loaded, False otherwise
        """
        return self.current_pixmap is not None

    def set_histogram_enabled(self, enabled: bool):
        """Enable or disable histogram overlay (globally controlled)."""
        self.histogram_visible = enabled
        if not enabled:
            self.histogram_overlay.hide()
            return

        self._refresh_histogram_overlay()

    def set_histogram_type(self, histogram_type: str):
        """Update histogram type for this panel."""
        if histogram_type not in {"rgb", "grayscale"}:
            raise ValueError("Histogram type must be 'rgb' or 'grayscale'")
        self.histogram_type = histogram_type
        self.histogram_overlay.set_histogram_type(histogram_type)
        self._refresh_histogram_overlay()

    def _refresh_histogram_overlay(self):
        """Update histogram overlay visibility and data."""
        if not self.histogram_visible or self.current_pixmap is None:
            self.histogram_overlay.hide()
            return

        self.histogram_overlay.set_histogram_type(self.histogram_type)
        self.histogram_overlay.set_pixmap(self.current_pixmap)

        if not self.histogram_overlay.has_custom_position():
            self.histogram_overlay.move_to_default()
        else:
            self.histogram_overlay.ensure_within_parent()

        self.histogram_overlay.show()
        # Ensure overlays stack correctly: histogram below info overlay
        self.histogram_overlay.raise_()
        self.info_label.raise_()

    def set_info_overlay_visible(self, visible: bool):
        """Show or hide the info overlay.

        Args:
            visible: True to show overlay, False to hide
        """
        if visible and self.image_path:
            self._update_info_overlay()
        else:
            self.info_label.hide()

    def set_enhancement_time(self, time_str: Optional[str]):
        """Set the enhancement time to display in info overlay.

        This method is used to display timing information for enhanced images.
        Call this after enhancement completes with the formatted time string.

        Args:
            time_str: Formatted time string (e.g., "2.34s" or "1m 34s"),
                     or None to clear the timing display

        Example:
            >>> panel.set_enhancement_time("2.34s")
            >>> panel.set_enhancement_time(None)  # Clear timing
        """
        self.enhancement_time = time_str
        # Update overlay if image is currently displayed
        if self.current_pixmap is not None:
            self._update_info_overlay()

    # ==================== Event Handlers ====================

    def mousePressEvent(self, event):
        """Handle mouse press events (for click to open).

        Args:
            event: Mouse event
        """
        # Only emit click signal if we have an image or if it's the input panel
        if event.button() == Qt.MouseButton.LeftButton and not self.is_processing:
            # For output/enhanced panel, only emit if there's an image
            if "Enhanced" in self.title or "Output" in self.title:
                if self.has_image():
                    self.image_clicked.emit()
            else:
                # For input panel, always emit (to open file dialog)
                self.image_clicked.emit()

    def contextMenuEvent(self, event):
        """Handle right-click context menu.

        Args:
            event: Context menu event
        """
        menu = QMenu(self)

        if self.has_image():
            # Add "Save Image" action for output panel
            if "Enhanced" in self.title or "Output" in self.title:
                save_action = menu.addAction("Save Image")
                save_action.triggered.connect(lambda: self._on_context_menu_save())

            # Add "Open Different Image" for input panel
            if "Input" in self.title:
                open_action = menu.addAction("Open Different Image")
                open_action.triggered.connect(lambda: self.image_clicked.emit())

            # Add "Clear" action
            clear_action = menu.addAction("Clear")
            clear_action.triggered.connect(self.clear)
        else:
            # No image loaded - only show "Open Image" for input panel
            if "Input" in self.title:
                open_action = menu.addAction("Open Image")
                open_action.triggered.connect(lambda: self.image_clicked.emit())

        # Only show menu if it has actions
        if not menu.isEmpty():
            menu.exec(event.globalPos())

    def _on_context_menu_save(self):
        """Handle save action from context menu."""
        # This will be connected to the main window's save functionality
        # For now, just emit a custom signal
        pass

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event (for drag & drop).

        Args:
            event: Drag enter event
        """
        if event.mimeData().hasUrls():
            # Check if any of the URLs are image files
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    filepath = url.toLocalFile()
                    if self._is_image_file(filepath):
                        event.acceptProposedAction()
                        return

    def dropEvent(self, event: QDropEvent):
        """Handle drop event (for drag & drop).

        Args:
            event: Drop event
        """
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    filepath = url.toLocalFile()
                    if self._is_image_file(filepath):
                        self.image_dropped.emit(filepath)
                        event.acceptProposedAction()
                        return

    def _is_image_file(self, filepath: str) -> bool:
        """Check if file is a supported image format.

        Args:
            filepath: Path to file

        Returns:
            True if file is a supported image
        """
        supported_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        return Path(filepath).suffix.lower() in supported_extensions

    def resizeEvent(self, event):
        """Handle resize event to update image display.

        Args:
            event: Resize event
        """
        super().resizeEvent(event)
        if self.current_pixmap is not None:
            self._update_image_display()
        if self.info_label.isVisible():
            self._update_info_overlay()
        if self.histogram_visible and self.current_pixmap is not None:
            if not self.histogram_overlay.has_custom_position():
                self.histogram_overlay.move_to_default()
            else:
                self.histogram_overlay.ensure_within_parent()
