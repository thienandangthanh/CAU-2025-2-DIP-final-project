"""Comparison cell widget for displaying individual enhancement results.

This widget is used within the ComparisonGrid to display a single enhancement
result with method name, timing information, and status indicator.
"""

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QColor

from .histogram_overlay import HistogramOverlay


class ComparisonCell(QWidget):
    """Widget for displaying a single enhancement result in comparison mode.
    
    This cell displays:
    - Method name label
    - Result image (scaled to fit)
    - Timing information
    - Status indicator (pending/running/done/error)
    
    Signals:
        clicked: Emitted when the cell is clicked (to show expanded view)
    """
    
    # Signals
    clicked = pyqtSignal(str)  # method_key
    
    def __init__(
        self,
        method_key: str,
        method_name: str,
        is_reference: bool = False,
        parent: Optional[QWidget] = None
    ):
        """Initialize the comparison cell.
        
        Args:
            method_key: Unique identifier for the method (e.g., "zero-dce")
            method_name: Display name for the method (e.g., "Zero-DCE")
            is_reference: Whether this is the reference/input cell
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.method_key = method_key
        self.method_name = method_name
        self.is_reference = is_reference
        self.current_pixmap: Optional[QPixmap] = None
        self.status = "pending"  # pending/running/done/error
        self.timing_text: Optional[str] = None
        
        # Set size policy - Expanding to fill space while maintaining similar dimensions
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(250, 300)  # Minimum size for readability
        
        # Enable mouse tracking for click events
        self.setMouseTracking(True)
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Method name label
        self.name_label = QLabel(self.method_name)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet(
            """
            QLabel {
                font-size: 13px;
                font-weight: bold;
                color: #333333;
                padding: 3px;
                background-color: #F0F0F0;
                border-radius: 4px;
            }
            """
        )
        layout.addWidget(self.name_label)
        
        # Image display label - expanding to use available space
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image_label.setMinimumSize(200, 200)  # Minimum for readability
        self.image_label.setStyleSheet(
            """
            QLabel {
                background-color: #F5F5F5;
                border: 2px solid #CCCCCC;
                border-radius: 6px;
            }
            """
        )
        layout.addWidget(self.image_label)

        # Histogram overlay per cell (shares same implementation as ImagePanel)
        self.histogram_overlay = HistogramOverlay(self.image_label)
        self.histogram_overlay.hide()
        self.histogram_visible = False
        self.histogram_type = "grayscale"
        
        # Status/timing label
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            """
            QLabel {
                font-size: 11px;
                color: #666666;
                padding: 2px;
            }
            """
        )
        layout.addWidget(self.status_label)
        
        # Show initial status
        self._update_status_display()
    
    def set_image(self, pixmap: QPixmap):
        """Set the image to display.
        
        Args:
            pixmap: QPixmap to display
        """
        if pixmap.isNull():
            return
        
        self.current_pixmap = pixmap
        self._update_image_display()
        self._refresh_histogram_overlay()
        
        # Update border style when image is loaded
        if not self.is_reference:
            self.image_label.setStyleSheet(
                """
                QLabel {
                    background-color: #FFFFFF;
                    border: 2px solid #4CAF50;
                    border-radius: 6px;
                }
                """
            )
    
    def set_status(self, status: str, timing_text: Optional[str] = None):
        """Set the status and optional timing information.
        
        Args:
            status: Status string ("pending", "running", "done", "error")
            timing_text: Optional timing text (e.g., "2.34s")
        """
        self.status = status
        self.timing_text = timing_text
        self._update_status_display()
        self._update_border_style()
    
    def set_error(self, error_message: str):
        """Set error status with message.
        
        Args:
            error_message: Error message to display
        """
        self.status = "error"
        self.timing_text = None
        self._update_status_display()
        self._update_border_style()
        
        # Show error in image label
        self.image_label.setText(f"Error: {error_message}")
        self.image_label.setStyleSheet(
            """
            QLabel {
                background-color: #FFEBEE;
                border: 2px solid #F44336;
                border-radius: 6px;
                color: #C62828;
                font-size: 12px;
                padding: 10px;
            }
            """
        )
    
    def _update_status_display(self):
        """Update the status label text based on current status."""
        if self.is_reference:
            # Reference cell shows different text
            if self.current_pixmap:
                self.status_label.setText("Original")
            else:
                self.status_label.setText("Waiting...")
            return
        
        # Enhancement method cell
        if self.status == "pending":
            self.status_label.setText("Pending...")
        elif self.status == "running":
            self.status_label.setText("Processing...")
        elif self.status == "done":
            if self.timing_text:
                self.status_label.setText(f"Completed in {self.timing_text}")
            else:
                self.status_label.setText("Completed")
        elif self.status == "error":
            self.status_label.setText("Error")
    
    def _update_border_style(self):
        """Update border style based on status."""
        if self.is_reference or not self.current_pixmap:
            return
        
        if self.status == "running":
            border_color = "#2196F3"  # Blue for running
        elif self.status == "done":
            border_color = "#4CAF50"  # Green for done
        elif self.status == "error":
            border_color = "#F44336"  # Red for error
        else:
            border_color = "#CCCCCC"  # Gray for pending
        
        self.image_label.setStyleSheet(
            f"""
            QLabel {{
                background-color: #FFFFFF;
                border: 2px solid {border_color};
                border-radius: 6px;
            }}
            """
        )
    
    def _update_image_display(self):
        """Update the image label to show the current pixmap fitted to the cell."""
        if self.current_pixmap is None:
            return
        
        # Get available size
        available_size = self.image_label.size()
        
        # Scale pixmap to fit while maintaining aspect ratio
        scaled_pixmap = self.current_pixmap.scaled(
            available_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        self._refresh_histogram_overlay()
    
    def clear(self):
        """Clear the cell and return to initial state."""
        self.current_pixmap = None
        self.status = "pending"
        self.timing_text = None
        self.image_label.clear()
        self.image_label.setStyleSheet(
            """
            QLabel {
                background-color: #F5F5F5;
                border: 2px solid #CCCCCC;
                border-radius: 6px;
            }
            """
        )
        self._update_status_display()
        self.histogram_overlay.hide()
    
    def mousePressEvent(self, event):
        """Handle mouse press events for cell click.
        
        Args:
            event: Mouse event
        """
        if event.button() == Qt.MouseButton.LeftButton and self.current_pixmap:
            self.clicked.emit(self.method_key)
    
    def resizeEvent(self, event):
        """Handle resize event to update image display.
        
        Args:
            event: Resize event
        """
        super().resizeEvent(event)
        if self.current_pixmap is not None:
            self._update_image_display()

    def set_histogram_enabled(self, enabled: bool):
        """Enable or disable histogram overlay for this cell."""
        self.histogram_visible = enabled
        if not enabled:
            self.histogram_overlay.hide()
            return
        self._refresh_histogram_overlay()

    def set_histogram_type(self, histogram_type: str):
        """Set histogram type (RGB or Grayscale)."""
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
