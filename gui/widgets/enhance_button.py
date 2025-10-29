"""Custom enhance button widget for Zero-DCE GUI.

This widget provides a prominent button for triggering image enhancement
with multiple states (disabled, ready, processing, completed).
"""

from typing import Optional
from PyQt6.QtWidgets import QPushButton, QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QIcon, QPalette, QColor


class EnhanceButton(QPushButton):
    """Custom button for triggering image enhancement.

    This button has multiple visual states:
    - Disabled: Grayed out when no image or model loaded
    - Ready: Full color, ready to enhance
    - Processing: Animated, shows enhancement in progress
    - Completed: Brief success animation

    Signals:
        enhance_clicked: Emitted when button is clicked in ready state
    """

    # Signals
    enhance_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the enhance button.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.is_ready = False
        self.is_processing = False

        # Initialize UI
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        # Set button text and icon
        self.setText("→")
        self.setToolTip("Load an image and model first")

        # Set size
        self.setFixedSize(64, 64)

        # Set style
        self._update_style()

        # Connect click event
        self.clicked.connect(self._on_clicked)

    def _update_style(self):
        """Update button style based on current state."""
        if not self.is_ready and not self.is_processing:
            # Disabled state
            self.setStyleSheet(
                """
                QPushButton {
                    background-color: #CCCCCC;
                    color: #666666;
                    border: none;
                    border-radius: 32px;
                    font-size: 24px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #CCCCCC;
                }
                """
            )
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
            self.setEnabled(False)

        elif self.is_processing:
            # Processing state
            self.setStyleSheet(
                """
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    border: none;
                    border-radius: 32px;
                    font-size: 20px;
                    font-weight: bold;
                }
                """
            )
            self.setText("⏳")
            self.setCursor(Qt.CursorShape.WaitCursor)
            self.setEnabled(False)

        else:
            # Ready state
            self.setStyleSheet(
                """
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 32px;
                    font-size: 24px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
                """
            )
            self.setText("→")
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setEnabled(True)

    def set_ready(self, ready: bool):
        """Set button ready state.

        Args:
            ready: True if button should be ready (image and model loaded)
        """
        self.is_ready = ready

        if ready:
            self.setToolTip("Enhance image (Ctrl+E)")
        else:
            self.setToolTip("Load an image and model first")

        self._update_style()

    def set_processing(self, processing: bool):
        """Set button processing state.

        Args:
            processing: True if enhancement is in progress
        """
        self.is_processing = processing

        if processing:
            self.setToolTip("Enhancing...")
        else:
            # Return to ready state
            self.setToolTip("Enhance image (Ctrl+E)")

        self._update_style()

    def set_completed(self):
        """Show brief success animation and return to ready state."""
        # Change to completed state (checkmark)
        self.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 32px;
                font-size: 24px;
                font-weight: bold;
            }
            """
        )
        self.setText("✓")
        self.setEnabled(False)

        # Return to ready state after 1 second
        QTimer.singleShot(1000, self._return_to_ready)

    def _return_to_ready(self):
        """Return button to ready state after completion animation."""
        self.is_processing = False
        self._update_style()

    def _on_clicked(self):
        """Handle button click event."""
        if self.is_ready and not self.is_processing:
            self.enhance_clicked.emit()

    def keyPressEvent(self, event):
        """Handle key press events (Space or Ctrl+E).

        Args:
            event: Key press event
        """
        if event.key() == Qt.Key.Key_Space and self.is_ready:
            self._on_clicked()
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_E:
            if self.is_ready:
                self._on_clicked()
        else:
            super().keyPressEvent(event)
