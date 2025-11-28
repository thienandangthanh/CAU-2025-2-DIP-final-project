"""Error dialog for displaying user-friendly error messages.

This module provides a dialog for showing errors with technical details
and suggested solutions.
"""

from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ErrorDialog(QDialog):
    """Dialog for displaying error messages with details.

    Shows:
    - User-friendly error message
    - Technical details (expandable)
    - Suggested solutions
    - Copy error button for bug reports
    """

    def __init__(
        self,
        title: str,
        message: str,
        details: str | None = None,
        solution: str | None = None,
        parent: QWidget | None = None,
    ):
        """Initialize the error dialog.

        Args:
            title: Dialog window title
            message: User-friendly error message
            details: Technical error details (optional)
            solution: Suggested solution (optional)
            parent: Parent widget
        """
        super().__init__(parent)

        self.title = title
        self.message = message
        self.details = details
        self.solution = solution

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(self.title)
        self.setMinimumWidth(500)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Error icon and message
        message_layout = QHBoxLayout()

        # Error icon (⚠️ or ❌)
        icon_label = QLabel("❌")
        icon_label.setStyleSheet(
            """
            QLabel {
                font-size: 32px;
                padding: 10px;
            }
            """
        )
        message_layout.addWidget(icon_label)

        # Error message
        message_label = QLabel(self.message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                color: #333333;
                padding: 10px;
            }
            """
        )
        message_layout.addWidget(message_label, 1)

        layout.addLayout(message_layout)

        # Suggested solution (if provided)
        if self.solution:
            solution_label = QLabel(f"<b>Solution:</b> {self.solution}")
            solution_label.setWordWrap(True)
            solution_label.setStyleSheet(
                """
                QLabel {
                    background-color: #E3F2FD;
                    color: #1976D2;
                    padding: 10px;
                    border-radius: 4px;
                    font-size: 12px;
                }
                """
            )
            layout.addWidget(solution_label)

        # Technical details (expandable)
        if self.details:
            self.details_widget = QTextEdit()
            self.details_widget.setReadOnly(True)
            self.details_widget.setPlainText(self.details)
            self.details_widget.setMaximumHeight(150)
            self.details_widget.setStyleSheet(
                """
                QTextEdit {
                    background-color: #F5F5F5;
                    font-family: monospace;
                    font-size: 10px;
                    border: 1px solid #CCCCCC;
                    border-radius: 4px;
                }
                """
            )
            self.details_widget.hide()  # Hidden by default

            # Toggle details button
            self.toggle_details_btn = QPushButton("Show Technical Details")
            self.toggle_details_btn.clicked.connect(self._toggle_details)
            self.toggle_details_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: transparent;
                    color: #1976D2;
                    border: none;
                    text-align: left;
                    padding: 5px;
                    text-decoration: underline;
                }
                QPushButton:hover {
                    color: #0D47A1;
                }
                """
            )
            layout.addWidget(self.toggle_details_btn)
            layout.addWidget(self.details_widget)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Copy error button (if details available)
        if self.details:
            copy_btn = QPushButton("Copy Error")
            copy_btn.clicked.connect(self._copy_error)
            copy_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #757575;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #616161;
                }
                """
            )
            button_layout.addWidget(copy_btn)

        # OK button
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        ok_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            """
        )
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

    def _toggle_details(self):
        """Toggle visibility of technical details."""
        if self.details_widget.isVisible():
            self.details_widget.hide()
            self.toggle_details_btn.setText("Show Technical Details")
        else:
            self.details_widget.show()
            self.toggle_details_btn.setText("Hide Technical Details")

    def _copy_error(self):
        """Copy error message and details to clipboard."""
        from PyQt6.QtWidgets import QApplication

        error_text = (
            f"{self.title}\n\n{self.message}\n\nTechnical Details:\n{self.details}"
        )
        clipboard = QApplication.clipboard()
        clipboard.setText(error_text)

        # Change button text temporarily to show success
        sender = self.sender()
        original_text = sender.text()
        sender.setText("Copied!")
        sender.setEnabled(False)

        # Reset after 1 second
        from PyQt6.QtCore import QTimer

        def reset_button():
            sender.setText(original_text)
            sender.setEnabled(True)

        QTimer.singleShot(1000, reset_button)

    @staticmethod
    def show_error(
        title: str,
        message: str,
        details: str | None = None,
        solution: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Show an error dialog (convenience method).

        Args:
            title: Dialog window title
            message: User-friendly error message
            details: Technical error details (optional)
            solution: Suggested solution (optional)
            parent: Parent widget
        """
        dialog = ErrorDialog(title, message, details, solution, parent)
        dialog.exec()

    # ==================== Common Error Types ====================

    @staticmethod
    def show_model_error(error: Exception, parent: QWidget | None = None):
        """Show a model loading/inference error.

        Args:
            error: Exception that occurred
            parent: Parent widget
        """
        ErrorDialog.show_error(
            title="Model Error",
            message="Failed to load or use the Zero-DCE model.",
            details=str(error),
            solution="Check that the model weights file is valid and compatible.",
            parent=parent,
        )

    @staticmethod
    def show_image_error(error: Exception, parent: QWidget | None = None):
        """Show an image loading/saving error.

        Args:
            error: Exception that occurred
            parent: Parent widget
        """
        ErrorDialog.show_error(
            title="Image Error",
            message="Failed to load or save the image.",
            details=str(error),
            solution="Check that the image file is valid and in a supported format (JPEG, PNG).",
            parent=parent,
        )

    @staticmethod
    def show_file_error(error: Exception, parent: QWidget | None = None):
        """Show a file I/O error.

        Args:
            error: Exception that occurred
            parent: Parent widget
        """
        ErrorDialog.show_error(
            title="File Error",
            message="Failed to access the file.",
            details=str(error),
            solution="Check file permissions and available disk space.",
            parent=parent,
        )

    @staticmethod
    def show_memory_error(parent: QWidget | None = None):
        """Show an out of memory error.

        Args:
            parent: Parent widget
        """
        ErrorDialog.show_error(
            title="Out of Memory",
            message="The application ran out of memory during processing.",
            solution="Try reducing the image size or closing other applications.",
            parent=parent,
        )
