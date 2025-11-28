#!/usr/bin/env python3
"""Entry point for Zero-DCE GUI application.

This script launches the PyQt6 GUI application for enhancing low-light images
using the Zero-DCE (Zero-Reference Deep Curve Estimation) method.

Usage:
    python gui_app.py
"""

import os
import sys

# Set Keras backend before any imports
os.environ["KERAS_BACKEND"] = "tensorflow"

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main():
    """Main entry point for the GUI application."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Zero-DCE GUI")
    app.setOrganizationName("ZeroDCE")
    app.setApplicationDisplayName("Zero-DCE Image Enhancement")

    # Create and show main window
    main_window = MainWindow()
    main_window.show()

    # Run application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
