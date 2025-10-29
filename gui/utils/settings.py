"""Application settings management using QSettings.

This module provides a centralized way to manage application preferences,
including default model paths, recent files, window geometry, and user preferences.
Uses Qt's QSettings for cross-platform persistent storage.
"""

from pathlib import Path
from typing import Optional, List
from PyQt6.QtCore import QSettings, QSize, QPoint


class AppSettings:
    """Manages application settings using QSettings.

    Settings are stored in platform-specific locations:
    - Windows: Registry (HKEY_CURRENT_USER\\Software\\ZeroDCE\\GUI)
    - macOS: ~/Library/Preferences/com.zerodce.gui.plist
    - Linux: ~/.config/ZeroDCE/GUI.conf

    Attributes:
        settings (QSettings): Qt settings object for persistent storage
    """

    # Default values
    DEFAULT_WEIGHTS_DIR = "./weights"
    DEFAULT_MODEL_FILE = "zero_dce.weights.h5"
    MAX_RECENT_FILES = 10
    DEFAULT_MAX_IMAGE_DIMENSION = 4096
    DEFAULT_JPEG_QUALITY = 95
    DEFAULT_ZOOM_MODE = "fit"  # "fit" or "actual"
    DEFAULT_SHOW_INFO_OVERLAY = True
    DEFAULT_SYNC_ZOOM = True
    DEFAULT_GPU_MODE = "auto"  # "auto", "enable", "disable"
    DEFAULT_ENHANCEMENT_ITERATIONS = 8
    DEFAULT_OUTPUT_FORMAT = "PNG"  # "PNG" or "JPEG"
    DEFAULT_AUTO_CLEAR_CACHE = True
    DEFAULT_DEBUG_LOGGING = False

    def __init__(self):
        """Initialize settings with organization and application name."""
        self.settings = QSettings("ZeroDCE", "GUI")

    # ==================== Model Settings ====================

    def get_weights_directory(self) -> str:
        """Get the default weights directory.

        Returns:
            Path to weights directory (default: "./weights")
        """
        return self.settings.value("model/weights_dir", self.DEFAULT_WEIGHTS_DIR)

    def set_weights_directory(self, directory: str) -> None:
        """Set the default weights directory.

        Args:
            directory: Path to weights directory
        """
        self.settings.setValue("model/weights_dir", directory)

    def get_default_model_file(self) -> str:
        """Get the default model filename.

        Returns:
            Model filename (default: "zero_dce.weights.h5")
        """
        return self.settings.value("model/default_file", self.DEFAULT_MODEL_FILE)

    def set_default_model_file(self, filename: str) -> None:
        """Set the default model filename.

        Args:
            filename: Model filename
        """
        self.settings.setValue("model/default_file", filename)

    def get_full_model_path(self) -> str:
        """Construct full model path from directory and filename.

        Returns:
            Full path to model weights file (e.g., "./weights/zero_dce.weights.h5")
        """
        return str(Path(self.get_weights_directory()) / self.get_default_model_file())

    def get_auto_load_model(self) -> bool:
        """Check if model should be auto-loaded on startup.

        Returns:
            True if auto-load is enabled
        """
        return self.settings.value("model/auto_load", True, type=bool)

    def set_auto_load_model(self, enabled: bool) -> None:
        """Set whether to auto-load model on startup.

        Args:
            enabled: True to enable auto-loading
        """
        self.settings.setValue("model/auto_load", enabled)

    def get_enhancement_iterations(self) -> int:
        """Get number of enhancement iterations.

        Returns:
            Number of iterations (default: 8)
        """
        return self.settings.value(
            "model/iterations", self.DEFAULT_ENHANCEMENT_ITERATIONS, type=int
        )

    def set_enhancement_iterations(self, iterations: int) -> None:
        """Set number of enhancement iterations.

        Args:
            iterations: Number of iterations (typically 8)
        """
        self.settings.setValue("model/iterations", iterations)

    # ==================== Recent Files ====================

    def get_recent_files(self) -> List[str]:
        """Get list of recently opened files.

        Returns:
            List of file paths, most recent first (max 10)
        """
        recent = self.settings.value("recent_files/list", [])
        # Handle case where settings returns a single string instead of list
        if isinstance(recent, str):
            return [recent] if recent else []
        return recent if recent else []

    def add_recent_file(self, filepath: str) -> None:
        """Add a file to recent files list.

        Args:
            filepath: Path to file to add
        """
        recent = self.get_recent_files()

        # Remove if already exists (to move to top)
        if filepath in recent:
            recent.remove(filepath)

        # Add to beginning
        recent.insert(0, filepath)

        # Limit to MAX_RECENT_FILES
        recent = recent[: self.MAX_RECENT_FILES]

        self.settings.setValue("recent_files/list", recent)

    def clear_recent_files(self) -> None:
        """Clear all recent files."""
        self.settings.setValue("recent_files/list", [])

    # ==================== Window Geometry ====================

    def get_window_geometry(self) -> tuple[Optional[QPoint], Optional[QSize]]:
        """Get saved window position and size.

        Returns:
            Tuple of (position, size), or (None, None) if not saved
        """
        pos = self.settings.value("window/position", None)
        size = self.settings.value("window/size", None)
        return pos, size

    def set_window_geometry(self, pos: QPoint, size: QSize) -> None:
        """Save window position and size.

        Args:
            pos: Window position
            size: Window size
        """
        self.settings.setValue("window/position", pos)
        self.settings.setValue("window/size", size)

    # ==================== Image Display Settings ====================

    def get_default_zoom_mode(self) -> str:
        """Get default zoom mode.

        Returns:
            "fit" or "actual"
        """
        return self.settings.value("display/zoom_mode", self.DEFAULT_ZOOM_MODE)

    def set_default_zoom_mode(self, mode: str) -> None:
        """Set default zoom mode.

        Args:
            mode: "fit" or "actual"
        """
        if mode not in ["fit", "actual"]:
            raise ValueError("Zoom mode must be 'fit' or 'actual'")
        self.settings.setValue("display/zoom_mode", mode)

    def get_sync_zoom(self) -> bool:
        """Check if zoom should be synchronized between panels.

        Returns:
            True if zoom should be synchronized
        """
        return self.settings.value(
            "display/sync_zoom", self.DEFAULT_SYNC_ZOOM, type=bool
        )

    def set_sync_zoom(self, enabled: bool) -> None:
        """Set whether to synchronize zoom between panels.

        Args:
            enabled: True to enable synchronized zoom
        """
        self.settings.setValue("display/sync_zoom", enabled)

    def get_show_info_overlay(self) -> bool:
        """Check if info overlay should be shown.

        Returns:
            True if info overlay should be shown
        """
        return self.settings.value(
            "display/show_info", self.DEFAULT_SHOW_INFO_OVERLAY, type=bool
        )

    def set_show_info_overlay(self, enabled: bool) -> None:
        """Set whether to show info overlay.

        Args:
            enabled: True to show info overlay
        """
        self.settings.setValue("display/show_info", enabled)

    # ==================== Performance Settings ====================

    def get_gpu_mode(self) -> str:
        """Get GPU acceleration mode.

        Returns:
            "auto", "enable", or "disable"
        """
        return self.settings.value("performance/gpu_mode", self.DEFAULT_GPU_MODE)

    def set_gpu_mode(self, mode: str) -> None:
        """Set GPU acceleration mode.

        Args:
            mode: "auto", "enable", or "disable"
        """
        if mode not in ["auto", "enable", "disable"]:
            raise ValueError("GPU mode must be 'auto', 'enable', or 'disable'")
        self.settings.setValue("performance/gpu_mode", mode)

    def get_max_image_dimension(self) -> int:
        """Get maximum image dimension.

        Returns:
            Maximum dimension in pixels
        """
        return self.settings.value(
            "performance/max_dimension", self.DEFAULT_MAX_IMAGE_DIMENSION, type=int
        )

    def set_max_image_dimension(self, dimension: int) -> None:
        """Set maximum image dimension.

        Args:
            dimension: Maximum dimension in pixels
        """
        self.settings.setValue("performance/max_dimension", dimension)

    # ==================== Output Settings ====================

    def get_output_format(self) -> str:
        """Get default output image format.

        Returns:
            "PNG" or "JPEG"
        """
        return self.settings.value("output/format", self.DEFAULT_OUTPUT_FORMAT)

    def set_output_format(self, format: str) -> None:
        """Set default output image format.

        Args:
            format: "PNG" or "JPEG"
        """
        if format not in ["PNG", "JPEG"]:
            raise ValueError("Output format must be 'PNG' or 'JPEG'")
        self.settings.setValue("output/format", format)

    def get_jpeg_quality(self) -> int:
        """Get JPEG quality setting.

        Returns:
            Quality value (0-100)
        """
        return self.settings.value(
            "output/jpeg_quality", self.DEFAULT_JPEG_QUALITY, type=int
        )

    def set_jpeg_quality(self, quality: int) -> None:
        """Set JPEG quality setting.

        Args:
            quality: Quality value (0-100)
        """
        if not 0 <= quality <= 100:
            raise ValueError("JPEG quality must be between 0 and 100")
        self.settings.setValue("output/jpeg_quality", quality)

    # ==================== Advanced Settings ====================

    def get_cache_directory(self) -> str:
        """Get cache directory path.

        Returns:
            Path to cache directory
        """
        default_cache = str(Path.home() / ".cache" / "zero-dce-gui")
        return self.settings.value("advanced/cache_dir", default_cache)

    def set_cache_directory(self, path: str) -> None:
        """Set cache directory path.

        Args:
            path: Path to cache directory
        """
        self.settings.setValue("advanced/cache_dir", path)

    def get_auto_clear_cache(self) -> bool:
        """Check if cache should be cleared on exit.

        Returns:
            True if auto-clear is enabled
        """
        return self.settings.value(
            "advanced/auto_clear_cache", self.DEFAULT_AUTO_CLEAR_CACHE, type=bool
        )

    def set_auto_clear_cache(self, enabled: bool) -> None:
        """Set whether to auto-clear cache on exit.

        Args:
            enabled: True to enable auto-clear
        """
        self.settings.setValue("advanced/auto_clear_cache", enabled)

    def get_debug_logging(self) -> bool:
        """Check if debug logging is enabled.

        Returns:
            True if debug logging is enabled
        """
        return self.settings.value(
            "advanced/debug_logging", self.DEFAULT_DEBUG_LOGGING, type=bool
        )

    def set_debug_logging(self, enabled: bool) -> None:
        """Set whether to enable debug logging.

        Args:
            enabled: True to enable debug logging
        """
        self.settings.setValue("advanced/debug_logging", enabled)

    def get_log_directory(self) -> str:
        """Get log directory path.

        Returns:
            Path to log directory
        """
        default_log = str(Path.home() / ".local" / "share" / "zero-dce-gui" / "logs")
        return self.settings.value("advanced/log_dir", default_log)

    def set_log_directory(self, path: str) -> None:
        """Set log directory path.

        Args:
            path: Path to log directory
        """
        self.settings.setValue("advanced/log_dir", path)

    # ==================== Utility Methods ====================

    def reset_to_defaults(self) -> None:
        """Reset all settings to default values."""
        self.settings.clear()

    def sync(self) -> None:
        """Force synchronization of settings to persistent storage."""
        self.settings.sync()
