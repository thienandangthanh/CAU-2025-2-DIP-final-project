"""Model loading and management for Zero-DCE GUI.

This module handles loading and validating Zero-DCE model weights.
Reuses logic from compare.py's load_model_for_inference() function.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path

import keras

from model import ZeroDCE


class ModelLoader:
    """Handles loading and managing Zero-DCE model weights.

    This class provides a simple interface for loading trained Zero-DCE models
    and validating weight files before loading.

    Attributes:
        current_model (Optional[ZeroDCE]): Currently loaded model instance
        current_weights_path (Optional[str]): Path to currently loaded weights
    """

    def __init__(self):
        """Initialize ModelLoader with no model loaded."""
        self.current_model: ZeroDCE | None = None
        self.current_weights_path: str | None = None

    def load_model(self, weights_path: str) -> ZeroDCE:
        """Load Zero-DCE model with specified weights.

        Creates a new ZeroDCE model instance and loads the weights from
        the specified file. Validates that the file exists before loading.

        Args:
            weights_path: Path to model weights file (.h5 or .weights.h5)

        Returns:
            Loaded ZeroDCE model ready for inference

        Raises:
            FileNotFoundError: If weights file doesn't exist
            ValueError: If weights file has invalid format
            Exception: If model loading fails for other reasons
        """
        # Validate file exists
        weights_path = str(Path(weights_path).resolve())
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {weights_path}")

        # Validate file extension
        valid_extensions = [".h5", ".weights.h5"]
        if not any(weights_path.endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Invalid weights file format. Expected {valid_extensions}, "
                f"got: {Path(weights_path).suffix}"
            )

        try:
            # Create model instance
            model = ZeroDCE()

            # Load weights
            model.load_weights(weights_path)

            # Store current model and path
            self.current_model = model
            self.current_weights_path = weights_path

            return model

        except Exception as e:
            raise Exception(f"Failed to load model weights: {str(e)}") from e

    def get_model(self) -> ZeroDCE | None:
        """Get the currently loaded model.

        Returns:
            Currently loaded ZeroDCE model, or None if no model is loaded
        """
        return self.current_model

    def get_weights_path(self) -> str | None:
        """Get the path to currently loaded weights.

        Returns:
            Path to current weights file, or None if no model is loaded
        """
        return self.current_weights_path

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded.

        Returns:
            True if a model is loaded, False otherwise
        """
        return self.current_model is not None

    def unload_model(self) -> None:
        """Unload the current model and free memory."""
        self.current_model = None
        self.current_weights_path = None
        # Force garbage collection to free GPU memory
        import gc

        gc.collect()

        # Clear Keras backend session if using TensorFlow
        try:
            keras.backend.clear_session()
        except Exception:
            pass

    def get_model_info(self) -> dict:
        """Get information about the currently loaded model.

        Returns:
            Dictionary containing model information:
            - loaded: bool, whether a model is loaded
            - weights_path: str or None, path to weights file
            - file_size: int or None, size of weights file in bytes
            - filename: str or None, name of weights file
        """
        if not self.is_model_loaded():
            return {
                "loaded": False,
                "weights_path": None,
                "file_size": None,
                "filename": None,
            }

        weights_file = Path(self.current_weights_path)
        return {
            "loaded": True,
            "weights_path": str(weights_file),
            "file_size": weights_file.stat().st_size,
            "filename": weights_file.name,
        }

    @staticmethod
    def validate_weights_file(weights_path: str) -> tuple[bool, str]:
        """Validate a weights file without loading it.

        Checks if the file exists and has a valid extension.

        Args:
            weights_path: Path to weights file to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if file is valid, False otherwise
            - error_message: Empty string if valid, error description if invalid
        """
        weights_path = Path(weights_path)

        # Check if file exists
        if not weights_path.exists():
            return False, f"File not found: {weights_path}"

        # Check if it's a file (not a directory)
        if not weights_path.is_file():
            return False, f"Path is not a file: {weights_path}"

        # Check file extension
        valid_extensions = [".h5", ".weights.h5"]
        if not any(str(weights_path).endswith(ext) for ext in valid_extensions):
            return False, f"Invalid file format. Expected {valid_extensions}"

        # Check file size (should be > 0)
        if weights_path.stat().st_size == 0:
            return False, "File is empty"

        return True, ""

    @staticmethod
    def list_available_weights(weights_dir: str = "./weights") -> list[str]:
        """List all available weight files in a directory.

        Scans the specified directory for valid weight files (.h5, .weights.h5).

        Args:
            weights_dir: Directory to scan for weight files (default: "./weights")

        Returns:
            List of full paths to valid weight files, sorted by modification time (newest first)
        """
        weights_path = Path(weights_dir)

        if not weights_path.exists() or not weights_path.is_dir():
            return []

        # Find all .h5 files (this includes .weights.h5 files)
        weight_files = list(weights_path.glob("*.h5"))

        # Sort by modification time (newest first)
        weight_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return [str(f) for f in weight_files]
