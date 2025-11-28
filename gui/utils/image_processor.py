"""Image processing utilities for Zero-DCE GUI.

This module handles image I/O, preprocessing for the model, and post-processing
of enhanced images. Reuses logic from compare.py's infer() function.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path

import numpy as np
from PIL import Image
from PyQt6.QtGui import QImage, QPixmap

from model import ZeroDCE


class ImageProcessor:
    """Handles image loading, preprocessing, and enhancement.

    This class provides utilities for:
    - Loading images from files
    - Preprocessing images for Zero-DCE model
    - Running inference through the model
    - Post-processing enhanced images
    - Converting between PIL, numpy, and QPixmap formats
    """

    # Image size limits
    MIN_DIMENSION = 64
    MAX_DIMENSION = 8192
    SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]

    @staticmethod
    def load_image(filepath: str, max_dimension: int | None = None) -> Image.Image:
        """Load an image from file.

        Args:
            filepath: Path to image file
            max_dimension: Maximum dimension (resize if exceeded), or None for no limit

        Returns:
            PIL Image in RGB mode

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or image is invalid
        """
        filepath = Path(filepath)

        # Check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"Image file not found: {filepath}")

        # Check file extension
        if filepath.suffix.lower() not in ImageProcessor.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {filepath.suffix}. "
                f"Supported formats: {ImageProcessor.SUPPORTED_FORMATS}"
            )

        try:
            # Load image with PIL
            image = Image.open(filepath)

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Check dimensions
            width, height = image.size
            if (
                width < ImageProcessor.MIN_DIMENSION
                or height < ImageProcessor.MIN_DIMENSION
            ):
                raise ValueError(
                    f"Image too small: {width}x{height}. "
                    f"Minimum dimension: {ImageProcessor.MIN_DIMENSION}px"
                )

            # Resize if exceeds max dimension
            if max_dimension is not None:
                max_dim = max(width, height)
                if max_dim > max_dimension:
                    # Calculate new size while preserving aspect ratio
                    scale = max_dimension / max_dim
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = image.resize(
                        (new_width, new_height), Image.Resampling.LANCZOS
                    )

            return image

        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}") from e

    @staticmethod
    def preprocess_image(image: Image.Image) -> np.ndarray:
        """Preprocess image for Zero-DCE model.

        Converts PIL image to numpy array and normalizes to [0, 1] range.

        Args:
            image: PIL Image in RGB mode

        Returns:
            Preprocessed numpy array with shape (1, H, W, 3) and values in [0, 1]
        """
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)

        # Normalize to [0, 1]
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    @staticmethod
    def postprocess_image(image_array: np.ndarray) -> Image.Image:
        """Post-process model output to PIL Image.

        Converts numpy array from model output back to PIL Image.

        Args:
            image_array: Numpy array with shape (1, H, W, 3) and values in [0, 1]

        Returns:
            PIL Image in RGB mode
        """
        # Remove batch dimension
        if image_array.ndim == 4:
            image_array = image_array[0]

        # Clip values to [0, 1] range
        image_array = np.clip(image_array, 0.0, 1.0)

        # Denormalize to [0, 255]
        image_array = (image_array * 255.0).astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(image_array, mode="RGB")

        return image

    @staticmethod
    def enhance_image(
        image: Image.Image, model: ZeroDCE, original_size: tuple[int, int] | None = None
    ) -> Image.Image:
        """Enhance an image using the Zero-DCE model.

        Args:
            image: Input PIL Image
            model: Loaded ZeroDCE model
            original_size: Optional (width, height) to resize output to

        Returns:
            Enhanced PIL Image
        """
        # Store original size for later
        if original_size is None:
            original_size = image.size

        # Preprocess image
        input_array = ImageProcessor.preprocess_image(image)

        # Run inference
        enhanced_array = model.call(input_array)

        # Convert to numpy if it's a tensor
        if hasattr(enhanced_array, "numpy"):
            enhanced_array = enhanced_array.numpy()

        # Post-process
        enhanced_image = ImageProcessor.postprocess_image(enhanced_array)

        # Resize to original size if needed
        if enhanced_image.size != original_size:
            enhanced_image = enhanced_image.resize(
                original_size, Image.Resampling.LANCZOS
            )

        return enhanced_image

    @staticmethod
    def save_image(
        image: Image.Image,
        filepath: str,
        format: str = "PNG",
        quality: int = 95,
        preserve_exif: bool = False,
    ) -> None:
        """Save an image to file.

        Args:
            image: PIL Image to save
            filepath: Output file path
            format: Image format ("PNG" or "JPEG")
            quality: JPEG quality (0-100), ignored for PNG
            preserve_exif: Whether to preserve EXIF metadata (if available)

        Raises:
            ValueError: If format is invalid
            Exception: If save operation fails
        """
        if format.upper() not in ["PNG", "JPEG", "JPG"]:
            raise ValueError(f"Invalid format: {format}. Use 'PNG' or 'JPEG'")

        try:
            save_kwargs = {}

            if format.upper() in ["JPEG", "JPG"]:
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True

            if preserve_exif and hasattr(image, "info") and "exif" in image.info:
                save_kwargs["exif"] = image.info["exif"]

            image.save(filepath, format=format.upper(), **save_kwargs)

        except Exception as e:
            raise Exception(f"Failed to save image: {str(e)}") from e

    @staticmethod
    def pil_to_pixmap(image: Image.Image) -> QPixmap:
        """Convert PIL Image to QPixmap.

        Args:
            image: PIL Image in RGB mode

        Returns:
            QPixmap for display in Qt widgets
        """
        # Convert PIL Image to numpy array
        image_array = np.array(image)

        # Get dimensions
        height, width, channels = image_array.shape

        # Convert to QImage
        bytes_per_line = channels * width
        qimage = QImage(
            image_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        # Convert to QPixmap
        return QPixmap.fromImage(qimage)

    @staticmethod
    def pixmap_to_pil(pixmap: QPixmap) -> Image.Image:
        """Convert QPixmap to PIL Image.

        Args:
            pixmap: QPixmap from Qt widgets

        Returns:
            PIL Image in RGB mode
        """
        # Convert to QImage
        qimage = pixmap.toImage()

        # Convert to RGB32 format if not already
        qimage = qimage.convertToFormat(QImage.Format.Format_RGB32)

        # Get dimensions
        width = qimage.width()
        height = qimage.height()

        # Get bytes
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

        # Convert RGBA to RGB
        rgb_array = arr[:, :, :3]

        # Create PIL Image
        return Image.fromarray(rgb_array, mode="RGB")

    @staticmethod
    def validate_image_file(filepath: str) -> tuple[bool, str]:
        """Validate an image file without fully loading it.

        Args:
            filepath: Path to image file

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if file is valid, False otherwise
            - error_message: Empty string if valid, error description if invalid
        """
        filepath = Path(filepath)

        # Check if file exists
        if not filepath.exists():
            return False, f"File not found: {filepath}"

        # Check if it's a file
        if not filepath.is_file():
            return False, f"Path is not a file: {filepath}"

        # Check file extension
        if filepath.suffix.lower() not in ImageProcessor.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {filepath.suffix}"

        # Check file size
        file_size = filepath.stat().st_size
        if file_size == 0:
            return False, "File is empty"

        # Try to open image to verify it's valid
        try:
            with Image.open(filepath) as img:
                # Check dimensions
                width, height = img.size
                if (
                    width < ImageProcessor.MIN_DIMENSION
                    or height < ImageProcessor.MIN_DIMENSION
                ):
                    return False, f"Image too small: {width}x{height}"
                if (
                    width > ImageProcessor.MAX_DIMENSION
                    or height > ImageProcessor.MAX_DIMENSION
                ):
                    return False, f"Image too large: {width}x{height}"

            return True, ""

        except Exception as e:
            return False, f"Invalid image file: {str(e)}"

    @staticmethod
    def get_image_info(filepath: str) -> dict:
        """Get information about an image file.

        Args:
            filepath: Path to image file

        Returns:
            Dictionary with image information:
            - width: Image width in pixels
            - height: Image height in pixels
            - format: Image format (JPEG, PNG, etc.)
            - mode: Color mode (RGB, RGBA, etc.)
            - size: File size in bytes
        """
        filepath = Path(filepath)

        try:
            with Image.open(filepath) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "size": filepath.stat().st_size,
                }
        except Exception:
            return {}
