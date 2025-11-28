"""Tests for ImageProcessor utility module.

Tests image processing functionality including:
- Loading images from files
- Preprocessing for model input
- Post-processing model output
- Saving images
- Image format conversions
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gui.utils.image_processor import ImageProcessor


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample RGB image for testing."""
    # Create 256x256 RGB image
    image_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(image_array, mode="RGB")

    # Save to temp file
    image_path = tmp_path / "test_image.png"
    image.save(image_path)

    return str(image_path)


@pytest.fixture
def small_image(tmp_path):
    """Create a small test image (50x50)."""
    image_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    image = Image.fromarray(image_array, mode="RGB")

    image_path = tmp_path / "small_image.png"
    image.save(image_path)

    return str(image_path)


class TestImageLoading:
    """Tests for image loading functionality."""

    def test_load_valid_image(self, sample_image):
        """Test loading a valid image file."""
        image = ImageProcessor.load_image(sample_image)
        assert image is not None
        assert image.mode == "RGB"
        assert image.size == (256, 256)

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ImageProcessor.load_image("/nonexistent/path.png")

    def test_load_unsupported_format(self, tmp_path):
        """Test that unsupported format raises ValueError."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("not an image")

        with pytest.raises(ValueError, match="Unsupported image format"):
            ImageProcessor.load_image(str(text_file))

    def test_load_image_with_max_dimension(self, tmp_path):
        """Test that images are resized when exceeding max dimension."""
        # Create large image (1000x1000)
        large_image = Image.new("RGB", (1000, 1000), color=(255, 0, 0))
        large_path = tmp_path / "large.png"
        large_image.save(large_path)

        # Load with max dimension of 500
        image = ImageProcessor.load_image(str(large_path), max_dimension=500)

        # Should be resized to 500x500
        assert max(image.size) <= 500

    def test_load_image_too_small(self, tmp_path):
        """Test that too-small images raise ValueError."""
        # Create tiny image (32x32, below MIN_DIMENSION of 64)
        tiny_image = Image.new("RGB", (32, 32), color=(255, 0, 0))
        tiny_path = tmp_path / "tiny.png"
        tiny_image.save(tiny_path)

        with pytest.raises(ValueError, match="Image too small"):
            ImageProcessor.load_image(str(tiny_path))


class TestImagePreprocessing:
    """Tests for image preprocessing."""

    def test_preprocess_image(self):
        """Test that preprocessing produces correct output shape and range."""
        # Create test image
        image = Image.new("RGB", (256, 256), color=(128, 128, 128))

        # Preprocess
        processed = ImageProcessor.preprocess_image(image)

        # Check shape (batch, height, width, channels)
        assert processed.shape == (1, 256, 256, 3)

        # Check value range [0, 1]
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0

        # Check data type
        assert processed.dtype == np.float32

    def test_preprocess_normalization(self):
        """Test that preprocessing correctly normalizes values."""
        # Create image with known pixel values
        image_array = np.full((100, 100, 3), 255, dtype=np.uint8)  # All white
        image = Image.fromarray(image_array, mode="RGB")

        processed = ImageProcessor.preprocess_image(image)

        # All values should be 1.0 (255 / 255)
        assert np.allclose(processed, 1.0)


class TestImagePostprocessing:
    """Tests for image post-processing."""

    def test_postprocess_image(self):
        """Test that post-processing produces valid PIL Image."""
        # Create mock model output (batch, H, W, C)
        output_array = np.random.rand(1, 256, 256, 3).astype(np.float32)

        # Post-process
        image = ImageProcessor.postprocess_image(output_array)

        # Check output
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (256, 256)

    def test_postprocess_clipping(self):
        """Test that values are clipped to [0, 1] range."""
        # Create output with values outside [0, 1]
        output_array = np.array([[[[1.5, -0.5, 0.5]]]], dtype=np.float32)

        image = ImageProcessor.postprocess_image(output_array)
        image_array = np.array(image)

        # Values should be clipped
        assert image_array.min() >= 0
        assert image_array.max() <= 255

    def test_postprocess_removes_batch_dimension(self):
        """Test that batch dimension is removed correctly."""
        # With batch dimension
        output_with_batch = np.random.rand(1, 100, 100, 3).astype(np.float32)
        image = ImageProcessor.postprocess_image(output_with_batch)
        assert image.size == (100, 100)

        # Without batch dimension (should still work)
        output_no_batch = np.random.rand(100, 100, 3).astype(np.float32)
        image = ImageProcessor.postprocess_image(output_no_batch)
        assert image.size == (100, 100)


class TestImageSaving:
    """Tests for image saving functionality."""

    def test_save_image_png(self, tmp_path):
        """Test saving image in PNG format."""
        image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        output_path = tmp_path / "output.png"

        ImageProcessor.save_image(image, str(output_path), format="PNG")

        assert output_path.exists()

        # Verify saved image
        loaded = Image.open(output_path)
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)

    def test_save_image_jpeg(self, tmp_path):
        """Test saving image in JPEG format."""
        image = Image.new("RGB", (100, 100), color=(0, 255, 0))
        output_path = tmp_path / "output.jpg"

        ImageProcessor.save_image(image, str(output_path), format="JPEG", quality=95)

        assert output_path.exists()

    def test_save_image_invalid_format(self, tmp_path):
        """Test that invalid format raises ValueError."""
        image = Image.new("RGB", (100, 100))
        output_path = tmp_path / "output.xyz"

        with pytest.raises(ValueError, match="Invalid format"):
            ImageProcessor.save_image(image, str(output_path), format="XYZ")


class TestFileValidation:
    """Tests for file validation."""

    def test_validate_valid_image(self, sample_image):
        """Test validation of valid image file."""
        is_valid, error_msg = ImageProcessor.validate_image_file(sample_image)
        assert is_valid
        assert error_msg == ""

    def test_validate_nonexistent_file(self):
        """Test validation of nonexistent file."""
        is_valid, error_msg = ImageProcessor.validate_image_file("/nonexistent.png")
        assert not is_valid
        assert "not found" in error_msg.lower()

    def test_validate_unsupported_format(self, tmp_path):
        """Test validation of unsupported format."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("not an image")

        is_valid, error_msg = ImageProcessor.validate_image_file(str(text_file))
        assert not is_valid
        assert "unsupported" in error_msg.lower()

    def test_validate_empty_file(self, tmp_path):
        """Test validation of empty file."""
        empty_file = tmp_path / "empty.png"
        empty_file.touch()

        is_valid, error_msg = ImageProcessor.validate_image_file(str(empty_file))
        assert not is_valid
        assert "empty" in error_msg.lower()


class TestImageInfo:
    """Tests for getting image information."""

    def test_get_image_info(self, sample_image):
        """Test getting image information."""
        info = ImageProcessor.get_image_info(sample_image)

        assert info["width"] == 256
        assert info["height"] == 256
        assert info["format"] == "PNG"
        assert info["mode"] == "RGB"
        assert info["size"] > 0

    def test_get_image_info_invalid_file(self):
        """Test getting info from invalid file returns empty dict."""
        info = ImageProcessor.get_image_info("/nonexistent.png")
        assert info == {}


@pytest.mark.unit
class TestImageProcessorConstants:
    """Tests for ImageProcessor constants."""

    def test_constants_are_defined(self):
        """Test that all constants are properly defined."""
        assert ImageProcessor.MIN_DIMENSION == 64
        assert ImageProcessor.MAX_DIMENSION == 8192
        assert len(ImageProcessor.SUPPORTED_FORMATS) > 0
        assert ".jpg" in ImageProcessor.SUPPORTED_FORMATS
        assert ".png" in ImageProcessor.SUPPORTED_FORMATS
