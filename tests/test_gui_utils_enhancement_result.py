"""Tests for EnhancementResult class.

This module tests the EnhancementResult class which encapsulates image
enhancement results with timing and metadata.
"""

import pytest
from PIL import Image

from gui.utils.enhancement_result import EnhancementResult


@pytest.fixture
def sample_image():
    """Provide a sample RGB image for testing."""
    return Image.new("RGB", (256, 256), color="red")


class TestEnhancementResultInit:
    """Tests for EnhancementResult initialization."""

    def test_init_with_valid_data(self, sample_image):
        """Test initialization with valid data."""
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34)

        assert result.image == sample_image
        assert result.method_name == "Zero-DCE"
        assert result.elapsed_time == 2.34
        assert result.quality_metrics == {}

    def test_init_with_quality_metrics(self, sample_image):
        """Test initialization with quality metrics."""
        metrics = {"PSNR": 24.5, "SSIM": 0.85}
        result = EnhancementResult(sample_image, "CLAHE", 0.12, metrics)

        assert result.quality_metrics == metrics

    def test_init_with_zero_time(self, sample_image):
        """Test initialization with zero elapsed time."""
        result = EnhancementResult(sample_image, "Fast", 0.0)

        assert result.elapsed_time == 0.0

    def test_init_with_negative_time_raises_error(self, sample_image):
        """Test that negative elapsed time raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            EnhancementResult(sample_image, "Invalid", -1.0)

        assert "elapsed_time must be non-negative" in str(exc_info.value)


class TestEnhancementResultTimeFormatting:
    """Tests for time formatting methods."""

    def test_format_time_less_than_one_second(self, sample_image):
        """Test formatting for times less than 1 second."""
        result = EnhancementResult(sample_image, "Fast", 0.12)

        assert result.format_time() == "0.12s"

    def test_format_time_exactly_one_second(self, sample_image):
        """Test formatting for exactly 1 second."""
        result = EnhancementResult(sample_image, "Medium", 1.0)

        assert result.format_time() == "1.00s"

    def test_format_time_few_seconds(self, sample_image):
        """Test formatting for a few seconds."""
        result = EnhancementResult(sample_image, "Zero-DCE", 2.345)

        # Should round to 2 decimal places
        assert result.format_time() == "2.35s"

    def test_format_time_just_under_60_seconds(self, sample_image):
        """Test formatting for times just under 60 seconds."""
        result = EnhancementResult(sample_image, "Slow", 59.99)

        assert result.format_time() == "59.99s"

    def test_format_time_exactly_60_seconds(self, sample_image):
        """Test formatting for exactly 60 seconds."""
        result = EnhancementResult(sample_image, "Slow", 60.0)

        assert result.format_time() == "1m 0s"

    def test_format_time_one_minute_34_seconds(self, sample_image):
        """Test formatting for 1 minute 34 seconds."""
        result = EnhancementResult(sample_image, "VerySlow", 94.5)

        # Should truncate to integer seconds
        assert result.format_time() == "1m 34s"

    def test_format_time_multiple_minutes(self, sample_image):
        """Test formatting for multiple minutes."""
        result = EnhancementResult(sample_image, "ExtraSlow", 185.7)

        # 185.7 seconds = 3 minutes 5 seconds
        assert result.format_time() == "3m 5s"

    def test_format_time_zero_seconds(self, sample_image):
        """Test formatting for zero seconds."""
        result = EnhancementResult(sample_image, "Instant", 0.0)

        assert result.format_time() == "0.00s"


class TestEnhancementResultSummary:
    """Tests for summary method."""

    def test_summary_short_time(self, sample_image):
        """Test summary with short elapsed time."""
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34)

        assert result.summary() == "Zero-DCE: 2.34s"

    def test_summary_long_time(self, sample_image):
        """Test summary with long elapsed time."""
        result = EnhancementResult(sample_image, "CLAHE", 94.5)

        assert result.summary() == "CLAHE: 1m 34s"

    def test_summary_with_different_methods(self, sample_image):
        """Test summary with different method names."""
        methods = ["Zero-DCE", "CLAHE", "Histogram Eq", "Gamma Correction"]

        for method in methods:
            result = EnhancementResult(sample_image, method, 1.0)
            assert result.summary().startswith(method)


class TestEnhancementResultImageInfo:
    """Tests for image information methods."""

    def test_get_image_info_returns_correct_data(self):
        """Test that get_image_info returns correct dimensions and mode."""
        image = Image.new("RGB", (1920, 1080))
        result = EnhancementResult(image, "Zero-DCE", 2.34)

        info = result.get_image_info()

        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["mode"] == "RGB"

    def test_get_image_info_with_different_sizes(self):
        """Test get_image_info with various image sizes."""
        sizes = [(256, 256), (512, 512), (1024, 768), (3840, 2160)]

        for width, height in sizes:
            image = Image.new("RGB", (width, height))
            result = EnhancementResult(image, "Test", 1.0)
            info = result.get_image_info()

            assert info["width"] == width
            assert info["height"] == height


class TestEnhancementResultQualityMetrics:
    """Tests for quality metrics functionality (Phase 3 preparation)."""

    def test_add_quality_metric_single(self, sample_image):
        """Test adding a single quality metric."""
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34)
        result.add_quality_metric("PSNR", 24.56)

        assert "PSNR" in result.quality_metrics
        assert result.quality_metrics["PSNR"] == 24.56

    def test_add_quality_metric_multiple(self, sample_image):
        """Test adding multiple quality metrics."""
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34)
        result.add_quality_metric("PSNR", 24.56)
        result.add_quality_metric("SSIM", 0.85)
        result.add_quality_metric("MSE", 120.5)

        assert len(result.quality_metrics) == 3
        assert result.quality_metrics["PSNR"] == 24.56
        assert result.quality_metrics["SSIM"] == 0.85
        assert result.quality_metrics["MSE"] == 120.5

    def test_add_quality_metric_overwrites_existing(self, sample_image):
        """Test that adding a metric with same name overwrites."""
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34)
        result.add_quality_metric("PSNR", 20.0)
        result.add_quality_metric("PSNR", 25.0)

        assert result.quality_metrics["PSNR"] == 25.0

    def test_init_with_none_quality_metrics(self, sample_image):
        """Test that None quality_metrics becomes empty dict."""
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34, None)

        assert result.quality_metrics == {}
        assert isinstance(result.quality_metrics, dict)


class TestEnhancementResultStringRepresentation:
    """Tests for string representation methods."""

    def test_str_returns_summary(self, sample_image):
        """Test that __str__ returns summary."""
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34)

        assert str(result) == "Zero-DCE: 2.34s"

    def test_repr_contains_method_name(self, sample_image):
        """Test that __repr__ contains method name."""
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34)

        repr_str = repr(result)
        assert "Zero-DCE" in repr_str
        assert "EnhancementResult" in repr_str

    def test_repr_contains_time(self, sample_image):
        """Test that __repr__ contains elapsed time."""
        result = EnhancementResult(sample_image, "CLAHE", 1.23)

        repr_str = repr(result)
        assert "1.23s" in repr_str

    def test_repr_contains_image_dimensions(self, sample_image):
        """Test that __repr__ contains image dimensions."""
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34)

        repr_str = repr(result)
        assert "256x256" in repr_str

    def test_repr_with_quality_metrics(self, sample_image):
        """Test that __repr__ includes quality metrics if present."""
        metrics = {"PSNR": 24.5}
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34, metrics)

        repr_str = repr(result)
        assert "metrics=" in repr_str
        assert "PSNR" in repr_str


class TestEnhancementResultEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_time(self, sample_image):
        """Test with very small elapsed time (microseconds)."""
        result = EnhancementResult(sample_image, "Fast", 0.001234)

        # Should format with 2 decimal places
        assert result.format_time() == "0.00s"

    def test_very_large_time(self, sample_image):
        """Test with very large elapsed time (hours)."""
        result = EnhancementResult(sample_image, "Slow", 3661.5)

        # 3661.5 seconds = 61 minutes 1 second
        assert result.format_time() == "61m 1s"

    def test_method_name_with_special_characters(self, sample_image):
        """Test with method name containing special characters."""
        result = EnhancementResult(sample_image, "Zero-DCE (v2.0)", 2.34)

        assert result.method_name == "Zero-DCE (v2.0)"
        assert "Zero-DCE (v2.0)" in result.summary()

    def test_different_image_modes(self):
        """Test with different image modes (RGB, L, RGBA)."""
        modes = ["RGB", "L", "RGBA"]

        for mode in modes:
            image = Image.new(mode, (256, 256))
            result = EnhancementResult(image, "Test", 1.0)
            info = result.get_image_info()

            assert info["mode"] == mode


@pytest.mark.integration
class TestEnhancementResultIntegration:
    """Integration tests for realistic usage scenarios."""

    def test_typical_zero_dce_workflow(self, sample_image):
        """Test typical Zero-DCE enhancement workflow."""
        # Simulate enhancement
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34)

        # Check all attributes
        assert result.image is not None
        assert result.method_name == "Zero-DCE"
        assert result.elapsed_time == 2.34

        # Check formatting
        time_str = result.format_time()
        assert "2.34s" == time_str

        # Check info
        info = result.get_image_info()
        assert info["width"] == 256
        assert info["height"] == 256

    def test_comparison_scenario(self, sample_image):
        """Test Phase 3 comparison scenario with multiple methods."""
        # Simulate multiple enhancement methods
        results = {
            "Zero-DCE": EnhancementResult(sample_image, "Zero-DCE", 2.34),
            "CLAHE": EnhancementResult(sample_image, "CLAHE", 0.12),
            "Histogram Eq": EnhancementResult(sample_image, "Histogram Eq", 0.05),
            "Gamma Correction": EnhancementResult(
                sample_image, "Gamma Correction", 0.08
            ),
        }

        # Verify all results
        for method_name, result in results.items():
            assert result.method_name == method_name
            assert result.elapsed_time > 0
            assert result.format_time().endswith("s")

    def test_with_quality_metrics_workflow(self, sample_image):
        """Test workflow with quality metrics (Phase 3)."""
        result = EnhancementResult(sample_image, "Zero-DCE", 2.34)

        # Add quality metrics after enhancement
        result.add_quality_metric("PSNR", 24.56)
        result.add_quality_metric("SSIM", 0.85)

        # Verify metrics are stored
        assert len(result.quality_metrics) == 2
        assert result.quality_metrics["PSNR"] == 24.56
        assert result.quality_metrics["SSIM"] == 0.85

        # Verify repr includes metrics
        repr_str = repr(result)
        assert "PSNR" in repr_str
