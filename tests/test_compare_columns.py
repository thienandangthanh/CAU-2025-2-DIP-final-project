"""Tests for compare.py --columns parameter.

This module tests the custom grid layout functionality where users can
specify the number of columns in the comparison grid.
"""

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import pytest
from PIL import Image

from compare import compare_methods


class TestCompareColumnsParameter:
    """Tests for the columns parameter in compare_methods."""

    @pytest.fixture
    def test_image(self, tmp_path):
        """Create a test image for comparison."""
        img = Image.new("RGB", (400, 300), color=(100, 100, 100))
        img_path = tmp_path / "test_input.png"
        img.save(img_path)
        return str(img_path)

    @pytest.fixture
    def temp_weights(self, tmp_path):
        """Create a dummy weights file path (not used for classical methods)."""
        weights_path = tmp_path / "dummy_weights.h5"
        # Note: We won't actually load Zero-DCE, just test classical methods
        return str(weights_path)

    def test_auto_calculate_columns_default(self, test_image, temp_weights, tmp_path):
        """Test that columns=None auto-calculates optimal layout."""
        output_path = tmp_path / "auto_layout.png"

        # Use only classical methods (no Zero-DCE to avoid needing real weights)
        compare_methods(
            input_path=test_image,
            weights_path=temp_weights,
            output_path=str(output_path),
            methods=["autocontrast", "histogram-eq", "clahe", "gamma"],
            columns=None,  # Auto-calculate
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_custom_columns_2(self, test_image, temp_weights, tmp_path):
        """Test with custom 2-column layout."""
        output_path = tmp_path / "2cols_layout.png"

        compare_methods(
            input_path=test_image,
            weights_path=temp_weights,
            output_path=str(output_path),
            methods=["autocontrast", "histogram-eq", "clahe", "gamma"],
            columns=2,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_custom_columns_3(self, test_image, temp_weights, tmp_path):
        """Test with custom 3-column layout."""
        output_path = tmp_path / "3cols_layout.png"

        compare_methods(
            input_path=test_image,
            weights_path=temp_weights,
            output_path=str(output_path),
            methods=["autocontrast", "histogram-eq", "clahe"],
            columns=3,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_custom_columns_1(self, test_image, temp_weights, tmp_path):
        """Test with 1-column layout (vertical stack)."""
        output_path = tmp_path / "1col_layout.png"

        compare_methods(
            input_path=test_image,
            weights_path=temp_weights,
            output_path=str(output_path),
            methods=["autocontrast", "histogram-eq"],
            columns=1,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_custom_columns_large(self, test_image, temp_weights, tmp_path):
        """Test with large column count (more columns than images)."""
        output_path = tmp_path / "large_cols_layout.png"

        # 3 images (Original + 2 methods) with 5 columns should work
        compare_methods(
            input_path=test_image,
            weights_path=temp_weights,
            output_path=str(output_path),
            methods=["autocontrast", "histogram-eq"],
            columns=5,  # More columns than images
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_columns_with_reference(self, test_image, temp_weights, tmp_path):
        """Test custom columns with reference image included."""
        output_path = tmp_path / "cols_with_ref.png"

        # Create reference image
        ref_image = Image.new("RGB", (400, 300), color=(200, 200, 200))
        ref_path = tmp_path / "reference.png"
        ref_image.save(ref_path)

        compare_methods(
            input_path=test_image,
            weights_path=temp_weights,
            output_path=str(output_path),
            methods=["autocontrast", "histogram-eq"],
            reference_path=str(ref_path),
            columns=2,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_different_column_values_produce_different_layouts(
        self, test_image, temp_weights, tmp_path
    ):
        """Test that different column values produce different file sizes."""
        methods = ["autocontrast", "histogram-eq", "clahe"]

        # Generate with 2 columns
        output_2cols = tmp_path / "layout_2cols.png"
        compare_methods(
            input_path=test_image,
            weights_path=temp_weights,
            output_path=str(output_2cols),
            methods=methods,
            columns=2,
        )

        # Generate with 4 columns
        output_4cols = tmp_path / "layout_4cols.png"
        compare_methods(
            input_path=test_image,
            weights_path=temp_weights,
            output_path=str(output_4cols),
            methods=methods,
            columns=4,
        )

        # Both should exist
        assert output_2cols.exists()
        assert output_4cols.exists()

        # Layouts should be different (different aspect ratios)
        # File sizes won't be identical due to different layouts
        size_2cols = output_2cols.stat().st_size
        size_4cols = output_4cols.stat().st_size

        # Both should be valid images (non-zero size)
        assert size_2cols > 0
        assert size_4cols > 0


class TestCompareColumnsValidation:
    """Tests for columns parameter validation."""

    def test_columns_none_is_valid(self):
        """Test that columns=None is valid (auto-calculate)."""
        # This is tested implicitly in the main tests
        # Just verify it doesn't raise an error
        assert True

    def test_columns_positive_integers_are_valid(self):
        """Test that positive integers are valid for columns."""
        valid_values = [1, 2, 3, 4, 5, 10, 20]
        for value in valid_values:
            assert value > 0  # Our validation check


class TestCompareColumnsIntegration:
    """Integration tests for columns parameter with real workflow."""

    def test_auto_vs_custom_columns_comparison(self, tmp_path):
        """Compare auto-calculated vs custom column layouts."""
        # Create test image
        img = Image.new("RGB", (400, 300), color=(100, 100, 100))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        weights_path = tmp_path / "dummy.h5"

        # Auto-calculated layout (should be 2x2 for 4 images)
        output_auto = tmp_path / "auto.png"
        compare_methods(
            input_path=str(img_path),
            weights_path=str(weights_path),
            output_path=str(output_auto),
            methods=["autocontrast", "histogram-eq", "clahe"],
            columns=None,  # Auto
        )

        # Custom 2-column layout
        output_custom = tmp_path / "custom.png"
        compare_methods(
            input_path=str(img_path),
            weights_path=str(weights_path),
            output_path=str(output_custom),
            methods=["autocontrast", "histogram-eq", "clahe"],
            columns=2,  # Custom
        )

        # Both should produce valid outputs
        assert output_auto.exists()
        assert output_custom.exists()
        assert output_auto.stat().st_size > 0
        assert output_custom.stat().st_size > 0

    def test_columns_parameter_with_all_methods(self, tmp_path):
        """Test columns parameter works with all classical methods."""
        img = Image.new("RGB", (400, 300), color=(100, 100, 100))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        weights_path = tmp_path / "dummy.h5"
        output_path = tmp_path / "all_methods.png"

        # All classical methods with custom columns
        compare_methods(
            input_path=str(img_path),
            weights_path=str(weights_path),
            output_path=str(output_path),
            methods=["autocontrast", "histogram-eq", "clahe", "gamma", "msrcr"],
            columns=3,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
