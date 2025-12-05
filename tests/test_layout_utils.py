"""Tests for layout_utils module.

This module tests the grid layout calculation functions used by both
the CLI comparison tool (compare.py) and the GUI comparison widget.
"""

import pytest

from layout_utils import calculate_optimal_columns, calculate_optimal_grid_layout


class TestCalculateOptimalGridLayout:
    """Tests for calculate_optimal_grid_layout function."""

    def test_zero_items(self):
        """Test with zero items."""
        rows, cols = calculate_optimal_grid_layout(0)
        assert rows == 1
        assert cols == 1

    def test_one_item(self):
        """Test with one item - should be 1x1 grid."""
        rows, cols = calculate_optimal_grid_layout(1)
        assert rows == 1
        assert cols == 1

    def test_two_items(self):
        """Test with two items - should be 1x2 grid."""
        rows, cols = calculate_optimal_grid_layout(2)
        assert rows == 1
        assert cols == 2

    def test_three_items(self):
        """Test with three items - should be 1x3 grid."""
        rows, cols = calculate_optimal_grid_layout(3)
        assert rows == 1
        assert cols == 3

    def test_four_items(self):
        """Test with four items - should be 2x2 grid."""
        rows, cols = calculate_optimal_grid_layout(4)
        assert rows == 2
        assert cols == 2

    def test_five_items(self):
        """Test with five items - should be 2x3 grid."""
        rows, cols = calculate_optimal_grid_layout(5)
        assert rows == 2
        assert cols == 3

    def test_six_items(self):
        """Test with six items - should be 2x3 grid."""
        rows, cols = calculate_optimal_grid_layout(6)
        assert rows == 2
        assert cols == 3

    def test_seven_items(self):
        """Test with seven items - should use 4 columns."""
        rows, cols = calculate_optimal_grid_layout(7)
        assert cols == 4
        assert rows == 2  # 7 items / 4 cols = 2 rows

    def test_eight_items(self):
        """Test with eight items - should be 2x4 grid."""
        rows, cols = calculate_optimal_grid_layout(8)
        assert rows == 2
        assert cols == 4

    def test_nine_items(self):
        """Test with nine items - should use 4 columns."""
        rows, cols = calculate_optimal_grid_layout(9)
        assert cols == 4
        assert rows == 3  # 9 items / 4 cols = 3 rows

    def test_twelve_items(self):
        """Test with twelve items - should be 3x4 grid."""
        rows, cols = calculate_optimal_grid_layout(12)
        assert rows == 3
        assert cols == 4

    def test_large_number(self):
        """Test with large number of items."""
        rows, cols = calculate_optimal_grid_layout(20)
        assert cols == 4  # Max 4 columns
        assert rows == 5  # 20 items / 4 cols = 5 rows

    def test_real_world_scenario_original_plus_reference_plus_six_methods(self):
        """Test real-world scenario: Original + Reference + 6 enhancement methods.

        This represents the typical use case in compare.py:
        - Original image
        - Reference (ground truth) image
        - Zero-DCE
        - AutoContrast
        - Histogram Eq
        - CLAHE
        - Gamma Correction
        - MSRCR

        Total: 8 images
        """
        rows, cols = calculate_optimal_grid_layout(8)
        assert rows == 2
        assert cols == 4
        # Verify all items fit in the grid
        assert rows * cols >= 8


class TestCalculateOptimalColumns:
    """Tests for calculate_optimal_columns convenience function."""

    def test_returns_column_count(self):
        """Test that function returns only the column count."""
        cols = calculate_optimal_columns(8)
        assert isinstance(cols, int)
        assert cols == 4

    def test_consistency_with_grid_layout(self):
        """Test that column count matches calculate_optimal_grid_layout."""
        for num_items in range(1, 20):
            rows, cols_from_layout = calculate_optimal_grid_layout(num_items)
            cols_direct = calculate_optimal_columns(num_items)
            assert cols_direct == cols_from_layout, (
                f"Column count mismatch for {num_items} items: "
                f"direct={cols_direct}, from_layout={cols_from_layout}"
            )

    def test_various_item_counts(self):
        """Test column count for various item counts."""
        test_cases = [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 2),
            (5, 3),
            (6, 3),
            (7, 4),
            (8, 4),
            (12, 4),
            (20, 4),
        ]

        for num_items, expected_cols in test_cases:
            cols = calculate_optimal_columns(num_items)
            assert cols == expected_cols, (
                f"Expected {expected_cols} columns for {num_items} items, got {cols}"
            )


class TestLayoutStrategyProperties:
    """Tests for general properties of the layout strategy."""

    def test_max_columns_is_four(self):
        """Test that the maximum number of columns is 4."""
        # Test various item counts, none should exceed 4 columns
        for num_items in range(1, 100):
            rows, cols = calculate_optimal_grid_layout(num_items)
            assert cols <= 4, (
                f"Expected max 4 columns, got {cols} for {num_items} items"
            )

    def test_grid_can_fit_all_items(self):
        """Test that the grid can always fit all items."""
        for num_items in range(1, 50):
            rows, cols = calculate_optimal_grid_layout(num_items)
            total_slots = rows * cols
            assert total_slots >= num_items, (
                f"Grid {rows}x{cols} cannot fit {num_items} items "
                f"(only {total_slots} slots)"
            )

    def test_grid_is_reasonably_efficient(self):
        """Test that the grid doesn't waste too many slots.

        The grid should not have more than one full row of empty slots.
        """
        for num_items in range(2, 50):  # Skip 1 item (always perfect fit)
            rows, cols = calculate_optimal_grid_layout(num_items)
            total_slots = rows * cols
            wasted_slots = total_slots - num_items

            # Allow at most (cols - 1) wasted slots (less than one full row)
            assert wasted_slots < cols, (
                f"Grid {rows}x{cols} wastes {wasted_slots} slots for {num_items} items "
                f"(more than one row)"
            )

    def test_returns_positive_values(self):
        """Test that rows and columns are always positive."""
        for num_items in range(0, 30):
            rows, cols = calculate_optimal_grid_layout(num_items)
            assert rows > 0, f"Rows should be positive, got {rows}"
            assert cols > 0, f"Columns should be positive, got {cols}"

    def test_single_row_for_three_or_less(self):
        """Test that 1-3 items always use a single row."""
        for num_items in [1, 2, 3]:
            rows, cols = calculate_optimal_grid_layout(num_items)
            assert rows == 1, f"Expected 1 row for {num_items} items, got {rows}"
