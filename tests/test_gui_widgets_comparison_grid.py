"""Tests for ComparisonGrid widget.

This module tests the comparison grid widget functionality including:
- Grid layout management
- Method cell creation
- Input/reference cell handling
- Result updates
"""

from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QSizePolicy

from gui.widgets.comparison_grid import ComparisonGrid


@pytest.fixture
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_pixmap():
    """Create a test pixmap."""
    pixmap = QPixmap(100, 100)
    pixmap.fill(Qt.GlobalColor.blue)
    return pixmap


class TestComparisonGrid:
    """Tests for ComparisonGrid widget."""

    def test_init(self, qapp):
        """Test grid initialization."""
        grid = ComparisonGrid(columns=3)

        assert grid.columns == 3
        assert len(grid.cells) == 0
        assert grid.input_cell is None
        assert grid.reference_cell is None
        assert grid.has_reference is False

    def test_set_methods_without_reference(self, qapp):
        """Test setting methods without reference image."""
        grid = ComparisonGrid()

        method_keys = ["method1", "method2", "method3"]
        method_names = {
            "method1": "Method 1",
            "method2": "Method 2",
            "method3": "Method 3",
        }

        grid.set_methods(
            method_keys, method_names, show_input=True, show_reference=False
        )

        # Should have input cell + 3 method cells
        assert len(grid.cells) == 4
        assert "input" in grid.cells
        assert "method1" in grid.cells
        assert "method2" in grid.cells
        assert "method3" in grid.cells
        assert grid.input_cell is not None
        assert grid.reference_cell is None
        assert grid.has_reference is False

    def test_set_methods_with_reference(self, qapp):
        """Test setting methods with reference image."""
        grid = ComparisonGrid()

        method_keys = ["method1", "method2"]
        method_names = {"method1": "Method 1", "method2": "Method 2"}

        grid.set_methods(
            method_keys, method_names, show_input=True, show_reference=True
        )

        # Should have input cell + reference cell + 2 method cells
        assert len(grid.cells) == 4
        assert "input" in grid.cells
        assert "reference" in grid.cells
        assert "method1" in grid.cells
        assert "method2" in grid.cells
        assert grid.input_cell is not None
        assert grid.reference_cell is not None
        assert grid.has_reference is True

    def test_set_methods_without_input(self, qapp):
        """Test setting methods without input cell."""
        grid = ComparisonGrid()

        method_keys = ["method1"]
        method_names = {"method1": "Method 1"}

        grid.set_methods(
            method_keys, method_names, show_input=False, show_reference=False
        )

        # Should only have 1 method cell
        assert len(grid.cells) == 1
        assert "input" not in grid.cells
        assert "method1" in grid.cells
        assert grid.input_cell is None

    def test_set_input_image(self, qapp, test_pixmap):
        """Test setting input image."""
        grid = ComparisonGrid()
        grid.set_methods(["method1"], {"method1": "Method 1"}, show_input=True)

        grid.set_input_image(test_pixmap)

        # Verify input cell has image
        assert grid.input_cell.current_pixmap is not None
        assert grid.input_cell.status == "done"

    def test_set_reference_image(self, qapp, test_pixmap):
        """Test setting reference image."""
        grid = ComparisonGrid()
        grid.set_methods(
            ["method1"], {"method1": "Method 1"}, show_input=True, show_reference=True
        )

        grid.set_reference_image(test_pixmap)

        # Verify reference cell has image
        assert grid.reference_cell.current_pixmap is not None
        assert grid.reference_cell.status == "done"

    def test_set_method_running(self, qapp):
        """Test marking method as running."""
        grid = ComparisonGrid()
        grid.set_methods(["method1"], {"method1": "Method 1"})

        grid.set_method_running("method1")

        assert grid.cells["method1"].status == "running"

    def test_set_method_result(self, qapp, test_pixmap):
        """Test setting method result."""
        grid = ComparisonGrid()
        grid.set_methods(["method1"], {"method1": "Method 1"})

        grid.set_method_result("method1", test_pixmap, "2.34s")

        cell = grid.cells["method1"]
        assert cell.current_pixmap is not None
        assert cell.status == "done"
        assert cell.timing_text == "2.34s"

    def test_set_method_error(self, qapp):
        """Test setting method error."""
        grid = ComparisonGrid()
        grid.set_methods(["method1"], {"method1": "Method 1"})

        grid.set_method_error("method1", "Test error")

        assert grid.cells["method1"].status == "error"

    def test_clear_results(self, qapp, test_pixmap):
        """Test clearing results while keeping structure."""
        grid = ComparisonGrid()
        grid.set_methods(["method1", "method2"], {"method1": "M1", "method2": "M2"})

        # Set some results
        grid.set_method_result("method1", test_pixmap, "1.0s")
        grid.set_method_result("method2", test_pixmap, "2.0s")

        # Clear results
        grid.clear_results()

        # Grid structure should remain
        assert len(grid.cells) > 0

        # Method cells should be cleared
        for cell in grid.cells.values():
            if not cell.is_reference:
                assert cell.current_pixmap is None
                assert cell.status == "pending"

    def test_clear_all(self, qapp):
        """Test clearing entire grid."""
        grid = ComparisonGrid()
        grid.set_methods(["method1"], {"method1": "Method 1"})

        grid.clear_all()

        assert len(grid.cells) == 0
        assert grid.input_cell is None
        assert grid.reference_cell is None

    def test_get_cell(self, qapp):
        """Test getting cell by key."""
        grid = ComparisonGrid()
        grid.set_methods(["method1"], {"method1": "Method 1"})

        cell = grid.get_cell("method1")

        assert cell is not None
        assert cell.method_key == "method1"

    def test_get_all_cells(self, qapp):
        """Test getting all cells."""
        grid = ComparisonGrid()
        grid.set_methods(
            ["method1", "method2"], {"method1": "M1", "method2": "M2"}, show_input=True
        )

        all_cells = grid.get_all_cells()

        # Should have input + 2 method cells = 3 total
        assert len(all_cells) == 3
        assert "input" in all_cells
        assert "method1" in all_cells
        assert "method2" in all_cells

    def test_update_header(self, qapp):
        """Test updating header text."""
        grid = ComparisonGrid()

        grid.update_header(3, has_reference=False)
        assert "3 methods" in grid.header_label.text()

        grid.update_header(1, has_reference=False)
        assert "1 method" in grid.header_label.text()

        grid.update_header(2, has_reference=True)
        assert "2 methods" in grid.header_label.text()
        assert "with reference" in grid.header_label.text()

    def test_set_columns(self, qapp):
        """Test changing column count."""
        grid = ComparisonGrid(columns=2)

        assert grid.columns == 2

        grid.set_columns(4)
        assert grid.columns == 4

    def test_set_columns_limits(self, qapp):
        """Test column count limits."""
        grid = ComparisonGrid()

        # Test minimum
        grid.set_columns(0)
        assert grid.columns == 1

        # Test maximum
        grid.set_columns(10)
        assert grid.columns == 4

    def test_cell_clicked_signal(self, qapp):
        """Test that cell clicked signal is propagated."""
        grid = ComparisonGrid()
        grid.set_methods(["method1"], {"method1": "Method 1"})

        # Track signals
        signals_received = []
        grid.cell_clicked.connect(lambda key: signals_received.append(key))

        # Emit click from cell
        grid.cells["method1"].clicked.emit("method1")

        # Verify signal was received
        assert len(signals_received) == 1
        assert signals_received[0] == "method1"


@pytest.mark.gui
class TestComparisonGridIntegration:
    """Integration tests for ComparisonGrid widget."""

    def test_grid_layout_with_multiple_methods(self, qapp, test_pixmap):
        """Test grid layout with multiple methods."""
        grid = ComparisonGrid(columns=2)

        method_keys = ["m1", "m2", "m3", "m4"]
        method_names = {k: f"Method {k}" for k in method_keys}

        grid.set_methods(method_keys, method_names, show_input=True)

        # Should create input cell + 4 method cells = 5 total
        assert len(grid.cells) == 5

        # All cells should be in the grid layout
        assert grid.grid_layout.count() == 5

    def test_grid_rearranges_on_column_change(self, qapp):
        """Test that grid rearranges when column count changes."""
        grid = ComparisonGrid(columns=2)

        method_keys = ["m1", "m2", "m3"]
        method_names = {k: f"Method {k}" for k in method_keys}

        grid.set_methods(method_keys, method_names, show_input=True)

        initial_count = grid.grid_layout.count()

        # Change columns
        grid.set_columns(3)

        # Should still have same number of cells
        assert grid.grid_layout.count() == initial_count

    def test_comparison_workflow(self, qapp, test_pixmap):
        """Test complete comparison workflow."""
        grid = ComparisonGrid(columns=2)

        # Set up methods
        method_keys = ["method1", "method2"]
        method_names = {"method1": "Method 1", "method2": "Method 2"}
        grid.set_methods(method_keys, method_names, show_input=True)

        # Set input image
        grid.set_input_image(test_pixmap)

        # Run method 1
        grid.set_method_running("method1")
        assert grid.cells["method1"].status == "running"

        # Complete method 1
        grid.set_method_result("method1", test_pixmap, "1.5s")
        assert grid.cells["method1"].status == "done"
        assert grid.cells["method1"].timing_text == "1.5s"

        # Run method 2
        grid.set_method_running("method2")
        assert grid.cells["method2"].status == "running"

        # Method 2 fails
        grid.set_method_error("method2", "Error occurred")
        assert grid.cells["method2"].status == "error"

    def test_histogram_settings_propagate(self, qapp):
        """Histogram settings should propagate to all cells."""
        grid = ComparisonGrid()
        method_keys = ["method1", "method2"]
        method_names = {"method1": "Method 1", "method2": "Method 2"}
        grid.set_methods(
            method_keys, method_names, show_input=True, show_reference=True
        )

        grid.set_histogram_settings(True, "grayscale")
        for cell in grid.get_all_cells().values():
            assert cell.histogram_visible is True

        grid.set_histogram_settings(True, "rgb")
        for cell in grid.get_all_cells().values():
            assert cell.histogram_type == "rgb"

        grid.set_histogram_settings(False, "rgb")
        for cell in grid.get_all_cells().values():
            assert cell.histogram_visible is False

    def test_uniform_expanding_cells(self, qapp):
        """Test that cells expand uniformly to fill space."""
        grid = ComparisonGrid()

        method_keys = ["method1", "method2"]
        method_names = {"method1": "Method 1", "method2": "Method 2"}
        grid.set_methods(method_keys, method_names, show_input=True)

        # All cells should have expanding size policy
        for cell in grid.cells.values():
            assert cell.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
            assert cell.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
            # All cells should have minimum size for readability
            assert cell.minimumWidth() == 250
            assert cell.minimumHeight() == 300

    def test_optimal_columns_by_cell_count(self, qapp):
        """Test optimal column count based on number of cells."""
        grid = ComparisonGrid()

        # Test 2 cells: should be 2 columns (1x2)
        grid.set_methods(["m1"], {"m1": "M1"}, show_input=True)
        assert grid.columns == 2

        # Test 3 cells: should be 3 columns (1x3)
        grid.set_methods(["m1", "m2"], {"m1": "M1", "m2": "M2"}, show_input=True)
        assert grid.columns == 3

        # Test 4 cells: should be 2 columns (2x2)
        grid.set_methods(
            ["m1", "m2", "m3"], {"m1": "M1", "m2": "M2", "m3": "M3"}, show_input=True
        )
        assert grid.columns == 2

        # Test 5 cells: should be 3 columns (2x3)
        grid.set_methods(
            ["m1", "m2", "m3", "m4"],
            {"m1": "M1", "m2": "M2", "m3": "M3", "m4": "M4"},
            show_input=True,
        )
        assert grid.columns == 3

        # Test 6 cells: should be 3 columns (2x3)
        grid.set_methods(
            ["m1", "m2", "m3", "m4", "m5"],
            {"m1": "M1", "m2": "M2", "m3": "M3", "m4": "M4", "m5": "M5"},
            show_input=True,
        )
        assert grid.columns == 3

        # Test 7+ cells: should be 4 columns
        grid.set_methods(
            ["m1", "m2", "m3", "m4", "m5", "m6"],
            {"m1": "M1", "m2": "M2", "m3": "M3", "m4": "M4", "m5": "M5", "m6": "M6"},
            show_input=True,
        )
        assert grid.columns == 4

    def test_uniform_stretch_factors(self, qapp):
        """Test that all rows and columns have uniform stretch factors."""
        grid = ComparisonGrid(columns=3)

        method_keys = ["m1", "m2", "m3", "m4"]
        method_names = {k: f"Method {k}" for k in method_keys}
        grid.set_methods(method_keys, method_names, show_input=True)

        # Should have 5 cells total (input + 4 methods) arranged in 3 columns
        # This means 2 rows (3 cells in row 0, 2 cells in row 1)

        # Check that all columns have equal stretch factor
        for col in range(grid.columns):
            stretch = grid.grid_layout.columnStretch(col)
            assert stretch == 1, (
                f"Column {col} should have stretch factor 1, got {stretch}"
            )

        # Check that all rows have equal stretch factor
        # We should have 2 rows (5 cells / 3 columns = 2 rows)
        num_rows = 2
        for row in range(num_rows):
            stretch = grid.grid_layout.rowStretch(row)
            assert stretch == 1, (
                f"Row {row} should have stretch factor 1, got {stretch}"
            )
