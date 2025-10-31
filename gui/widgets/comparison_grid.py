"""Comparison grid widget for displaying multiple enhancement results.

This widget provides a scrollable grid layout to display and compare multiple
enhancement results side-by-side with the original input image and optional
reference image.
"""

from typing import Dict, Optional, List
from PyQt6.QtWidgets import (
    QWidget,
    QScrollArea,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap

from .comparison_cell import ComparisonCell


class ComparisonGrid(QWidget):
    """Grid widget for displaying multiple enhancement method results.
    
    This widget displays:
    - Original input image cell (always shown)
    - Optional reference image cell (when provided)
    - Enhancement method result cells
    
    The grid automatically adjusts layout based on the number of methods
    and window size (responsive: 1-4 columns).
    
    Signals:
        cell_clicked: Emitted when a cell is clicked (method_key: str)
    """
    
    # Signals
    cell_clicked = pyqtSignal(str)  # method_key
    
    def __init__(
        self,
        columns: int = 3,
        parent: Optional[QWidget] = None
    ):
        """Initialize the comparison grid.
        
        Args:
            columns: Number of columns in the grid (default: 3)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.columns = columns
        self.cells: Dict[str, ComparisonCell] = {}  # method_key -> ComparisonCell
        self.input_cell: Optional[ComparisonCell] = None
        self.reference_cell: Optional[ComparisonCell] = None
        self.has_reference = False
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header label
        self.header_label = QLabel("Comparison Mode")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_label.setStyleSheet(
            """
            QLabel {
                font-size: 15px;
                font-weight: bold;
                color: #2196F3;
                padding: 10px;
                background-color: #E3F2FD;
                border-bottom: 2px solid #2196F3;
            }
            """
        )
        main_layout.addWidget(self.header_label)
        
        # Scroll area for grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background-color: #FAFAFA;
            }
            """
        )
        
        # Grid container widget
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        self.grid_layout.setSpacing(15)
        
        scroll_area.setWidget(self.grid_container)
        main_layout.addWidget(scroll_area)
    
    def set_methods(
        self,
        method_keys: List[str],
        method_names: Dict[str, str],
        show_input: bool = True,
        show_reference: bool = False
    ):
        """Set up the grid with specified methods.
        
        Args:
            method_keys: List of method keys to display
            method_names: Dictionary mapping method_key to display name
            show_input: Whether to show input image cell (default: True)
            show_reference: Whether to show reference image cell (default: False)
        """
        # Clear existing cells
        self._clear_grid()
        
        self.has_reference = show_reference
        
        # Create input cell if requested
        if show_input:
            self.input_cell = ComparisonCell(
                method_key="input",
                method_name="Original Input",
                is_reference=True
            )
            self.input_cell.clicked.connect(self.cell_clicked.emit)
            self.cells["input"] = self.input_cell
        
        # Create reference cell if requested
        if show_reference:
            self.reference_cell = ComparisonCell(
                method_key="reference",
                method_name="Reference (High-Light)",
                is_reference=True
            )
            self.reference_cell.clicked.connect(self.cell_clicked.emit)
            self.cells["reference"] = self.reference_cell
        
        # Create cells for each method
        for method_key in method_keys:
            method_name = method_names.get(method_key, method_key)
            cell = ComparisonCell(
                method_key=method_key,
                method_name=method_name,
                is_reference=False
            )
            cell.clicked.connect(self.cell_clicked.emit)
            self.cells[method_key] = cell
        
        # Calculate optimal column count based on total cells
        total_cells = len(self.cells)
        self.columns = self._calculate_optimal_columns(total_cells)
        
        # Arrange cells in grid
        self._arrange_grid()
    
    def _calculate_optimal_columns(self, total_cells: int) -> int:
        """Calculate optimal number of columns based on cell count.
        
        Args:
            total_cells: Total number of cells to display
            
        Returns:
            Optimal number of columns (1-4)
            
        Layout strategy:
        - 1 cell: 1 column
        - 2 cells: 2 columns (1x2 grid)
        - 3 cells: 3 columns (1x3 grid)
        - 4 cells: 2 columns (2x2 grid)
        - 5-6 cells: 3 columns (2x3 grid)
        - 7+ cells: 4 columns (multiple rows)
        """
        if total_cells <= 1:
            return 1
        elif total_cells == 2:
            return 2
        elif total_cells == 3:
            return 3
        elif total_cells == 4:
            return 2  # 2x2 grid
        elif total_cells <= 6:
            return 3  # 2x3 or 3x3 grid
        else:
            return 4  # 4+ columns for many cells
    
    def _clear_grid(self):
        """Clear all cells from the grid."""
        # Remove all widgets from grid layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Clear cell references
        self.cells.clear()
        self.input_cell = None
        self.reference_cell = None
        self.has_reference = False
    
    def _arrange_grid(self):
        """Arrange cells in the grid layout with uniform spacing."""
        # Clear all existing stretch factors first to avoid leftover settings
        # Reset all row stretches (assume max 10 rows)
        for i in range(10):
            self.grid_layout.setRowStretch(i, 0)
        
        # Reset all column stretches (assume max 4 columns)
        for i in range(4):
            self.grid_layout.setColumnStretch(i, 0)
        
        row = 0
        col = 0
        
        # Add input cell first (always at top-left)
        if self.input_cell:
            self.grid_layout.addWidget(self.input_cell, row, col)
            col += 1
            if col >= self.columns:
                col = 0
                row += 1
        
        # Add reference cell next (if present)
        if self.reference_cell:
            self.grid_layout.addWidget(self.reference_cell, row, col)
            col += 1
            if col >= self.columns:
                col = 0
                row += 1
        
        # Add enhancement method cells
        for method_key, cell in self.cells.items():
            # Skip input and reference cells (already added)
            if method_key in ["input", "reference"]:
                continue
            
            self.grid_layout.addWidget(cell, row, col)
            col += 1
            if col >= self.columns:
                col = 0
                row += 1
        
        # Set uniform stretch factors for all rows and columns
        # This ensures all cells get equal space regardless of position
        num_rows = row + 1 if col > 0 else row
        for i in range(num_rows):
            self.grid_layout.setRowStretch(i, 1)  # Equal stretch for all rows
        
        for i in range(self.columns):
            self.grid_layout.setColumnStretch(i, 1)  # Equal stretch for all columns
    
    def set_input_image(self, pixmap: QPixmap):
        """Set the input image for the input cell.
        
        Args:
            pixmap: QPixmap of the input image
        """
        if self.input_cell:
            self.input_cell.set_image(pixmap)
            self.input_cell.set_status("done")
    
    def set_reference_image(self, pixmap: QPixmap):
        """Set the reference image for the reference cell.
        
        Args:
            pixmap: QPixmap of the reference image
        """
        if self.reference_cell:
            self.reference_cell.set_image(pixmap)
            self.reference_cell.set_status("done")
    
    def set_method_running(self, method_key: str):
        """Mark a method as currently running.
        
        Args:
            method_key: Method key to update
        """
        if method_key in self.cells:
            self.cells[method_key].set_status("running")
    
    def set_method_result(
        self,
        method_key: str,
        pixmap: QPixmap,
        timing_text: Optional[str] = None
    ):
        """Set the result for a method.
        
        Args:
            method_key: Method key to update
            pixmap: Result image pixmap
            timing_text: Optional timing text (e.g., "2.34s")
        """
        if method_key in self.cells:
            self.cells[method_key].set_image(pixmap)
            self.cells[method_key].set_status("done", timing_text)
    
    def set_method_error(self, method_key: str, error_message: str):
        """Set error status for a method.
        
        Args:
            method_key: Method key to update
            error_message: Error message to display
        """
        if method_key in self.cells:
            self.cells[method_key].set_error(error_message)
    
    def clear_results(self):
        """Clear all results while keeping the grid structure."""
        for cell in self.cells.values():
            if not cell.is_reference:
                cell.clear()
    
    def clear_all(self):
        """Clear the entire grid including input and reference."""
        self._clear_grid()
    
    def get_cell(self, method_key: str) -> Optional[ComparisonCell]:
        """Get a cell by method key.
        
        Args:
            method_key: Method key to retrieve
            
        Returns:
            ComparisonCell if found, None otherwise
        """
        return self.cells.get(method_key)
    
    def get_all_cells(self) -> Dict[str, ComparisonCell]:
        """Get all cells in the grid.
        
        Returns:
            Dictionary mapping method_key to ComparisonCell
        """
        return self.cells.copy()
    
    def update_header(self, method_count: int, has_reference: bool = False):
        """Update the header label with current comparison info.
        
        Args:
            method_count: Number of enhancement methods being compared
            has_reference: Whether reference image is included
        """
        parts = []
        parts.append(f"{method_count} method{'s' if method_count != 1 else ''}")
        
        if has_reference:
            parts.append("with reference")
        
        header_text = "Comparison Mode: " + ", ".join(parts)
        self.header_label.setText(header_text)
    
    def set_columns(self, columns: int):
        """Update the number of columns and rearrange grid.
        
        Args:
            columns: New number of columns (1-4)
        """
        if columns < 1:
            columns = 1
        elif columns > 4:
            columns = 4
        
        self.columns = columns
        
        # Rearrange grid if we have cells
        if self.cells:
            # Temporarily remove all widgets
            cells_backup = self.cells.copy()
            self._clear_grid()
            
            # Restore cells
            self.cells = cells_backup
            if "input" in self.cells:
                self.input_cell = self.cells["input"]
            if "reference" in self.cells:
                self.reference_cell = self.cells["reference"]
            
            # Rearrange with new column count
            self._arrange_grid()
    
