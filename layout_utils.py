"""Shared layout utilities for grid-based image comparisons.

This module provides common functions for calculating optimal grid layouts
that are used by both the CLI comparison tool (compare.py) and the GUI
comparison widget.
"""


def calculate_optimal_grid_layout(total_items: int) -> tuple[int, int]:
    """Calculate optimal grid layout (rows, columns) based on item count.

    This function determines the best grid dimensions to display multiple
    images in a visually balanced way. The strategy balances readability
    (not too many columns) with efficient use of space.

    Args:
        total_items: Total number of items to display in the grid

    Returns:
        Tuple of (rows, columns) for the grid layout

    Layout strategy:
        - 1 item: 1x1 grid (1 row, 1 column)
        - 2 items: 1x2 grid (1 row, 2 columns)
        - 3 items: 1x3 grid (1 row, 3 columns)
        - 4 items: 2x2 grid (2 rows, 2 columns)
        - 5-6 items: 2x3 grid (2 rows, 3 columns)
        - 7+ items: multiple rows, 4 columns max

    Examples:
        >>> calculate_optimal_grid_layout(1)
        (1, 1)
        >>> calculate_optimal_grid_layout(4)
        (2, 2)
        >>> calculate_optimal_grid_layout(8)
        (2, 4)
        >>> calculate_optimal_grid_layout(9)
        (3, 3)
    """
    if total_items <= 0:
        return (1, 1)
    elif total_items == 1:
        return (1, 1)
    elif total_items == 2:
        return (1, 2)
    elif total_items == 3:
        return (1, 3)
    elif total_items == 4:
        return (2, 2)
    elif total_items <= 6:
        # 5-6 items: 2x3 grid
        return (2, 3)
    else:
        # 7+ items: Use 4 columns max, calculate rows needed
        columns = 4
        rows = (total_items + columns - 1) // columns  # Ceiling division
        return (rows, columns)


def calculate_optimal_columns(total_items: int) -> int:
    """Calculate optimal number of columns based on item count.

    This is a convenience function that returns only the column count,
    useful for widgets that manage rows automatically.

    Args:
        total_items: Total number of items to display

    Returns:
        Optimal number of columns (1-4)

    Examples:
        >>> calculate_optimal_columns(4)
        2
        >>> calculate_optimal_columns(8)
        4
    """
    rows, cols = calculate_optimal_grid_layout(total_items)
    return cols
