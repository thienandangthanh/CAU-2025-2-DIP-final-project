"""Plotting utilities for publication-quality figures.

This module provides consistent matplotlib styling for all plots in the project,
ensuring manuscript-ready visualizations with proper font configuration and
colorblind-friendly palettes.
"""

import subprocess
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt


def check_latex_available() -> bool:
    """Check if LaTeX is available on the system.

    Attempts to run 'latex --version' to detect LaTeX installation.
    This is required for text.usetex=True in matplotlib.

    Returns:
        True if LaTeX is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["latex", "--version"], capture_output=True, timeout=5, check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_publication_style() -> dict[str, any]:
    """Get matplotlib configuration for publication-quality plots.

    Configures fonts, sizes, and rendering options suitable for academic
    manuscripts. Automatically detects LaTeX availability and disables
    LaTeX rendering if not found.

    Returns:
        Dictionary of matplotlib rcParams settings
    """
    # Check LaTeX availability
    use_latex = check_latex_available()

    if not use_latex:
        warnings.warn(
            "LaTeX not found on system. Using standard matplotlib rendering.\n"
            "For publication-quality plots, install LaTeX:\n"
            "  - Linux: sudo apt-get install texlive texlive-latex-extra cm-super dvipng\n"
            "  - macOS: Install MacTeX (https://www.tug.org/mactex/)\n"
            "  - Windows: Install MiKTeX (https://miktex.org/)\n"
            "See README.md for more details.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "text.usetex": use_latex,  # Use LaTeX if available
        "font.family": "serif",  # Use serif fonts to match most papers
        "axes.labelsize": 10,  # Font size for axis labels
        "font.size": 10,  # General font size
        "legend.fontsize": 8,  # Smaller font for legends
        "xtick.labelsize": 8,  # Font size for tick labels
        "ytick.labelsize": 8,
    }


def configure_publication_style():
    """Apply publication-quality styling to matplotlib.

    This function should be called once at the beginning of a script
    to configure matplotlib for consistent, manuscript-ready plots.
    It applies:
    - Serif fonts and appropriate sizes
    - LaTeX rendering (if available)
    - Colorblind-friendly color palette

    Example:
        >>> from plot_utils import configure_publication_style
        >>> configure_publication_style()
        >>> # Now all plots will use publication styling
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> plt.savefig('figure.png')
    """
    # Apply font and rendering configuration
    style_config = get_publication_style()
    mpl.rcParams.update(style_config)

    # Use colorblind-friendly palette
    # Note: 'seaborn-v0_8-colorblind' for matplotlib 3.6+
    # Falls back to 'seaborn-colorblind' for older versions
    try:
        plt.style.use("seaborn-v0_8-colorblind")
    except OSError:
        # Fallback for older matplotlib versions
        try:
            plt.style.use("seaborn-colorblind")
        except OSError:
            warnings.warn(
                "Colorblind-friendly style not available. Using default style.",
                UserWarning,
                stacklevel=2,
            )
