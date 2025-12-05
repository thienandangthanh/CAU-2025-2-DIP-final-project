"""Tests for plot_utils module.

This module tests the plotting utilities that provide consistent
matplotlib styling across the project.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from plot_utils import (
    check_latex_available,
    configure_publication_style,
    get_publication_style,
)


class TestLatexDetection:
    """Tests for LaTeX availability detection."""

    def test_check_latex_available_returns_bool(self):
        """Test that check_latex_available returns a boolean value."""
        result = check_latex_available()
        assert isinstance(result, bool)

    def test_check_latex_available_is_consistent(self):
        """Test that check_latex_available gives consistent results."""
        result1 = check_latex_available()
        result2 = check_latex_available()
        assert result1 == result2


class TestPublicationStyle:
    """Tests for publication style configuration."""

    def test_get_publication_style_returns_dict(self):
        """Test that get_publication_style returns a dictionary."""
        style = get_publication_style()
        assert isinstance(style, dict)

    def test_get_publication_style_has_required_keys(self):
        """Test that style dictionary contains all required keys."""
        style = get_publication_style()

        required_keys = [
            "text.usetex",
            "font.family",
            "axes.labelsize",
            "font.size",
            "legend.fontsize",
            "xtick.labelsize",
            "ytick.labelsize",
        ]

        for key in required_keys:
            assert key in style, f"Missing required key: {key}"

    def test_get_publication_style_font_family_is_serif(self):
        """Test that font family is set to serif."""
        style = get_publication_style()
        assert style["font.family"] == "serif"

    def test_get_publication_style_font_sizes_are_reasonable(self):
        """Test that font sizes are in reasonable range (6-14pt)."""
        style = get_publication_style()

        font_size_keys = [
            "axes.labelsize",
            "font.size",
            "legend.fontsize",
            "xtick.labelsize",
            "ytick.labelsize",
        ]

        for key in font_size_keys:
            size = style[key]
            assert isinstance(size, (int, float))
            assert 6 <= size <= 14, f"{key} = {size} is outside reasonable range"

    def test_get_publication_style_usetex_depends_on_latex(self):
        """Test that text.usetex matches LaTeX availability."""
        latex_available = check_latex_available()
        style = get_publication_style()

        assert style["text.usetex"] == latex_available

    def test_get_publication_style_warns_when_latex_unavailable(self):
        """Test that warning is issued when LaTeX is not available."""
        latex_available = check_latex_available()

        if not latex_available:
            with pytest.warns(UserWarning, match="LaTeX not found"):
                get_publication_style()


class TestConfigurePublicationStyle:
    """Tests for configure_publication_style function."""

    def setup_method(self):
        """Save original matplotlib rcParams before each test."""
        self.original_params = mpl.rcParams.copy()

    def teardown_method(self):
        """Restore original matplotlib rcParams after each test."""
        mpl.rcParams.update(self.original_params)

    def test_configure_publication_style_modifies_rcparams(self):
        """Test that configure_publication_style modifies matplotlib rcParams."""
        # Apply publication style
        configure_publication_style()

        # Check that rcParams were modified
        # Since we set font.size to 10, it should be 10 now
        assert mpl.rcParams["font.size"] == 10

    def test_configure_publication_style_sets_font_family(self):
        """Test that font family is set to serif."""
        configure_publication_style()
        assert mpl.rcParams["font.family"] == ["serif"]

    def test_configure_publication_style_sets_all_sizes(self):
        """Test that all font sizes are properly set."""
        configure_publication_style()

        assert mpl.rcParams["axes.labelsize"] == 10
        assert mpl.rcParams["font.size"] == 10
        assert mpl.rcParams["legend.fontsize"] == 8
        assert mpl.rcParams["xtick.labelsize"] == 8
        assert mpl.rcParams["ytick.labelsize"] == 8

    def test_configure_publication_style_warns_if_no_latex(self):
        """Test that warning is issued if LaTeX is not available."""
        latex_available = check_latex_available()

        if not latex_available:
            with pytest.warns(UserWarning, match="LaTeX not found"):
                configure_publication_style()

    def test_configure_publication_style_applies_colorblind_palette(self):
        """Test that colorblind-friendly style is attempted."""
        # This test just ensures no exception is raised
        # We can't easily test if the style was actually applied
        configure_publication_style()

        # Should not raise any exception
        assert True


class TestIntegrationWithMatplotlib:
    """Integration tests with actual matplotlib plotting."""

    def setup_method(self):
        """Save original matplotlib rcParams before each test."""
        self.original_params = mpl.rcParams.copy()

    def teardown_method(self):
        """Restore original matplotlib rcParams and close plots."""
        mpl.rcParams.update(self.original_params)
        plt.close("all")

    def test_plotting_works_after_configuration(self):
        """Test that basic plotting works after applying publication style."""
        configure_publication_style()

        # Create a simple plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_title("Test Plot")

        # Should not raise any exception
        assert True

        plt.close(fig)

    def test_multiple_subplots_work_after_configuration(self):
        """Test that multi-panel plots work after applying publication style."""
        configure_publication_style()

        # Create multi-panel plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        for ax in axes.flat:
            ax.plot([1, 2, 3], [1, 4, 9])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        # Should not raise any exception
        assert True

        plt.close(fig)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_configure_publication_style_can_be_called_multiple_times(self):
        """Test that configure_publication_style can be called multiple times."""
        # Should not raise exception or cause issues
        configure_publication_style()
        configure_publication_style()
        configure_publication_style()

        assert mpl.rcParams["font.size"] == 10

    def test_get_publication_style_does_not_modify_rcparams(self):
        """Test that get_publication_style doesn't modify rcParams."""
        original_font_size = mpl.rcParams["font.size"]

        # Call get_publication_style (should not modify rcParams)
        get_publication_style()

        # rcParams should remain unchanged
        assert mpl.rcParams["font.size"] == original_font_size
