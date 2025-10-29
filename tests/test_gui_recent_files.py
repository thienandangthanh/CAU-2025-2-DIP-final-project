"""Tests for Recent Files functionality.

This module tests the Recent Files submenu implementation in the main window,
including adding files, clearing files, and menu updates.
"""

import pytest
from pathlib import Path
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from gui.main_window import MainWindow
from gui.utils import AppSettings


@pytest.fixture
def main_window(qtbot, tmp_path):
    """Provide a MainWindow instance for testing.

    Args:
        qtbot: pytest-qt fixture for testing Qt applications
        tmp_path: pytest fixture for temporary directory

    Yields:
        MainWindow instance with cleared settings
    """
    # Clear settings before test
    settings = AppSettings()
    settings.clear_recent_files()
    settings.sync()

    window = MainWindow()
    qtbot.addWidget(window)

    yield window

    # Cleanup
    settings.clear_recent_files()
    settings.sync()


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample image file for testing.

    Args:
        tmp_path: pytest fixture for temporary directory

    Returns:
        Path to sample image file
    """
    from PIL import Image

    # Create a simple test image
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test_image.png"
    img.save(img_path)

    return str(img_path)


class TestRecentFilesMenu:
    """Tests for Recent Files menu functionality."""

    def test_recent_files_menu_exists(self, main_window):
        """Test that Recent Files submenu exists in File menu."""
        assert hasattr(main_window, "recent_files_menu")
        assert isinstance(main_window.recent_files_menu, QMenu)
        assert main_window.recent_files_menu.title() == "Recent &Files"

    def test_recent_files_menu_empty_state(self, main_window):
        """Test that empty recent files menu shows '(Empty)' item."""
        # Menu should have one action: "(Empty)"
        actions = main_window.recent_files_menu.actions()
        assert len(actions) == 1
        assert actions[0].text() == "(Empty)"
        assert not actions[0].isEnabled()

    def test_update_recent_files_menu_called_on_load(
        self, main_window, sample_image, qtbot
    ):
        """Test that recent files menu is updated when an image is loaded."""
        # Load an image
        main_window._load_image(sample_image)
        qtbot.wait(100)  # Wait for UI updates

        # Check that menu is updated
        actions = main_window.recent_files_menu.actions()

        # Should have: 1 file action + separator + clear action = 3 total
        assert len(actions) == 3

        # First action should be the recent file
        assert actions[0].text() == Path(sample_image).name
        assert actions[0].isEnabled()

        # Second should be separator
        assert actions[1].isSeparator()

        # Third should be Clear Recent Files
        assert actions[2].text() == "&Clear Recent Files"
        assert actions[2].isEnabled()

    def test_recent_files_limit_to_10(self, main_window, tmp_path, qtbot):
        """Test that recent files list is limited to 10 items."""
        from PIL import Image

        # Create and load 15 images
        for i in range(15):
            img = Image.new("RGB", (100, 100), color="blue")
            img_path = tmp_path / f"test_image_{i}.png"
            img.save(img_path)
            main_window._load_image(str(img_path))

        qtbot.wait(100)

        # Get menu actions (excluding separator and clear action)
        actions = main_window.recent_files_menu.actions()
        file_actions = [
            a
            for a in actions
            if not a.isSeparator() and a.text() != "&Clear Recent Files"
        ]

        # Should have exactly 10 recent files
        assert len(file_actions) == 10

    def test_recent_file_click_loads_image(self, main_window, sample_image, qtbot):
        """Test that clicking a recent file loads that image."""
        # Load initial image
        main_window._load_image(sample_image)
        qtbot.wait(100)

        # Clear the current image
        main_window.current_input_image = None
        main_window.input_panel.clear()

        # Get first recent file action
        actions = main_window.recent_files_menu.actions()
        first_action = actions[0]

        # Trigger the action (should load the image)
        first_action.trigger()
        qtbot.wait(100)

        # Verify image was loaded
        assert main_window.current_input_image is not None
        assert main_window.input_panel.get_image_path() == sample_image

    def test_clear_recent_files_action(self, main_window, sample_image, qtbot):
        """Test that Clear Recent Files action clears the list."""
        # Load an image
        main_window._load_image(sample_image)
        qtbot.wait(100)

        # Verify file is in recent list
        actions = main_window.recent_files_menu.actions()
        assert len(actions) == 3  # file + separator + clear

        # Click Clear Recent Files
        clear_action = [a for a in actions if a.text() == "&Clear Recent Files"][0]
        clear_action.trigger()
        qtbot.wait(100)

        # Verify menu is now empty
        actions = main_window.recent_files_menu.actions()
        assert len(actions) == 1
        assert actions[0].text() == "(Empty)"

    def test_recent_file_not_found_disabled(self, main_window, tmp_path, qtbot):
        """Test that non-existent files are shown as disabled in menu."""
        # Create an image
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="green")
        img_path = tmp_path / "temp_image.png"
        img.save(img_path)

        # Load the image
        main_window._load_image(str(img_path))
        qtbot.wait(100)

        # Delete the image file
        img_path.unlink()

        # Update menu
        main_window._update_recent_files_menu()
        qtbot.wait(100)

        # Check that the action is disabled
        actions = main_window.recent_files_menu.actions()
        file_action = actions[0]

        assert not file_action.isEnabled()
        assert "(not found)" in file_action.text()

    def test_filename_truncation(self, main_window, tmp_path, qtbot):
        """Test that very long filenames are truncated in menu."""
        from PIL import Image

        # Create image with very long filename
        long_name = "a" * 60 + ".png"
        img = Image.new("RGB", (100, 100), color="yellow")
        img_path = tmp_path / long_name
        img.save(img_path)

        # Load the image
        main_window._load_image(str(img_path))
        qtbot.wait(100)

        # Check that filename is truncated
        actions = main_window.recent_files_menu.actions()
        file_action = actions[0]

        assert len(file_action.text()) <= 50
        assert file_action.text().endswith("...")

    def test_most_recent_first(self, main_window, tmp_path, qtbot):
        """Test that most recently opened file appears first in menu."""
        from PIL import Image

        # Create and load 3 images
        image_paths = []
        for i in range(3):
            img = Image.new("RGB", (100, 100), color="red")
            img_path = tmp_path / f"image_{i}.png"
            img.save(img_path)
            image_paths.append(str(img_path))
            main_window._load_image(str(img_path))
            qtbot.wait(50)

        # Get file actions
        actions = main_window.recent_files_menu.actions()
        file_actions = [
            a
            for a in actions
            if not a.isSeparator() and a.text() != "&Clear Recent Files"
        ]

        # Most recent should be first
        assert file_actions[0].text() == Path(image_paths[-1]).name
        assert file_actions[1].text() == Path(image_paths[-2]).name
        assert file_actions[2].text() == Path(image_paths[-3]).name

    def test_duplicate_file_moves_to_top(
        self, main_window, sample_image, tmp_path, qtbot
    ):
        """Test that reopening a file moves it to the top of recent list."""
        from PIL import Image

        # Load first image
        main_window._load_image(sample_image)
        qtbot.wait(50)

        # Load a second image
        img2 = Image.new("RGB", (100, 100), color="blue")
        img_path2 = tmp_path / "image_2.png"
        img2.save(img_path2)
        main_window._load_image(str(img_path2))
        qtbot.wait(50)

        # Reload the first image
        main_window._load_image(sample_image)
        qtbot.wait(50)

        # Check that first image is now at the top
        actions = main_window.recent_files_menu.actions()
        file_actions = [
            a
            for a in actions
            if not a.isSeparator() and a.text() != "&Clear Recent Files"
        ]

        assert file_actions[0].text() == Path(sample_image).name


class TestRecentFilesSettings:
    """Tests for Recent Files settings integration."""

    def test_recent_files_persist_across_instances(self, sample_image, qtbot, tmp_path):
        """Test that recent files persist when app is restarted."""
        # First instance - load image
        window1 = MainWindow()
        qtbot.addWidget(window1)
        window1._load_image(sample_image)
        qtbot.wait(100)
        window1.close()

        # Second instance - check recent files
        window2 = MainWindow()
        qtbot.addWidget(window2)
        qtbot.wait(100)

        actions = window2.recent_files_menu.actions()
        file_actions = [
            a
            for a in actions
            if not a.isSeparator() and a.text() != "&Clear Recent Files"
        ]

        assert len(file_actions) == 1
        assert file_actions[0].text() == Path(sample_image).name

        window2.close()

        # Cleanup
        settings = AppSettings()
        settings.clear_recent_files()
        settings.sync()

    def test_settings_sync_on_load(self, main_window, sample_image, qtbot):
        """Test that settings are synced when image is loaded."""
        # Load image
        main_window._load_image(sample_image)
        qtbot.wait(100)

        # Verify settings were updated
        settings = AppSettings()
        recent = settings.get_recent_files()

        assert len(recent) == 1
        assert recent[0] == sample_image


@pytest.mark.gui
class TestRecentFilesUI:
    """UI-focused tests for Recent Files menu."""

    def test_status_tip_shows_full_path(self, main_window, sample_image, qtbot):
        """Test that hovering over recent file shows full path in status tip."""
        # Load image
        main_window._load_image(sample_image)
        qtbot.wait(100)

        # Get first action
        actions = main_window.recent_files_menu.actions()
        file_action = actions[0]

        # Check status tip shows full path
        assert file_action.statusTip() == sample_image

    def test_clear_action_shows_status_message(self, main_window, sample_image, qtbot):
        """Test that clearing recent files shows status message."""
        # Load image
        main_window._load_image(sample_image)
        qtbot.wait(100)

        # Clear recent files
        main_window._clear_recent_files()
        qtbot.wait(100)

        # Check status bar message (it should have been shown)
        # Note: We can't easily test the actual message display in automated tests,
        # but we can verify the method executed without errors
        assert main_window.settings.get_recent_files() == []
