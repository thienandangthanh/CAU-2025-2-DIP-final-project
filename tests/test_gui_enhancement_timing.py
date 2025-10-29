"""Tests for enhancement timing functionality in MainWindow.

This module tests the timing infrastructure added in Task 2.9 (Phase 2)
to track elapsed time during image enhancement.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.utils import EnhancementResult


@pytest.fixture(scope="module")
def qapp():
    """Provide QApplication instance for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def main_window(qtbot, qapp):
    """Provide MainWindow instance for testing."""
    window = MainWindow()
    qtbot.addWidget(window)
    return window


@pytest.fixture
def sample_image():
    """Provide a sample RGB image for testing."""
    return Image.new('RGB', (256, 256), color='red')


class TestEnhancementTimingInfrastructure:
    """Tests for enhancement timing infrastructure (Step 2 of Task 2.9)."""
    
    def test_enhancement_results_dict_initialized(self, main_window):
        """Test that _enhancement_results dict is initialized."""
        assert hasattr(main_window, '_enhancement_results')
        assert isinstance(main_window._enhancement_results, dict)
        assert len(main_window._enhancement_results) == 0
    
    def test_enhancement_start_time_initialized(self, main_window):
        """Test that _enhancement_start_time is initialized to None."""
        assert hasattr(main_window, '_enhancement_start_time')
        assert main_window._enhancement_start_time is None
    
    def test_current_enhancement_method_initialized(self, main_window):
        """Test that _current_enhancement_method is initialized."""
        assert hasattr(main_window, '_current_enhancement_method')
        assert main_window._current_enhancement_method == "Zero-DCE"


class TestEnhancementTimingCapture:
    """Tests for timing capture during enhancement."""
    
    def test_start_time_captured_on_enhance(self, main_window, sample_image, qtbot):
        """Test that start time is captured when enhancement begins."""
        # Mock model as loaded
        main_window.model_loader.is_model_loaded = Mock(return_value=True)
        main_window.model_loader.get_model = Mock(return_value=Mock())
        
        # Set input image
        main_window.current_input_image = sample_image
        
        # Mock the worker thread to prevent actual enhancement
        with patch('gui.main_window.EnhancementWorker') as MockWorker:
            mock_worker = Mock()
            MockWorker.return_value = mock_worker
            
            # Record time before enhancement
            time_before = time.perf_counter()
            
            # Trigger enhancement
            main_window._enhance_image()
            
            # Record time after enhancement
            time_after = time.perf_counter()
            
            # Verify start time was captured
            assert main_window._enhancement_start_time is not None
            assert time_before <= main_window._enhancement_start_time <= time_after
    
    def test_start_time_not_captured_without_image(self, main_window):
        """Test that start time is not captured if no image loaded."""
        main_window.current_input_image = None
        main_window._enhancement_start_time = None
        
        main_window._enhance_image()
        
        # Start time should still be None
        assert main_window._enhancement_start_time is None
    
    def test_start_time_not_captured_without_model(self, main_window, sample_image):
        """Test that start time is not captured if no model loaded."""
        main_window.current_input_image = sample_image
        main_window.model_loader.is_model_loaded = Mock(return_value=False)
        main_window._enhancement_start_time = None
        
        main_window._enhance_image()
        
        # Start time should still be None
        assert main_window._enhancement_start_time is None


class TestEnhancementResultCreation:
    """Tests for EnhancementResult creation on completion."""
    
    def test_enhancement_result_created_on_finish(self, main_window, sample_image, qtbot):
        """Test that EnhancementResult is created when enhancement finishes."""
        # Set up timing
        main_window._enhancement_start_time = time.perf_counter()
        
        # Wait a bit to get measurable elapsed time
        time.sleep(0.01)
        
        # Mock the image processor
        with patch('gui.main_window.ImageProcessor.pil_to_pixmap') as mock_pil_to_pixmap:
            mock_pil_to_pixmap.return_value = Mock()
            
            # Trigger enhancement finished
            main_window._on_enhancement_finished(sample_image)
        
        # Verify result was created and stored
        assert "Zero-DCE" in main_window._enhancement_results
        result = main_window._enhancement_results["Zero-DCE"]
        
        assert isinstance(result, EnhancementResult)
        assert result.image == sample_image
        assert result.method_name == "Zero-DCE"
        assert result.elapsed_time > 0
        assert result.elapsed_time < 1.0  # Should be very quick
    
    def test_enhancement_result_has_accurate_timing(self, main_window, sample_image):
        """Test that EnhancementResult has accurate elapsed time."""
        # Record start time
        start = time.perf_counter()
        main_window._enhancement_start_time = start
        
        # Wait a specific amount
        sleep_duration = 0.05  # 50ms
        time.sleep(sleep_duration)
        
        # Mock the image processor
        with patch('gui.main_window.ImageProcessor.pil_to_pixmap'):
            main_window._on_enhancement_finished(sample_image)
        
        # Verify timing is accurate (within 10ms tolerance)
        result = main_window._enhancement_results["Zero-DCE"]
        assert abs(result.elapsed_time - sleep_duration) < 0.01
    
    def test_multiple_enhancements_update_result(self, main_window, sample_image):
        """Test that multiple enhancements update the same result key."""
        # First enhancement
        main_window._enhancement_start_time = time.perf_counter()
        time.sleep(0.01)
        
        with patch('gui.main_window.ImageProcessor.pil_to_pixmap'):
            main_window._on_enhancement_finished(sample_image)
        
        first_result = main_window._enhancement_results["Zero-DCE"]
        first_time = first_result.elapsed_time
        
        # Second enhancement
        main_window._enhancement_start_time = time.perf_counter()
        time.sleep(0.02)
        
        with patch('gui.main_window.ImageProcessor.pil_to_pixmap'):
            main_window._on_enhancement_finished(sample_image)
        
        # Result should be updated, not added
        assert len(main_window._enhancement_results) == 1
        second_result = main_window._enhancement_results["Zero-DCE"]
        second_time = second_result.elapsed_time
        
        # Second time should be different (longer)
        assert second_time > first_time


class TestStatusBarTimingDisplay:
    """Tests for timing display in status bar."""
    
    def test_status_bar_shows_timing(self, main_window, sample_image):
        """Test that status bar displays elapsed time."""
        main_window._enhancement_start_time = time.perf_counter()
        time.sleep(0.01)
        
        with patch('gui.main_window.ImageProcessor.pil_to_pixmap'):
            main_window._on_enhancement_finished(sample_image)
        
        # Check status bar message
        status_message = main_window.statusBar().currentMessage()
        
        assert "Enhanced successfully" in status_message
        assert "in" in status_message
        assert "s" in status_message  # Should contain time unit (seconds)
    
    def test_status_bar_timing_format_short(self, main_window, sample_image):
        """Test status bar timing format for short durations."""
        main_window._enhancement_start_time = time.perf_counter()
        time.sleep(0.05)  # 50ms
        
        with patch('gui.main_window.ImageProcessor.pil_to_pixmap'):
            main_window._on_enhancement_finished(sample_image)
        
        status_message = main_window.statusBar().currentMessage()
        
        # Should show format like "Enhanced successfully in 0.05s"
        assert "Enhanced successfully in" in status_message
        # Should have decimal format (e.g., "0.05s")
        import re
        assert re.search(r'\d+\.\d+s', status_message)


class TestFutureProofDesign:
    """Tests for future-proof design (Phase 3 preparation)."""
    
    def test_results_dict_supports_multiple_methods(self, main_window, sample_image):
        """Test that results dict can store multiple enhancement methods."""
        # Simulate multiple methods (as will be done in Phase 3)
        methods = ["Zero-DCE", "CLAHE", "Histogram Eq", "Gamma Correction"]
        
        for method_name in methods:
            main_window._current_enhancement_method = method_name
            main_window._enhancement_start_time = time.perf_counter()
            time.sleep(0.01)
            
            with patch('gui.main_window.ImageProcessor.pil_to_pixmap'):
                main_window._on_enhancement_finished(sample_image)
        
        # All methods should be stored
        assert len(main_window._enhancement_results) == 4
        
        for method_name in methods:
            assert method_name in main_window._enhancement_results
            result = main_window._enhancement_results[method_name]
            assert isinstance(result, EnhancementResult)
            assert result.method_name == method_name
    
    def test_results_dict_allows_iteration(self, main_window, sample_image):
        """Test that results dict can be iterated (for Phase 3 display)."""
        # Add multiple results
        methods = ["Zero-DCE", "CLAHE"]
        
        for method_name in methods:
            main_window._current_enhancement_method = method_name
            main_window._enhancement_start_time = time.perf_counter()
            
            with patch('gui.main_window.ImageProcessor.pil_to_pixmap'):
                main_window._on_enhancement_finished(sample_image)
        
        # Test iteration (as will be used in Phase 3)
        results_list = []
        for method_name, result in main_window._enhancement_results.items():
            results_list.append((method_name, result.format_time()))
        
        assert len(results_list) == 2
        assert all(isinstance(item[1], str) for item in results_list)
    
    def test_current_method_can_be_changed(self, main_window):
        """Test that current enhancement method can be changed (Phase 3 feature)."""
        # Test changing method name (as will be done in Phase 3)
        assert main_window._current_enhancement_method == "Zero-DCE"
        
        main_window._current_enhancement_method = "CLAHE"
        assert main_window._current_enhancement_method == "CLAHE"
        
        main_window._current_enhancement_method = "Histogram Eq"
        assert main_window._current_enhancement_method == "Histogram Eq"


@pytest.mark.integration
class TestEnhancementTimingIntegration:
    """Integration tests for complete enhancement timing workflow."""
    
    def test_complete_enhancement_timing_workflow(self, main_window, sample_image, qtbot):
        """Test complete workflow from enhance to result storage."""
        # Setup
        main_window.current_input_image = sample_image
        main_window.model_loader.is_model_loaded = Mock(return_value=True)
        main_window.model_loader.get_model = Mock(return_value=Mock())
        
        # Initially no results
        assert len(main_window._enhancement_results) == 0
        assert main_window._enhancement_start_time is None
        
        # Start enhancement
        with patch('gui.main_window.EnhancementWorker') as MockWorker:
            mock_worker = Mock()
            MockWorker.return_value = mock_worker
            
            main_window._enhance_image()
            
            # Verify start time captured
            assert main_window._enhancement_start_time is not None
            start_time = main_window._enhancement_start_time
        
        # Simulate completion
        time.sleep(0.01)
        
        with patch('gui.main_window.ImageProcessor.pil_to_pixmap'):
            main_window._on_enhancement_finished(sample_image)
        
        # Verify result stored
        assert len(main_window._enhancement_results) == 1
        assert "Zero-DCE" in main_window._enhancement_results
        
        result = main_window._enhancement_results["Zero-DCE"]
        assert result.elapsed_time > 0
        assert result.method_name == "Zero-DCE"
        assert result.image == sample_image
        
        # Verify status bar updated
        status_message = main_window.statusBar().currentMessage()
        assert "Enhanced successfully" in status_message
