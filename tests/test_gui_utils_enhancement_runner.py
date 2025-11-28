"""Tests for gui/utils/enhancement_runner.py module.

This module tests the multi-method enhancement execution engine, including
synchronous and threaded execution, error handling, and progress reporting.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtTest import QSignalSpy

from gui.utils.enhancement_result import EnhancementResult
from gui.utils.enhancement_runner import (
    EnhancementRunner,
    EnhancementRunnerThread,
    enhance_with_methods,
)


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def runner():
    """Provide a fresh EnhancementRunner instance."""
    return EnhancementRunner()


class TestEnhancementRunner:
    """Tests for EnhancementRunner class."""

    def test_runner_initialization(self, runner):
        """Test that runner initializes correctly."""
        assert runner.registry is not None
        assert runner.is_running() is False

    def test_run_single_method_success(self, runner, sample_image):
        """Test running a single enhancement method successfully."""
        result = runner.run_single_method(sample_image, "gamma", model=None)

        assert result is not None
        assert isinstance(result, EnhancementResult)
        assert result.method_name == "Gamma Correction"
        assert result.elapsed_time > 0
        assert result.image.size == sample_image.size

    def test_run_single_method_with_different_methods(self, runner, sample_image):
        """Test running different single methods."""
        methods = ["autocontrast", "histogram-eq", "clahe", "gamma"]

        for method_key in methods:
            result = runner.run_single_method(sample_image, method_key, model=None)

            assert result is not None
            assert isinstance(result, EnhancementResult)
            assert result.elapsed_time > 0

    def test_run_single_method_invalid_key(self, runner, sample_image):
        """Test that invalid method key raises KeyError."""
        with pytest.raises(KeyError):
            runner.run_single_method(sample_image, "invalid-method", model=None)

    def test_run_single_method_zero_dce_without_model(self, runner, sample_image):
        """Test that Zero-DCE fails gracefully without model."""
        result = runner.run_single_method(sample_image, "zero-dce", model=None)

        # Should return None and emit failure signal
        assert result is None

    def test_run_multiple_methods_success(self, runner, sample_image):
        """Test running multiple enhancement methods successfully."""
        method_keys = ["autocontrast", "clahe", "gamma"]

        results = runner.run_multiple_methods(
            sample_image, method_keys, model=None, emit_signals=False
        )

        assert len(results) == 3
        assert "autocontrast" in results
        assert "clahe" in results
        assert "gamma" in results

        for result in results.values():
            assert isinstance(result, EnhancementResult)
            assert result.elapsed_time > 0

    def test_run_multiple_methods_empty_list(self, runner, sample_image):
        """Test running with empty method list."""
        results = runner.run_multiple_methods(
            sample_image, [], model=None, emit_signals=False
        )

        assert len(results) == 0
        assert isinstance(results, dict)

    def test_run_multiple_methods_single_method(self, runner, sample_image):
        """Test running with single method in list."""
        results = runner.run_multiple_methods(
            sample_image, ["clahe"], model=None, emit_signals=False
        )

        assert len(results) == 1
        assert "clahe" in results

    def test_run_multiple_methods_timing(self, runner, sample_image):
        """Test that timing is tracked correctly for each method."""
        method_keys = ["autocontrast", "clahe"]

        results = runner.run_multiple_methods(
            sample_image, method_keys, model=None, emit_signals=False
        )

        for result in results.values():
            # Timing should be positive and reasonable (< 1 second for classical methods)
            assert result.elapsed_time > 0
            assert result.elapsed_time < 1.0

    def test_run_multiple_methods_with_failure(self, runner, sample_image):
        """Test that failure in one method doesn't stop others."""
        # Include zero-dce (will fail without model) and classical methods
        method_keys = ["clahe", "zero-dce", "gamma"]

        results = runner.run_multiple_methods(
            sample_image, method_keys, model=None, emit_signals=False
        )

        # Should have 2 results (clahe and gamma), zero-dce failed
        assert len(results) == 2
        assert "clahe" in results
        assert "gamma" in results
        assert "zero-dce" not in results

    def test_run_multiple_methods_all_fail(self, runner, sample_image):
        """Test running methods that all fail."""
        # Only zero-dce without model
        method_keys = ["zero-dce"]

        results = runner.run_multiple_methods(
            sample_image, method_keys, model=None, emit_signals=False
        )

        assert len(results) == 0

    def test_run_multiple_methods_sequential_execution(self, runner, sample_image):
        """Test that methods execute sequentially (not parallel)."""
        method_keys = ["autocontrast", "clahe", "gamma"]

        start_time = time.perf_counter()
        results = runner.run_multiple_methods(
            sample_image, method_keys, model=None, emit_signals=False
        )
        total_time = time.perf_counter() - start_time

        # Total time should be approximately sum of individual times
        individual_times = sum(r.elapsed_time for r in results.values())

        # Allow some overhead, but should be close
        assert total_time >= individual_times
        assert total_time < individual_times + 0.5

    def test_cancel_functionality(self, runner, sample_image):
        """Test cancellation method exists and sets flag."""
        # Test that cancel() method works
        assert runner._should_cancel is False
        runner.cancel()
        assert runner._should_cancel is True

        # Verify is_running() method works
        assert isinstance(runner.is_running(), bool)
        assert runner.is_running() is False


class TestEnhancementRunnerSignals:
    """Tests for EnhancementRunner signal emissions."""

    def test_method_started_signal(self, runner, sample_image, qtbot):
        """Test that method_started signal is emitted."""
        spy = QSignalSpy(runner.method_started)

        runner.run_single_method(sample_image, "clahe", model=None)

        assert len(spy) == 1
        method_key, method_name = spy[0]
        assert method_key == "clahe"
        assert method_name == "CLAHE"

    def test_method_completed_signal(self, runner, sample_image, qtbot):
        """Test that method_completed signal is emitted."""
        spy = QSignalSpy(runner.method_completed)

        runner.run_single_method(sample_image, "gamma", model=None)

        assert len(spy) == 1
        method_key, result = spy[0]
        assert method_key == "gamma"
        assert isinstance(result, EnhancementResult)

    def test_method_failed_signal(self, runner, sample_image, qtbot):
        """Test that method_failed signal is emitted on failure."""
        spy = QSignalSpy(runner.method_failed)

        # Zero-DCE without model should fail
        runner.run_single_method(sample_image, "zero-dce", model=None)

        assert len(spy) == 1
        method_key, error_msg = spy[0]
        assert method_key == "zero-dce"
        assert "model" in error_msg.lower()

    def test_all_completed_signal(self, runner, sample_image, qtbot):
        """Test that all_completed signal is emitted after all methods."""
        spy = QSignalSpy(runner.all_completed)

        method_keys = ["clahe", "gamma"]
        runner.run_multiple_methods(
            sample_image, method_keys, model=None, emit_signals=True
        )

        assert len(spy) == 1
        results = spy[0][0]
        assert len(results) == 2

    def test_progress_updated_signal(self, runner, sample_image, qtbot):
        """Test that progress_updated signal is emitted."""
        spy = QSignalSpy(runner.progress_updated)

        method_keys = ["clahe", "gamma"]
        runner.run_multiple_methods(
            sample_image, method_keys, model=None, emit_signals=True
        )

        # Should have progress updates (at least initial and final)
        assert len(spy) >= 2


class TestEnhancementRunnerThread:
    """Tests for EnhancementRunnerThread class."""

    def test_thread_initialization(self, sample_image):
        """Test that thread initializes correctly."""
        method_keys = ["clahe", "gamma"]
        thread = EnhancementRunnerThread(sample_image, method_keys, model=None)

        assert thread.image == sample_image
        assert thread.method_keys == method_keys
        assert thread.model is None
        assert thread.runner is not None

    def test_thread_execution(self, sample_image, qtbot):
        """Test that thread executes methods in background."""
        method_keys = ["clahe", "gamma"]
        thread = EnhancementRunnerThread(sample_image, method_keys, model=None)

        # Track completion with a flag
        results_received = []

        def on_completed(results):
            results_received.append(results)

        thread.all_completed.connect(on_completed)

        # Start thread
        thread.start()

        # Wait for thread to complete (with timeout)
        assert thread.wait(5000)  # 5 second timeout

        # Process pending events to ensure signal is delivered
        qtbot.wait(100)

        # Check that completion callback was called
        assert len(results_received) == 1
        assert len(results_received[0]) == 2

    def test_thread_signals_emitted(self, sample_image, qtbot):
        """Test that thread emits all expected signals."""
        method_keys = ["clahe"]
        thread = EnhancementRunnerThread(sample_image, method_keys, model=None)

        # Track signals with counters
        signal_counts = {"started": 0, "completed": 0, "all": 0}

        thread.method_started.connect(
            lambda *args: signal_counts.update(
                {"started": signal_counts["started"] + 1}
            )
        )
        thread.method_completed.connect(
            lambda *args: signal_counts.update(
                {"completed": signal_counts["completed"] + 1}
            )
        )
        thread.all_completed.connect(
            lambda *args: signal_counts.update({"all": signal_counts["all"] + 1})
        )

        thread.start()
        thread.wait(5000)
        qtbot.wait(100)

        assert signal_counts["started"] == 1
        assert signal_counts["completed"] == 1
        assert signal_counts["all"] == 1

    def test_thread_with_failure(self, sample_image, qtbot):
        """Test that thread handles failures correctly."""
        method_keys = ["clahe", "zero-dce", "gamma"]  # zero-dce will fail
        thread = EnhancementRunnerThread(sample_image, method_keys, model=None)

        # Track failures and completion
        failures = []
        results_received = []

        thread.method_failed.connect(lambda key, msg: failures.append((key, msg)))
        thread.all_completed.connect(lambda results: results_received.append(results))

        thread.start()
        thread.wait(5000)
        qtbot.wait(100)

        # Should have one failure
        assert len(failures) == 1
        assert failures[0][0] == "zero-dce"

        # Should still complete with successful methods
        assert len(results_received) == 1
        results = results_received[0]
        assert len(results) == 2  # clahe and gamma succeeded


class TestConvenienceFunction:
    """Tests for enhance_with_methods convenience function."""

    def test_enhance_with_methods_basic(self, sample_image):
        """Test basic usage of enhance_with_methods function."""
        method_keys = ["clahe", "gamma"]

        results = enhance_with_methods(sample_image, method_keys, model=None)

        assert len(results) == 2
        assert "clahe" in results
        assert "gamma" in results

        for result in results.values():
            assert isinstance(result, EnhancementResult)

    def test_enhance_with_methods_empty_list(self, sample_image):
        """Test enhance_with_methods with empty list."""
        results = enhance_with_methods(sample_image, [], model=None)

        assert len(results) == 0
        assert isinstance(results, dict)

    def test_enhance_with_methods_with_failure(self, sample_image):
        """Test enhance_with_methods with failing method."""
        method_keys = ["clahe", "zero-dce", "gamma"]

        results = enhance_with_methods(sample_image, method_keys, model=None)

        # Should have 2 results (zero-dce failed)
        assert len(results) == 2
        assert "zero-dce" not in results


@pytest.mark.integration
class TestEnhancementRunnerIntegration:
    """Integration tests with real enhancement methods."""

    def test_all_classical_methods_in_sequence(self, sample_image):
        """Test running all classical methods in sequence."""
        method_keys = ["autocontrast", "histogram-eq", "clahe", "gamma"]

        results = enhance_with_methods(sample_image, method_keys, model=None)

        assert len(results) == 4

        for result in results.values():
            assert result.image.size == sample_image.size
            assert result.image.mode == "RGB"
            assert result.elapsed_time > 0

    def test_enhanced_images_are_different(self, sample_image):
        """Test that different methods produce different results."""
        method_keys = ["clahe", "gamma"]

        results = enhance_with_methods(sample_image, method_keys, model=None)

        clahe_array = np.array(results["clahe"].image)
        gamma_array = np.array(results["gamma"].image)

        # Different methods should produce different results
        assert not np.array_equal(clahe_array, gamma_array)

    def test_timing_is_reasonable(self, sample_image):
        """Test that timing measurements are reasonable."""
        method_keys = ["clahe", "gamma"]

        start = time.perf_counter()
        results = enhance_with_methods(sample_image, method_keys, model=None)
        total_wall_time = time.perf_counter() - start

        # Sum of individual times should be close to total wall time
        sum_individual = sum(r.elapsed_time for r in results.values())

        # Allow some overhead, but should be close
        assert sum_individual <= total_wall_time
        assert sum_individual >= total_wall_time * 0.8  # At least 80% of wall time

    def test_runner_state_tracking(self, sample_image):
        """Test that runner tracks its running state correctly."""
        runner = EnhancementRunner()

        assert runner.is_running() is False

        # Can't easily test is_running during execution without threading
        # Just verify the method exists and returns boolean
        assert isinstance(runner.is_running(), bool)
