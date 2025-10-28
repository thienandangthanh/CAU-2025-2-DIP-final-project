"""Tests for gui.utils.model_loader module.

Tests the ModelLoader class which handles loading and managing Zero-DCE model weights.
"""

import os
import tempfile
from pathlib import Path

import pytest

from gui.utils.model_loader import ModelLoader
from model import ZeroDCE


class TestModelLoaderValidation:
    """Tests for static validation methods."""

    def test_validate_weights_file_valid(self, tmp_path):
        """Test validation passes for valid weight file."""
        # Create a valid dummy weights file
        weights_file = tmp_path / "test.weights.h5"
        weights_file.write_bytes(b"dummy content")

        is_valid, error = ModelLoader.validate_weights_file(str(weights_file))

        assert is_valid is True
        assert error == ""

    def test_validate_weights_file_h5_extension(self, tmp_path):
        """Test validation passes for .h5 extension."""
        weights_file = tmp_path / "test.h5"
        weights_file.write_bytes(b"dummy content")

        is_valid, error = ModelLoader.validate_weights_file(str(weights_file))

        assert is_valid is True
        assert error == ""

    def test_validate_weights_file_not_found(self):
        """Test validation fails when file doesn't exist."""
        is_valid, error = ModelLoader.validate_weights_file("/nonexistent/path.h5")

        assert is_valid is False
        assert "not found" in error.lower()

    def test_validate_weights_file_invalid_extension(self, tmp_path):
        """Test validation fails for invalid file extension."""
        weights_file = tmp_path / "test.txt"
        weights_file.write_bytes(b"dummy content")

        is_valid, error = ModelLoader.validate_weights_file(str(weights_file))

        assert is_valid is False
        assert "invalid file format" in error.lower()

    def test_validate_weights_file_is_directory(self, tmp_path):
        """Test validation fails when path is a directory."""
        is_valid, error = ModelLoader.validate_weights_file(str(tmp_path))

        assert is_valid is False
        assert "not a file" in error.lower()

    def test_validate_weights_file_empty_file(self, tmp_path):
        """Test validation fails for empty file."""
        weights_file = tmp_path / "empty.h5"
        weights_file.touch()  # Create empty file

        is_valid, error = ModelLoader.validate_weights_file(str(weights_file))

        assert is_valid is False
        assert "empty" in error.lower()


class TestModelLoaderListWeights:
    """Tests for listing available weight files."""

    def test_list_available_weights_empty_directory(self, tmp_path):
        """Test listing weights in empty directory."""
        result = ModelLoader.list_available_weights(str(tmp_path))

        assert result == []

    def test_list_available_weights_nonexistent_directory(self):
        """Test listing weights in nonexistent directory."""
        result = ModelLoader.list_available_weights("/nonexistent/directory")

        assert result == []

    def test_list_available_weights_finds_h5_files(self, tmp_path):
        """Test listing finds .h5 files."""
        # Create test files
        (tmp_path / "model1.h5").write_bytes(b"dummy")
        (tmp_path / "model2.weights.h5").write_bytes(b"dummy")
        (tmp_path / "model3.h5").write_bytes(b"dummy")
        (tmp_path / "readme.txt").write_bytes(b"dummy")  # Should be ignored

        result = ModelLoader.list_available_weights(str(tmp_path))

        assert len(result) == 3
        # Check all returned files have valid extensions
        for path in result:
            assert path.endswith(".h5") or path.endswith(".weights.h5")

    def test_list_available_weights_sorted_by_time(self, tmp_path):
        """Test weights are sorted by modification time (newest first)."""
        import time

        # Create files with different modification times
        file1 = tmp_path / "old.h5"
        file1.write_bytes(b"dummy")
        time.sleep(0.1)

        file2 = tmp_path / "middle.h5"
        file2.write_bytes(b"dummy")
        time.sleep(0.1)

        file3 = tmp_path / "newest.h5"
        file3.write_bytes(b"dummy")

        result = ModelLoader.list_available_weights(str(tmp_path))

        # Should be sorted newest first
        assert len(result) == 3
        assert result[0].endswith("newest.h5")
        assert result[2].endswith("old.h5")


class TestModelLoaderInstantiation:
    """Tests for ModelLoader initialization and state."""

    def test_init_creates_empty_loader(self):
        """Test ModelLoader initializes with no model loaded."""
        loader = ModelLoader()

        assert loader.current_model is None
        assert loader.current_weights_path is None
        assert loader.is_model_loaded() is False

    def test_get_model_returns_none_initially(self):
        """Test get_model returns None when no model loaded."""
        loader = ModelLoader()

        assert loader.get_model() is None

    def test_get_weights_path_returns_none_initially(self):
        """Test get_weights_path returns None when no model loaded."""
        loader = ModelLoader()

        assert loader.get_weights_path() is None

    def test_get_model_info_empty_state(self):
        """Test get_model_info returns empty info when no model loaded."""
        loader = ModelLoader()
        info = loader.get_model_info()

        assert info["loaded"] is False
        assert info["weights_path"] is None
        assert info["file_size"] is None
        assert info["filename"] is None


class TestModelLoaderUnload:
    """Tests for model unloading functionality."""

    def test_unload_model_clears_state(self):
        """Test unload_model clears model and path."""
        loader = ModelLoader()
        # Manually set state to simulate loaded model
        loader.current_model = ZeroDCE()
        loader.current_weights_path = "/some/path.h5"

        loader.unload_model()

        assert loader.current_model is None
        assert loader.current_weights_path is None
        assert loader.is_model_loaded() is False

    def test_unload_model_safe_when_empty(self):
        """Test unload_model doesn't error when no model loaded."""
        loader = ModelLoader()

        # Should not raise any exception
        loader.unload_model()

        assert loader.current_model is None


class TestModelLoaderLoadModel:
    """Tests for model loading functionality."""

    def test_load_model_raises_on_nonexistent_file(self):
        """Test load_model raises FileNotFoundError for missing file."""
        loader = ModelLoader()

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_model("/nonexistent/path.h5")

        assert "not found" in str(exc_info.value).lower()

    def test_load_model_raises_on_invalid_extension(self, tmp_path):
        """Test load_model raises ValueError for invalid extension."""
        # Create a file with wrong extension
        invalid_file = tmp_path / "model.txt"
        invalid_file.write_bytes(b"dummy content")

        loader = ModelLoader()

        with pytest.raises(ValueError) as exc_info:
            loader.load_model(str(invalid_file))

        assert "invalid weights file format" in str(exc_info.value).lower()

    def test_load_model_with_real_weights(self):
        """Test loading model with actual trained weights."""
        # Check if real weights exist
        weights_path = "./weights/zero_dce.weights.h5"
        if not Path(weights_path).exists():
            pytest.skip("Real weights file not found - skipping integration test")

        loader = ModelLoader()
        model = loader.load_model(weights_path)

        # Verify model was loaded and state updated
        assert model is not None
        assert isinstance(model, ZeroDCE)
        assert loader.is_model_loaded() is True
        assert loader.get_model() is model
        assert loader.get_weights_path() == str(Path(weights_path).resolve())

    def test_load_model_updates_state(self):
        """Test load_model updates internal state correctly."""
        weights_path = "./weights/zero_dce.weights.h5"
        if not Path(weights_path).exists():
            pytest.skip("Real weights file not found")

        loader = ModelLoader()
        loader.load_model(weights_path)

        assert loader.is_model_loaded() is True
        assert loader.current_model is not None
        assert loader.current_weights_path is not None

    def test_get_model_info_after_loading(self):
        """Test get_model_info returns correct info after loading."""
        weights_path = "./weights/zero_dce.weights.h5"
        if not Path(weights_path).exists():
            pytest.skip("Real weights file not found")

        loader = ModelLoader()
        loader.load_model(weights_path)
        info = loader.get_model_info()

        assert info["loaded"] is True
        assert info["weights_path"] == str(Path(weights_path).resolve())
        assert info["file_size"] > 0
        assert info["filename"] == "zero_dce.weights.h5"

    def test_load_model_twice_replaces_previous(self):
        """Test loading a second model replaces the first."""
        weights_path = "./weights/zero_dce.weights.h5"
        if not Path(weights_path).exists():
            pytest.skip("Real weights file not found")

        loader = ModelLoader()
        model1 = loader.load_model(weights_path)
        model2 = loader.load_model(weights_path)

        # Should have replaced the model (new instance)
        assert loader.get_model() is model2
        assert loader.get_model() is not model1


class TestModelLoaderIntegration:
    """Integration tests for typical usage workflows."""

    def test_typical_workflow_load_unload(self):
        """Test typical workflow: load model, use it, unload it."""
        weights_path = "./weights/zero_dce.weights.h5"
        if not Path(weights_path).exists():
            pytest.skip("Real weights file not found")

        loader = ModelLoader()

        # Initial state
        assert not loader.is_model_loaded()

        # Load model
        model = loader.load_model(weights_path)
        assert loader.is_model_loaded()
        assert loader.get_model() is model

        # Get info
        info = loader.get_model_info()
        assert info["loaded"] is True

        # Unload
        loader.unload_model()
        assert not loader.is_model_loaded()
        assert loader.get_model() is None

    def test_validate_before_load_workflow(self, tmp_path):
        """Test workflow: validate file before attempting to load."""
        # Create invalid file
        invalid_file = tmp_path / "bad.txt"
        invalid_file.write_bytes(b"not a model")

        loader = ModelLoader()

        # Validate first
        is_valid, error = loader.validate_weights_file(str(invalid_file))
        assert not is_valid

        # Should not attempt to load invalid file
        # (This is what GUI should do)

    def test_list_and_load_workflow(self):
        """Test workflow: list available weights, then load one."""
        # List available weights
        weights = ModelLoader.list_available_weights("./weights")

        if not weights:
            pytest.skip("No weight files found in ./weights")

        # Load the first (most recent) weight file
        loader = ModelLoader()
        model = loader.load_model(weights[0])

        assert loader.is_model_loaded()
        assert isinstance(model, ZeroDCE)
