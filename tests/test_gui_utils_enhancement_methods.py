"""Tests for gui/utils/enhancement_methods.py module.

This module tests the enhancement method registry functionality, including
method discovery, validation, and execution.
"""

import pytest
from PIL import Image
import numpy as np

from gui.utils.enhancement_methods import (
    EnhancementMethod,
    EnhancementMethodRegistry,
    ExecutionSpeed,
    get_registry,
    get_available_methods,
    get_method_info,
    can_run_method,
    get_runnable_methods,
    enhance_image,
)


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a 256x256 RGB image with random values
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def registry():
    """Provide a fresh registry instance."""
    return EnhancementMethodRegistry()


class TestEnhancementMethodRegistry:
    """Tests for EnhancementMethodRegistry class."""
    
    def test_registry_initialization(self, registry):
        """Test that registry initializes with methods."""
        methods = registry.get_available_methods()
        assert len(methods) > 0
        assert "zero-dce" in methods
        assert "clahe" in methods
        assert "autocontrast" in methods
    
    def test_get_available_methods(self, registry):
        """Test getting all available method keys."""
        methods = registry.get_available_methods()
        
        # Should include Zero-DCE + 4 classical methods
        assert len(methods) == 5
        assert "zero-dce" in methods
        assert "autocontrast" in methods
        assert "histogram-eq" in methods
        assert "clahe" in methods
        assert "gamma" in methods
    
    def test_get_method_info_valid(self, registry):
        """Test getting method info for valid method."""
        method = registry.get_method_info("zero-dce")
        
        assert isinstance(method, EnhancementMethod)
        assert method.key == "zero-dce"
        assert method.name == "Zero-DCE"
        assert method.requires_model is True
        assert method.category == "deep_learning"
        assert method.speed_category == ExecutionSpeed.SLOW
    
    def test_get_method_info_invalid(self, registry):
        """Test getting method info for invalid method raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            registry.get_method_info("nonexistent-method")
        
        assert "not found in registry" in str(exc_info.value)
    
    def test_can_run_method_zero_dce_with_model(self, registry):
        """Test that Zero-DCE can run when model is loaded."""
        assert registry.can_run_method("zero-dce", model_loaded=True) is True
    
    def test_can_run_method_zero_dce_without_model(self, registry):
        """Test that Zero-DCE cannot run without model."""
        assert registry.can_run_method("zero-dce", model_loaded=False) is False
    
    def test_can_run_method_classical(self, registry):
        """Test that classical methods can always run."""
        assert registry.can_run_method("clahe", model_loaded=False) is True
        assert registry.can_run_method("clahe", model_loaded=True) is True
    
    def test_get_methods_by_category_classical(self, registry):
        """Test getting methods by classical category."""
        classical = registry.get_methods_by_category("classical")
        
        assert len(classical) == 4
        assert "autocontrast" in classical
        assert "histogram-eq" in classical
        assert "clahe" in classical
        assert "gamma" in classical
    
    def test_get_methods_by_category_deep_learning(self, registry):
        """Test getting methods by deep_learning category."""
        deep_learning = registry.get_methods_by_category("deep_learning")
        
        assert len(deep_learning) == 1
        assert "zero-dce" in deep_learning
    
    def test_get_methods_by_speed_fast(self, registry):
        """Test getting fast methods."""
        fast_methods = registry.get_methods_by_speed(ExecutionSpeed.FAST)
        
        # All classical methods should be fast
        assert len(fast_methods) == 4
        assert "clahe" in fast_methods
        assert "autocontrast" in fast_methods
    
    def test_get_methods_by_speed_slow(self, registry):
        """Test getting slow methods."""
        slow_methods = registry.get_methods_by_speed(ExecutionSpeed.SLOW)
        
        # Only Zero-DCE should be slow
        assert len(slow_methods) == 1
        assert "zero-dce" in slow_methods
    
    def test_get_runnable_methods_with_model(self, registry):
        """Test getting runnable methods when model is loaded."""
        runnable = registry.get_runnable_methods(model_loaded=True)
        
        # All methods should be runnable
        assert len(runnable) == 5
        assert "zero-dce" in runnable
        assert "clahe" in runnable
    
    def test_get_runnable_methods_without_model(self, registry):
        """Test getting runnable methods when model is not loaded."""
        runnable = registry.get_runnable_methods(model_loaded=False)
        
        # Only classical methods should be runnable
        assert len(runnable) == 4
        assert "zero-dce" not in runnable
        assert "clahe" in runnable
        assert "autocontrast" in runnable
    
    def test_enhance_image_classical_method(self, registry, sample_image):
        """Test enhancing image with classical method."""
        enhanced = registry.enhance_image(sample_image, "autocontrast", model=None)
        
        assert isinstance(enhanced, Image.Image)
        assert enhanced.size == sample_image.size
        assert enhanced.mode == "RGB"
    
    def test_enhance_image_zero_dce_without_model(self, registry, sample_image):
        """Test that Zero-DCE enhancement fails without model."""
        with pytest.raises(ValueError) as exc_info:
            registry.enhance_image(sample_image, "zero-dce", model=None)
        
        assert "requires a model" in str(exc_info.value)
    
    def test_enhance_image_invalid_method(self, registry, sample_image):
        """Test that enhancing with invalid method raises KeyError."""
        with pytest.raises(KeyError):
            registry.enhance_image(sample_image, "invalid-method", model=None)
    
    def test_get_all_methods_info(self, registry):
        """Test getting all methods info returns complete dictionary."""
        all_methods = registry.get_all_methods_info()
        
        assert len(all_methods) == 5
        assert all(isinstance(m, EnhancementMethod) for m in all_methods.values())
        assert "zero-dce" in all_methods
        assert "clahe" in all_methods


class TestGlobalRegistryFunctions:
    """Tests for global convenience functions."""
    
    def test_get_registry_singleton(self):
        """Test that get_registry returns singleton instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        
        assert registry1 is registry2
    
    def test_get_available_methods_function(self):
        """Test get_available_methods convenience function."""
        methods = get_available_methods()
        
        assert isinstance(methods, list)
        assert len(methods) == 5
        assert "zero-dce" in methods
    
    def test_get_method_info_function(self):
        """Test get_method_info convenience function."""
        method = get_method_info("clahe")
        
        assert isinstance(method, EnhancementMethod)
        assert method.key == "clahe"
        assert method.name == "CLAHE"
    
    def test_can_run_method_function(self):
        """Test can_run_method convenience function."""
        assert can_run_method("zero-dce", model_loaded=True) is True
        assert can_run_method("zero-dce", model_loaded=False) is False
        assert can_run_method("clahe", model_loaded=False) is True
    
    def test_get_runnable_methods_function(self):
        """Test get_runnable_methods convenience function."""
        with_model = get_runnable_methods(model_loaded=True)
        without_model = get_runnable_methods(model_loaded=False)
        
        assert len(with_model) == 5
        assert len(without_model) == 4
        assert "zero-dce" in with_model
        assert "zero-dce" not in without_model
    
    def test_enhance_image_function(self, sample_image):
        """Test enhance_image convenience function."""
        enhanced = enhance_image(sample_image, "gamma", model=None)
        
        assert isinstance(enhanced, Image.Image)
        assert enhanced.size == sample_image.size


class TestEnhancementMethodDataclass:
    """Tests for EnhancementMethod dataclass."""
    
    def test_enhancement_method_creation(self):
        """Test creating an EnhancementMethod instance."""
        def dummy_function(img):
            return img
        
        method = EnhancementMethod(
            key="test-method",
            name="Test Method",
            description="A test method",
            category="test",
            requires_model=False,
            speed_category=ExecutionSpeed.FAST,
            enhance_function=dummy_function
        )
        
        assert method.key == "test-method"
        assert method.name == "Test Method"
        assert method.description == "A test method"
        assert method.category == "test"
        assert method.requires_model is False
        assert method.speed_category == ExecutionSpeed.FAST
        assert method.enhance_function == dummy_function


class TestExecutionSpeed:
    """Tests for ExecutionSpeed enum."""
    
    def test_execution_speed_values(self):
        """Test that ExecutionSpeed enum has correct values."""
        assert ExecutionSpeed.FAST.value == "fast"
        assert ExecutionSpeed.MEDIUM.value == "medium"
        assert ExecutionSpeed.SLOW.value == "slow"


@pytest.mark.integration
class TestEnhancementWithRealImages:
    """Integration tests with real image processing."""
    
    def test_all_classical_methods_execute(self, sample_image):
        """Test that all classical methods can execute successfully."""
        classical_methods = ["autocontrast", "histogram-eq", "clahe", "gamma"]
        
        for method_key in classical_methods:
            enhanced = enhance_image(sample_image, method_key, model=None)
            
            assert isinstance(enhanced, Image.Image)
            assert enhanced.size == sample_image.size
            assert enhanced.mode == "RGB"
    
    def test_enhanced_images_are_different(self, sample_image):
        """Test that enhanced images differ from original."""
        enhanced = enhance_image(sample_image, "gamma", model=None)
        
        # Convert to numpy arrays
        original_array = np.array(sample_image)
        enhanced_array = np.array(enhanced)
        
        # Images should not be identical (gamma correction changes values)
        assert not np.array_equal(original_array, enhanced_array)
    
    def test_different_methods_produce_different_results(self, sample_image):
        """Test that different methods produce different results."""
        clahe = enhance_image(sample_image, "clahe", model=None)
        gamma = enhance_image(sample_image, "gamma", model=None)
        
        clahe_array = np.array(clahe)
        gamma_array = np.array(gamma)
        
        # Different methods should produce different results
        assert not np.array_equal(clahe_array, gamma_array)
