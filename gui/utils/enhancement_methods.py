"""Enhancement method registry for comparison feature.

This module provides a unified registry for all available image enhancement methods,
including both deep learning (Zero-DCE) and classical methods. It defines a standard
interface for method metadata and execution, making it easy to add new methods and
discover available options.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from PIL import Image

import classical_methods
from gui.utils.image_processor import ImageProcessor
from model import ZeroDCE


class ExecutionSpeed(Enum):
    """Execution speed category for enhancement methods."""

    FAST = "fast"  # < 0.5s (classical methods)
    MEDIUM = "medium"  # 0.5s - 2s
    SLOW = "slow"  # > 2s (deep learning methods)


@dataclass
class EnhancementMethod:
    """Metadata and execution information for an enhancement method.

    Attributes:
        key: Unique identifier for the method (e.g., "zero-dce", "clahe")
        name: Display name for the method (e.g., "Zero-DCE", "CLAHE")
        description: Brief description of the method
        category: Method category ("deep_learning" or "classical")
        requires_model: Whether the method requires a loaded model
        speed_category: Estimated execution speed (fast/medium/slow)
        enhance_function: Function to execute the enhancement
    """

    key: str
    name: str
    description: str
    category: str
    requires_model: bool
    speed_category: ExecutionSpeed
    enhance_function: Callable


class EnhancementMethodRegistry:
    """Registry for all available enhancement methods.

    This class maintains a dictionary of all registered enhancement methods
    and provides utilities for discovering, querying, and executing them.
    """

    def __init__(self):
        """Initialize the registry with all available methods."""
        self._methods: dict[str, EnhancementMethod] = {}
        self._register_all_methods()

    def _register_all_methods(self) -> None:
        """Register all available enhancement methods."""
        # Register Zero-DCE (deep learning method)
        self._methods["zero-dce"] = EnhancementMethod(
            key="zero-dce",
            name="Zero-DCE",
            description="Deep learning enhancement using Zero-Reference Deep Curve Estimation",
            category="deep_learning",
            requires_model=True,
            speed_category=ExecutionSpeed.SLOW,
            enhance_function=self._enhance_with_zero_dce,
        )

        # Register classical methods from classical_methods module
        for method_key, method_info in classical_methods.CLASSICAL_METHODS.items():
            self._methods[method_key] = EnhancementMethod(
                key=method_key,
                name=method_info["name"],
                description=method_info["description"],
                category="classical",
                requires_model=False,
                speed_category=ExecutionSpeed.FAST,
                enhance_function=method_info["function"],
            )

    def _enhance_with_zero_dce(
        self, image: Image.Image, model: ZeroDCE | None = None
    ) -> Image.Image:
        """Wrapper function for Zero-DCE enhancement.

        Args:
            image: Input PIL Image
            model: ZeroDCE model instance (required)

        Returns:
            Enhanced PIL Image

        Raises:
            ValueError: If model is not provided
        """
        if model is None:
            raise ValueError("Zero-DCE requires a loaded model")

        return ImageProcessor.enhance_image(image, model)

    def get_available_methods(self) -> list[str]:
        """Get list of all available method keys.

        Returns:
            List of method keys (e.g., ["zero-dce", "autocontrast", "clahe", ...])
        """
        return list(self._methods.keys())

    def get_method_info(self, method_key: str) -> EnhancementMethod:
        """Get detailed information about a specific method.

        Args:
            method_key: Unique key for the method

        Returns:
            EnhancementMethod object with method metadata

        Raises:
            KeyError: If method_key is not found in registry
        """
        if method_key not in self._methods:
            raise KeyError(f"Enhancement method '{method_key}' not found in registry")

        return self._methods[method_key]

    def can_run_method(self, method_key: str, model_loaded: bool) -> bool:
        """Check if a method can be executed given current state.

        Args:
            method_key: Unique key for the method
            model_loaded: Whether a model is currently loaded

        Returns:
            True if method can run, False otherwise

        Raises:
            KeyError: If method_key is not found in registry
        """
        method = self.get_method_info(method_key)

        # Method can run if it doesn't require a model, or if model is loaded
        if method.requires_model:
            return model_loaded
        return True

    def get_methods_by_category(self, category: str) -> list[str]:
        """Get all method keys for a specific category.

        Args:
            category: Method category ("deep_learning" or "classical")

        Returns:
            List of method keys in the specified category
        """
        return [
            key for key, method in self._methods.items() if method.category == category
        ]

    def get_methods_by_speed(self, speed: ExecutionSpeed) -> list[str]:
        """Get all method keys for a specific speed category.

        Args:
            speed: ExecutionSpeed enum value (FAST, MEDIUM, or SLOW)

        Returns:
            List of method keys in the specified speed category
        """
        return [
            key
            for key, method in self._methods.items()
            if method.speed_category == speed
        ]

    def get_runnable_methods(self, model_loaded: bool) -> list[str]:
        """Get all methods that can currently run.

        Args:
            model_loaded: Whether a model is currently loaded

        Returns:
            List of method keys that can be executed
        """
        return [
            key
            for key in self._methods.keys()
            if self.can_run_method(key, model_loaded)
        ]

    def enhance_image(
        self, image: Image.Image, method_key: str, model: ZeroDCE | None = None
    ) -> Image.Image:
        """Execute enhancement using the specified method.

        Args:
            image: Input PIL Image
            method_key: Unique key for the method to use
            model: ZeroDCE model (required for Zero-DCE, ignored for classical methods)

        Returns:
            Enhanced PIL Image

        Raises:
            KeyError: If method_key is not found
            ValueError: If method requires model but none is provided
        """
        method = self.get_method_info(method_key)

        # Check if method can run
        if method.requires_model and model is None:
            raise ValueError(
                f"Method '{method.name}' requires a model, but none was provided"
            )

        # Execute enhancement
        if method.requires_model:
            return method.enhance_function(image, model)
        else:
            return method.enhance_function(image)

    def get_all_methods_info(self) -> dict[str, EnhancementMethod]:
        """Get all registered methods with their metadata.

        Returns:
            Dictionary mapping method keys to EnhancementMethod objects
        """
        return self._methods.copy()


# Global registry instance (singleton pattern)
_registry = None


def get_registry() -> EnhancementMethodRegistry:
    """Get the global enhancement method registry.

    Returns:
        Singleton instance of EnhancementMethodRegistry
    """
    global _registry
    if _registry is None:
        _registry = EnhancementMethodRegistry()
    return _registry


# Convenience functions for backward compatibility and ease of use


def get_available_methods() -> list[str]:
    """Get list of all available method keys.

    Returns:
        List of method keys
    """
    return get_registry().get_available_methods()


def get_method_info(method_key: str) -> EnhancementMethod:
    """Get information about a specific method.

    Args:
        method_key: Unique key for the method

    Returns:
        EnhancementMethod object with method metadata
    """
    return get_registry().get_method_info(method_key)


def can_run_method(method_key: str, model_loaded: bool) -> bool:
    """Check if a method can be executed.

    Args:
        method_key: Unique key for the method
        model_loaded: Whether a model is currently loaded

    Returns:
        True if method can run, False otherwise
    """
    return get_registry().can_run_method(method_key, model_loaded)


def get_runnable_methods(model_loaded: bool) -> list[str]:
    """Get all methods that can currently run.

    Args:
        model_loaded: Whether a model is currently loaded

    Returns:
        List of method keys that can be executed
    """
    return get_registry().get_runnable_methods(model_loaded)


def enhance_image(
    image: Image.Image, method_key: str, model: ZeroDCE | None = None
) -> Image.Image:
    """Execute enhancement using the specified method.

    Args:
        image: Input PIL Image
        method_key: Unique key for the method to use
        model: ZeroDCE model (required for Zero-DCE)

    Returns:
        Enhanced PIL Image
    """
    return get_registry().enhance_image(image, method_key, model)
