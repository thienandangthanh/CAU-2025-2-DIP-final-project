"""Multi-method enhancement execution engine with threading support.

This module provides the EnhancementRunner class for executing multiple image
enhancement methods sequentially with progress reporting and error handling.
Designed for integration with the GUI comparison feature.
"""

import time
from typing import Dict, List, Optional
from PIL import Image

from PyQt6.QtCore import QThread, pyqtSignal, QObject

from model import ZeroDCE
from gui.utils.enhancement_result import EnhancementResult
from gui.utils.enhancement_methods import get_registry


class EnhancementRunner(QObject):
    """Execute multiple enhancement methods with progress reporting.
    
    This class handles the execution of one or more enhancement methods on an image,
    tracking timing for each method and handling errors gracefully. Designed to be
    run in a background thread to keep the GUI responsive.
    
    Signals:
        method_started: Emitted when a method starts (method_key: str, method_name: str)
        method_completed: Emitted when a method finishes (method_key: str, result: EnhancementResult)
        method_failed: Emitted when a method fails (method_key: str, error_message: str)
        all_completed: Emitted when all methods finish (results: Dict[str, EnhancementResult])
        progress_updated: Emitted to report progress (current: int, total: int, message: str)
    
    Example:
        >>> runner = EnhancementRunner()
        >>> runner.method_completed.connect(on_method_done)
        >>> runner.all_completed.connect(on_all_done)
        >>> results = runner.run_multiple_methods(image, ["clahe", "gamma"], model=None)
    """
    
    # Signals for progress reporting
    method_started = pyqtSignal(str, str)  # method_key, method_name
    method_completed = pyqtSignal(str, object)  # method_key, EnhancementResult
    method_failed = pyqtSignal(str, str)  # method_key, error_message
    all_completed = pyqtSignal(dict)  # {method_key: EnhancementResult}
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    
    def __init__(self):
        """Initialize the enhancement runner."""
        super().__init__()
        self.registry = get_registry()
        self._is_running = False
        self._should_cancel = False
    
    def run_single_method(
        self,
        image: Image.Image,
        method_key: str,
        model: Optional[ZeroDCE] = None
    ) -> Optional[EnhancementResult]:
        """Execute a single enhancement method.
        
        Args:
            image: Input PIL Image
            method_key: Key of the method to execute
            model: ZeroDCE model (required for Zero-DCE)
            
        Returns:
            EnhancementResult object, or None if method failed
            
        Raises:
            KeyError: If method_key is not found in registry
        """
        # Get method info for display name
        method_info = self.registry.get_method_info(method_key)
        
        # Check if method can run
        model_loaded = model is not None
        if not self.registry.can_run_method(method_key, model_loaded):
            error_msg = f"Method '{method_info.name}' cannot run (model required but not loaded)"
            self.method_failed.emit(method_key, error_msg)
            return None
        
        # Emit start signal
        self.method_started.emit(method_key, method_info.name)
        
        try:
            # Execute enhancement with timing
            start_time = time.perf_counter()
            enhanced_image = self.registry.enhance_image(image, method_key, model)
            elapsed_time = time.perf_counter() - start_time
            
            # Create result
            result = EnhancementResult(
                image=enhanced_image,
                method_name=method_info.name,
                elapsed_time=elapsed_time
            )
            
            # Emit completion signal
            self.method_completed.emit(method_key, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error in {method_info.name}: {str(e)}"
            self.method_failed.emit(method_key, error_msg)
            return None
    
    def run_multiple_methods(
        self,
        image: Image.Image,
        method_keys: List[str],
        model: Optional[ZeroDCE] = None,
        emit_signals: bool = True
    ) -> Dict[str, EnhancementResult]:
        """Execute multiple enhancement methods sequentially.
        
        Methods are executed in the order provided. If one method fails, execution
        continues with the remaining methods. Progress signals are emitted for
        each method completion.
        
        Args:
            image: Input PIL Image
            method_keys: List of method keys to execute
            model: ZeroDCE model (required for Zero-DCE)
            emit_signals: Whether to emit progress signals (default: True)
            
        Returns:
            Dictionary mapping method_key to EnhancementResult.
            Failed methods are not included in the results.
            
        Example:
            >>> runner = EnhancementRunner()
            >>> results = runner.run_multiple_methods(
            ...     image,
            ...     ["clahe", "gamma", "zero-dce"],
            ...     model=my_model
            ... )
            >>> print(f"Successfully enhanced with {len(results)} methods")
        """
        self._is_running = True
        self._should_cancel = False
        
        results = {}
        total_methods = len(method_keys)
        
        for idx, method_key in enumerate(method_keys, start=1):
            # Check for cancellation
            if self._should_cancel:
                if emit_signals:
                    self.progress_updated.emit(
                        idx - 1, total_methods, "Cancelled"
                    )
                break
            
            # Update progress
            if emit_signals:
                method_info = self.registry.get_method_info(method_key)
                progress_msg = f"Processing {method_info.name}... ({idx}/{total_methods})"
                self.progress_updated.emit(idx - 1, total_methods, progress_msg)
            
            # Execute method
            result = self.run_single_method(image, method_key, model)
            
            # Store result if successful
            if result is not None:
                results[method_key] = result
        
        # Emit all completed signal
        if emit_signals and not self._should_cancel:
            self.all_completed.emit(results)
            final_msg = f"Completed {len(results)}/{total_methods} methods"
            self.progress_updated.emit(total_methods, total_methods, final_msg)
        
        self._is_running = False
        return results
    
    def cancel(self):
        """Request cancellation of the current operation.
        
        Note: Cancellation happens between methods, not during method execution.
        The current method will complete before cancellation takes effect.
        """
        self._should_cancel = True
    
    def is_running(self) -> bool:
        """Check if runner is currently executing methods.
        
        Returns:
            True if methods are being executed, False otherwise
        """
        return self._is_running


class EnhancementRunnerThread(QThread):
    """Background thread for running enhancement methods without blocking UI.
    
    This class wraps EnhancementRunner in a QThread to execute enhancement
    operations in the background, keeping the GUI responsive during processing.
    
    Signals:
        method_started: Emitted when a method starts (method_key: str, method_name: str)
        method_completed: Emitted when a method finishes (method_key: str, result: EnhancementResult)
        method_failed: Emitted when a method fails (method_key: str, error_message: str)
        all_completed: Emitted when all methods finish (results: Dict[str, EnhancementResult])
        progress_updated: Emitted to report progress (current: int, total: int, message: str)
        error_occurred: Emitted when a fatal error occurs (error_message: str)
    
    Example:
        >>> thread = EnhancementRunnerThread(image, ["clahe", "gamma"], model=None)
        >>> thread.all_completed.connect(on_results_ready)
        >>> thread.start()
    """
    
    # Signals (re-exposed from EnhancementRunner)
    method_started = pyqtSignal(str, str)
    method_completed = pyqtSignal(str, object)
    method_failed = pyqtSignal(str, str)
    all_completed = pyqtSignal(dict)
    progress_updated = pyqtSignal(int, int, str)
    error_occurred = pyqtSignal(str)
    
    def __init__(
        self,
        image: Image.Image,
        method_keys: List[str],
        model: Optional[ZeroDCE] = None,
        parent=None
    ):
        """Initialize the enhancement runner thread.
        
        Args:
            image: Input PIL Image
            method_keys: List of method keys to execute
            model: ZeroDCE model (required for Zero-DCE)
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self.image = image
        self.method_keys = method_keys
        self.model = model
        self.runner = EnhancementRunner()
        
        # Connect runner signals to thread signals
        self.runner.method_started.connect(self.method_started.emit)
        self.runner.method_completed.connect(self.method_completed.emit)
        self.runner.method_failed.connect(self.method_failed.emit)
        self.runner.all_completed.connect(self.all_completed.emit)
        self.runner.progress_updated.connect(self.progress_updated.emit)
    
    def run(self):
        """Execute enhancement methods in background thread.
        
        This method is called automatically when thread.start() is called.
        Do not call this method directly.
        """
        try:
            self.runner.run_multiple_methods(
                self.image,
                self.method_keys,
                self.model,
                emit_signals=True
            )
        except Exception as e:
            error_msg = f"Fatal error during enhancement: {str(e)}"
            self.error_occurred.emit(error_msg)
    
    def cancel(self):
        """Request cancellation of enhancement operations.
        
        Note: Cancellation happens between methods, not during method execution.
        The current method will complete before cancellation takes effect.
        """
        self.runner.cancel()
    
    def is_running(self) -> bool:
        """Check if thread is currently running.
        
        Returns:
            True if thread is running, False otherwise
        """
        return self.isRunning()


# Convenience function for synchronous execution
def enhance_with_methods(
    image: Image.Image,
    method_keys: List[str],
    model: Optional[ZeroDCE] = None
) -> Dict[str, EnhancementResult]:
    """Execute multiple enhancement methods synchronously (blocking).
    
    This is a convenience function for simple use cases where threading is
    not needed. For GUI applications, use EnhancementRunnerThread instead.
    
    Args:
        image: Input PIL Image
        method_keys: List of method keys to execute
        model: ZeroDCE model (required for Zero-DCE)
        
    Returns:
        Dictionary mapping method_key to EnhancementResult
        
    Example:
        >>> from PIL import Image
        >>> image = Image.open("input.jpg")
        >>> results = enhance_with_methods(image, ["clahe", "gamma"])
        >>> for key, result in results.items():
        ...     print(f"{result.method_name}: {result.format_time()}")
    """
    runner = EnhancementRunner()
    return runner.run_multiple_methods(image, method_keys, model, emit_signals=False)
