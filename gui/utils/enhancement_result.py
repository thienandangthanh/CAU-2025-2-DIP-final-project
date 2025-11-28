"""Enhancement result container with timing and metadata.

This module provides the EnhancementResult class which encapsulates the result
of an image enhancement operation along with metadata such as elapsed time,
method name, and quality metrics. This class is designed to be extensible for
comparison features.
"""

from PIL import Image


class EnhancementResult:
    """Container for enhancement results with timing and metadata.

    This class encapsulates the result of an image enhancement operation,
    including the enhanced image, method name, and elapsed time. It provides
    utility methods for formatting timing information in a human-readable way.

    This class will be extended to include quality metrics (PSNR,
    SSIM), memory usage, and method parameters for comprehensive comparison.

    Attributes:
        image: The enhanced PIL Image (RGB)
        method_name: Name of the enhancement method (e.g., "Zero-DCE")
        elapsed_time: Time taken for enhancement in seconds (float)
        quality_metrics: Optional dictionary of quality metrics (future use)

    Example:
        >>> from PIL import Image
        >>> enhanced_img = Image.new('RGB', (256, 256))
        >>> result = EnhancementResult(enhanced_img, "Zero-DCE", 2.34)
        >>> print(result.format_time())
        '2.34s'
        >>> print(result.summary())
        'Zero-DCE: 2.34s'
    """

    def __init__(
        self,
        image: Image.Image,
        method_name: str,
        elapsed_time: float,
        quality_metrics: dict | None = None,
    ):
        """Initialize EnhancementResult.

        Args:
            image: The enhanced PIL Image (RGB)
            method_name: Name of the enhancement method (e.g., "Zero-DCE")
            elapsed_time: Time taken for enhancement in seconds
            quality_metrics: Optional dictionary of quality metrics (for Phase 3)

        Raises:
            ValueError: If elapsed_time is negative
        """
        if elapsed_time < 0:
            raise ValueError(f"elapsed_time must be non-negative, got {elapsed_time}")

        self.image = image
        self.method_name = method_name
        self.elapsed_time = elapsed_time
        self.quality_metrics = quality_metrics or {}

    def format_time(self) -> str:
        """Format elapsed time as human-readable string.

        Formats the elapsed time with appropriate precision:
        - Less than 60 seconds: "X.XXs" (2 decimal places)
        - 60 seconds or more: "Xm Ys" (minutes and seconds)

        Returns:
            Formatted time string (e.g., "2.34s" or "1m 34s")

        Example:
            >>> result = EnhancementResult(img, "Zero-DCE", 2.345)
            >>> result.format_time()
            '2.35s'
            >>> result = EnhancementResult(img, "Zero-DCE", 94.5)
            >>> result.format_time()
            '1m 34s'
        """
        if self.elapsed_time < 60:
            return f"{self.elapsed_time:.2f}s"
        else:
            minutes = int(self.elapsed_time // 60)
            seconds = int(self.elapsed_time % 60)
            return f"{minutes}m {seconds}s"

    def summary(self) -> str:
        """Get a summary string of the enhancement result.

        Returns:
            Summary in format "MethodName: time" (e.g., "Zero-DCE: 2.34s")

        Example:
            >>> result = EnhancementResult(img, "CLAHE", 0.12)
            >>> result.summary()
            'CLAHE: 0.12s'
        """
        return f"{self.method_name}: {self.format_time()}"

    def get_image_info(self) -> dict:
        """Get basic information about the enhanced image.

        Returns:
            Dictionary containing image dimensions and mode

        Example:
            >>> result = EnhancementResult(img, "Zero-DCE", 2.34)
            >>> result.get_image_info()
            {'width': 256, 'height': 256, 'mode': 'RGB'}
        """
        return {
            "width": self.image.width,
            "height": self.image.height,
            "mode": self.image.mode,
        }

    def add_quality_metric(self, metric_name: str, value: float) -> None:
        """Add a quality metric to the result (for Phase 3).

        Args:
            metric_name: Name of the quality metric (e.g., "PSNR", "SSIM")
            value: Numerical value of the metric

        Example:
            >>> result = EnhancementResult(img, "Zero-DCE", 2.34)
            >>> result.add_quality_metric("PSNR", 24.56)
            >>> result.quality_metrics
            {'PSNR': 24.56}
        """
        self.quality_metrics[metric_name] = value

    def __repr__(self) -> str:
        """Return detailed string representation of the result.

        Returns:
            String representation including all metadata
        """
        img_info = f"{self.image.width}x{self.image.height}"
        metrics_str = ""
        if self.quality_metrics:
            metrics_str = f", metrics={self.quality_metrics}"
        return (
            f"EnhancementResult(method='{self.method_name}', "
            f"time={self.elapsed_time:.2f}s, "
            f"image={img_info}{metrics_str})"
        )

    def __str__(self) -> str:
        """Return user-friendly string representation.

        Returns:
            Summary string
        """
        return self.summary()
