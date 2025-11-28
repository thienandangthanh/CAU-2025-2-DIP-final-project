"""Histogram overlay widget for image panels.

Provides a lightweight, draggable overlay that can render either grayscale or
RGB histograms for a given QPixmap. The overlay is intentionally
self-contained so it can be embedded inside both ImagePanel and
ComparisonCell widgets without creating tight coupling.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import QWidget

HistogramData = np.ndarray | dict[str, np.ndarray]


class HistogramOverlay(QWidget):
    """Draggable histogram overlay that renders over an image label."""

    DEFAULT_SIZE = (220, 130)
    TARGET_BINS = 64

    def __init__(
        self,
        parent: QWidget | None = None,
        histogram_type: str = "grayscale",
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.resize(*self.DEFAULT_SIZE)

        self.histogram_type = histogram_type
        self.current_pixmap: QPixmap | None = None
        self._hist_cache: dict[str, HistogramData] = {}
        self._dragging = False
        self._drag_offset = QPoint()
        self._custom_position: QPoint | None = None

        # Hidden by default; parent widget controls visibility explicitly
        self.hide()

    # ------------------------------------------------------------------ API --

    def set_pixmap(self, pixmap: QPixmap | None):
        """Attach a pixmap and invalidate cached histograms."""
        self.current_pixmap = pixmap
        self._hist_cache.clear()
        if pixmap is None or pixmap.isNull():
            return
        self._ensure_histogram_cached()
        if self.isVisible():
            self.update()

    def set_histogram_type(self, histogram_type: str):
        """Change histogram type (``rgb`` or ``grayscale``)."""
        if histogram_type not in {"rgb", "grayscale"}:
            raise ValueError("Histogram type must be 'rgb' or 'grayscale'")
        if histogram_type == self.histogram_type:
            return

        self.histogram_type = histogram_type
        if self.current_pixmap is not None:
            self._ensure_histogram_cached()
            self.update()

    def histogram_snapshot(self) -> list[float] | dict[str, list[float]] | None:
        """Return a copy of the currently cached histogram data.

        This is primarily used for tests and potential analytics integrations.
        """
        data = self._hist_cache.get(self.histogram_type)
        if data is None:
            return None

        if isinstance(data, dict):
            return {channel: values.tolist() for channel, values in data.items()}
        return data.tolist()

    def has_custom_position(self) -> bool:
        """Return True if the overlay has been manually repositioned."""
        return self._custom_position is not None

    def move_to_default(self):
        """Snap overlay to bottom-right corner of the parent widget."""
        parent = self.parentWidget()
        if not parent:
            return

        margin = 12
        x = max(margin, parent.width() - self.width() - margin)
        y = max(margin, parent.height() - self.height() - margin)
        self.move(x, y)
        self._custom_position = None

    def ensure_within_parent(self):
        """Make sure the overlay stays fully inside the parent widget."""
        parent = self.parentWidget()
        if not parent:
            return

        x = min(max(0, self.x()), max(0, parent.width() - self.width()))
        y = min(max(0, self.y()), max(0, parent.height() - self.height()))
        self.move(x, y)

    # ------------------------------------------------------------ Qt events --

    def mousePressEvent(self, event: QMouseEvent):  # noqa: D401
        """Start drag when user clicks the overlay."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_offset = event.position().toPoint()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):  # noqa: D401
        """Update overlay position during drag."""
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            parent = self.parentWidget()
            if parent:
                new_pos = event.globalPosition().toPoint() - self._drag_offset
                new_pos = parent.mapFromGlobal(new_pos)
                self.move(new_pos)
                self.ensure_within_parent()
                self._custom_position = self.pos()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):  # noqa: D401
        """Stop dragging."""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def paintEvent(self, event: QPaintEvent):  # noqa: D401
        """Render histogram."""
        super().paintEvent(event)
        if self.current_pixmap is None:
            return

        data = self._ensure_histogram_cached()
        if data is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 10, 10)

        content_rect = self.rect().adjusted(12, 18, -12, -20)
        self._draw_axes(painter, content_rect)

        if self.histogram_type == "grayscale":
            self._draw_grayscale(painter, content_rect, data)  # type: ignore[arg-type]
        else:
            self._draw_rgb(painter, content_rect, data)  # type: ignore[arg-type]

        # Title
        painter.setPen(QPen(Qt.GlobalColor.white))
        title = "Histogram (RGB)" if self.histogram_type == "rgb" else "Histogram"
        painter.drawText(
            QRect(self.rect().left() + 10, self.rect().top() + 5, self.width(), 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            title,
        )

    # --------------------------------------------------------- Drawing utils --

    def _draw_axes(self, painter: QPainter, rect: QRect):
        painter.setPen(QColor(255, 255, 255, 100))
        painter.drawRect(rect)

    def _draw_grayscale(self, painter: QPainter, rect: QRect, data: np.ndarray):
        bins = self._downsample(data)
        if bins is None:
            return

        painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
        width = rect.width()
        height = rect.height()
        step = width / max(1, len(bins) - 1)

        path_points = []
        for idx, value in enumerate(bins):
            x = rect.left() + idx * step
            y = rect.bottom() - value * height
            path_points.append((x, y))

        for idx in range(len(path_points) - 1):
            start = path_points[idx]
            end = path_points[idx + 1]
            painter.drawLine(
                int(start[0]),
                int(start[1]),
                int(end[0]),
                int(end[1]),
            )

    def _draw_rgb(self, painter: QPainter, rect: QRect, data: dict[str, np.ndarray]):
        colors = {
            "r": QColor(244, 67, 54, 210),
            "g": QColor(76, 175, 80, 210),
            "b": QColor(33, 150, 243, 210),
        }

        bins = {channel: self._downsample(values) for channel, values in data.items()}

        width = rect.width()
        height = rect.height()
        max_bins = max(len(values) for values in bins.values() if values is not None)
        step = width / max(1, max_bins - 1)

        for channel, values in bins.items():
            if values is None:
                continue
            painter.setPen(QPen(colors[channel], 2))
            for idx in range(len(values) - 1):
                start_y = rect.bottom() - values[idx] * height
                end_y = rect.bottom() - values[idx + 1] * height
                start_x = rect.left() + idx * step
                end_x = rect.left() + (idx + 1) * step
                painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))

    def _downsample(self, values: np.ndarray) -> np.ndarray | None:
        if values.size == 0:
            return None
        if values.size == self.TARGET_BINS:
            return values
        factor = values.size // self.TARGET_BINS
        if factor <= 1:
            return values
        trimmed = values[: factor * self.TARGET_BINS]
        reshaped = trimmed.reshape(self.TARGET_BINS, factor)
        return reshaped.mean(axis=1)

    # ------------------------------------------------------- Histogram calc --

    def _ensure_histogram_cached(self) -> HistogramData | None:
        if self.current_pixmap is None:
            return None

        if self.histogram_type in self._hist_cache:
            return self._hist_cache[self.histogram_type]

        rgb_data = self._pixmap_to_rgb_array()
        if rgb_data is None:
            return None

        if self.histogram_type == "grayscale":
            hist = self._calculate_grayscale(rgb_data)
        else:
            hist = self._calculate_rgb(rgb_data)

        self._hist_cache[self.histogram_type] = hist
        return hist

    def _pixmap_to_rgb_array(self) -> np.ndarray | None:
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return None

        image = self.current_pixmap.toImage().convertToFormat(
            QImage.Format.Format_RGBA8888
        )
        width = image.width()
        height = image.height()
        if width == 0 or height == 0:
            return None

        ptr = image.constBits()
        ptr.setsize(width * height * 4)
        array = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return array[..., :3]

    def _calculate_grayscale(self, rgb: np.ndarray) -> np.ndarray:
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
            np.uint8
        )
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        return self._normalize_histogram(hist)

    def _calculate_rgb(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        channels = {}
        for idx, key in enumerate(["r", "g", "b"]):
            hist, _ = np.histogram(rgb[..., idx], bins=256, range=(0, 255))
            channels[key] = self._normalize_histogram(hist)
        return channels

    def _normalize_histogram(self, hist: np.ndarray) -> np.ndarray:
        hist = hist.astype(np.float32)
        max_value = hist.max()
        if max_value > 0:
            hist /= max_value
        return hist
