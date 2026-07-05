
import html
import time
import numpy as np
from typing import Dict, Any, Iterable, List

from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QWidget, QGroupBox, QComboBox, QPushButton, QTextEdit,
    QPlainTextEdit, QCheckBox, QToolTip, QRubberBand
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent, QRect, QSize

import pyqtgraph as pg

from scipy.spatial import cKDTree

from vmeans.ai_client import MODEL_PRESETS, ask_ai
from vmeans.colors import get_colors_for_centers

class HoverPlotSelectionMixin:
    def _draw_scatter(self):
        """Draw a clean Cartesian scatter colored by region using Qt-native graphics."""
        self.pg_plot.clear()
        self.pg_scatter_items = []
        legend = self.pg_plot.addLegend(offset=(-12, 12))

        unique_region_ids = sorted(set(int(v) for v in self.region_ids))
        for rid in unique_region_ids:
            indices = np.where(self.region_ids == rid)[0]
            if len(indices) == 0:
                continue

            color = self.colors_hex[rid] if 0 <= rid < len(self.colors_hex) else '#cccccc'
            label = self.region_labels[rid] if 0 <= rid < len(self.region_labels) else f"Region {rid}"
            item = pg.ScatterPlotItem(
                x=self.plot_x[indices],
                y=self.plot_y[indices],
                data=[int(i) for i in indices],
                size=7,
                pxMode=True,
                brush=pg.mkBrush(color),
                pen=pg.mkPen((255, 255, 255, 150), width=0.45),
                hoverable=True,
                hoverPen=pg.mkPen('#222222', width=1.2),
                name=label,
            )
            item.sigClicked.connect(self._on_pg_points_clicked)
            self.pg_plot.addItem(item)
            self.pg_scatter_items.append(item)

        self.selection_sc = pg.ScatterPlotItem(
            size=14,
            pxMode=True,
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen('#ff8c00', width=2.2),
        )
        self.pg_plot.addItem(self.selection_sc)
        self._set_equal_plot_range()


    def _set_equal_plot_range(self):
        """Match the animation view: preserve data geometry with equal x/y units."""
        if len(self.plot_x) == 0:
            return

        xmin, xmax = float(np.min(self.plot_x)), float(np.max(self.plot_x))
        ymin, ymax = float(np.min(self.plot_y)), float(np.max(self.plot_y))
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        span = max(xmax - xmin, ymax - ymin)
        if span <= 0:
            span = 1.0
        span *= 1.16
        half = span / 2.0
        self.pg_plot.setXRange(cx - half, cx + half, padding=0)
        self.pg_plot.setYRange(cy - half, cy + half, padding=0)


    def _format_value(self, value):
        if isinstance(value, (float, np.floating)):
            return f"{float(value):,.2f}"
        return str(value)


    def _region_name(self, idx: int) -> str:
        rid = int(self.region_ids[idx])
        if 0 <= rid < len(self.region_labels):
            return self.region_labels[rid]
        return "Unassigned"


    def _event_additive(self, event=None) -> bool:
        key = (getattr(event, "key", None) or "").lower()
        if "shift" in key or "control" in key or "ctrl" in key or "cmd" in key:
            return True

        modifiers = QApplication.keyboardModifiers()
        return bool(
            modifiers & Qt.KeyboardModifier.ShiftModifier
            or modifiers & Qt.KeyboardModifier.ControlModifier
            or modifiers & Qt.KeyboardModifier.MetaModifier
        )


    def eventFilter(self, obj, event):
        plot = getattr(self, "pg_plot", None)
        if plot is not None and obj is plot.viewport():
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
                self._clear_selection()
                QToolTip.hideText()
                return True
            if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                return self._on_pg_mouse_press(event)
            if etype == QEvent.Type.MouseMove:
                if self._mouse_press is not None and event.buttons() & Qt.MouseButton.LeftButton:
                    self._update_pg_drag_rectangle(event)
                    return True
                self._on_pg_hover(event)
                return False
            if etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                return self._on_pg_mouse_release(event)
            if etype == QEvent.Type.Leave:
                QToolTip.hideText()
                self._last_hover_idx = None
        elif plot is not None and self._mouse_press is not None:
            etype = event.type()
            if etype == QEvent.Type.MouseMove and hasattr(event, "globalPosition"):
                if event.buttons() & Qt.MouseButton.LeftButton:
                    local_pos = plot.viewport().mapFromGlobal(event.globalPosition().toPoint())
                    self._update_pg_drag_rectangle_at(local_pos, preview=False, remember=False)
            elif etype == QEvent.Type.MouseButtonRelease and hasattr(event, "globalPosition"):
                if event.button() == Qt.MouseButton.LeftButton:
                    local_pos = plot.viewport().mapFromGlobal(event.globalPosition().toPoint())
                    self._finish_pg_selection_at(self._last_drag_local_pos or local_pos)
        return super().eventFilter(obj, event)


    def _clamp_local_pos(self, local_pos):
        viewport = self.pg_plot.viewport()
        local_pos.setX(max(0, min(local_pos.x(), viewport.width() - 1)))
        local_pos.setY(max(0, min(local_pos.y(), viewport.height() - 1)))
        return local_pos


    def _local_to_plot_pos(self, local_pos):
        scene_pos = self.pg_plot.mapToScene(local_pos)
        if not self.pg_view.sceneBoundingRect().contains(scene_pos):
            return None
        view_pos = self.pg_view.mapSceneToView(scene_pos)
        return float(view_pos.x()), float(view_pos.y()), local_pos


    def _event_to_plot_pos(self, event, clamp: bool = False):
        local_pos = event.position().toPoint()
        if clamp:
            local_pos = self._clamp_local_pos(local_pos)
        return self._local_to_plot_pos(local_pos)


    def _nearest_point_index(self, local_pos, x: float, y: float, max_px: float = 11.0):
        k = min(8, len(self._plot_points))
        _, candidates = self._hover_tree.query([x, y], k=k)
        candidates = np.atleast_1d(candidates)

        best_idx = None
        best_dist2 = float("inf")
        for candidate in candidates:
            idx = int(candidate)
            scene_point = self.pg_view.mapViewToScene(pg.Point(self.plot_x[idx], self.plot_y[idx]))
            widget_point = self.pg_plot.mapFromScene(scene_point)
            dx = widget_point.x() - local_pos.x()
            dy = widget_point.y() - local_pos.y()
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_idx = idx
                best_dist2 = dist2

        if best_idx is not None and best_dist2 <= max_px * max_px:
            return best_idx
        return None


    def _on_pg_mouse_press(self, event) -> bool:
        mapped = self._event_to_plot_pos(event)
        if mapped is None:
            self._mouse_press = None
            return False
        x, y, local_pos = mapped
        self._mouse_press = (x, y, local_pos.x(), local_pos.y(), self._event_additive())
        self._rubber_origin = local_pos
        self._last_drag_local_pos = local_pos
        self._selection_before_drag = set(self.selected_indices)
        QToolTip.hideText()
        return True


    def _on_pg_mouse_release(self, event) -> bool:
        if self._mouse_press is None:
            self.rubber_band.hide()
            return False

        mapped = self._event_to_plot_pos(event, clamp=True)
        if mapped is None and self._last_drag_local_pos is not None:
            mapped = self._local_to_plot_pos(self._last_drag_local_pos)
        if mapped is None:
            self._mouse_press = None
            self._rubber_origin = None
            self._last_drag_local_pos = None
            self._selection_before_drag = set()
            self.rubber_band.hide()
            return True

        return self._finish_pg_selection_at(mapped[2])


    def _finish_pg_selection_at(self, local_pos) -> bool:
        local_pos = self._clamp_local_pos(local_pos)
        mapped = self._local_to_plot_pos(local_pos)
        if mapped is None:
            self._mouse_press = None
            self._rubber_origin = None
            self._last_drag_local_pos = None
            self.rubber_band.hide()
            return True

        x0, y0, px0, py0, additive_at_press = self._mouse_press
        additive = additive_at_press or self._event_additive()
        self._mouse_press = None
        self._rubber_origin = None
        self._last_drag_local_pos = None
        self._selection_before_drag = set()
        self.rubber_band.hide()

        x1, y1, local_pos = mapped
        pixel_distance = ((local_pos.x() - px0) ** 2 + (local_pos.y() - py0) ** 2) ** 0.5
        if pixel_distance < 6:
            idx = self._nearest_point_index(local_pos, x1, y1)
            if idx is not None:
                self._select_indices([idx], additive=additive, toggle_single=additive)
            elif not additive:
                self._clear_selection()
            return True

        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        indices = np.where(
            (self.plot_x >= xmin) & (self.plot_x <= xmax) &
            (self.plot_y >= ymin) & (self.plot_y <= ymax)
        )[0]
        self._select_indices(indices, additive=additive, toggle_single=False)
        return True


    def _update_pg_drag_rectangle(self, event):
        if self._rubber_origin is None:
            return
        self._update_pg_drag_rectangle_at(event.position().toPoint())


    def _update_pg_drag_rectangle_at(self, current, preview: bool = True, remember: bool = True):
        if self._rubber_origin is None:
            return
        current = self._clamp_local_pos(current)
        if remember:
            self._last_drag_local_pos = current
        rect = QRect(self._rubber_origin, current).normalized()
        if rect.size().width() < 2 and rect.size().height() < 2:
            rect.setSize(QSize(2, 2))
        self.rubber_band.setGeometry(rect)
        self.rubber_band.show()
        if preview:
            self._preview_drag_selection(current)


    def _preview_drag_selection(self, current):
        if self._mouse_press is None:
            return

        x0, y0, px0, py0, additive_at_press = self._mouse_press
        pixel_distance = ((current.x() - px0) ** 2 + (current.y() - py0) ** 2) ** 0.5
        if pixel_distance < 6:
            return

        mapped = self._local_to_plot_pos(current)
        if mapped is None:
            return

        x1, y1, _ = mapped
        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        indices = {
            int(i) for i in np.where(
                (self.plot_x >= xmin) & (self.plot_x <= xmax) &
                (self.plot_y >= ymin) & (self.plot_y <= ymax)
            )[0]
        }

        if additive_at_press or self._event_additive():
            self.selected_indices = set(self._selection_before_drag) | indices
        else:
            self.selected_indices = indices
        self._update_selection_display()


    def _select_indices(self, indices: Iterable[int], additive: bool, toggle_single: bool):
        indices = [int(i) for i in indices]
        if toggle_single and len(indices) == 1:
            idx = indices[0]
            if idx in self.selected_indices:
                self.selected_indices.remove(idx)
            else:
                self.selected_indices.add(idx)
        elif additive:
            self.selected_indices.update(indices)
        else:
            self.selected_indices = set(indices)
        self._update_selection_display()


    def _on_pg_points_clicked(self, _item, points, _event):
        if not points:
            return
        idx = int(points[0].data())
        additive = self._event_additive()
        self._select_indices([idx], additive=additive, toggle_single=additive)


    def _clear_selection(self):
        self.selected_indices.clear()
        self._update_selection_display()


    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._clear_selection()
            QToolTip.hideText()
            event.accept()
            return
        super().keyPressEvent(event)


    def closeEvent(self, event):
        if self._app_event_filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._app_event_filter_installed = False
        super().closeEvent(event)


    def _update_selection_display(self):
        self._last_hover_idx = None
        if self.selected_indices:
            indices = np.array(sorted(self.selected_indices), dtype=int)
            x = self.plot_x[indices]
            y = self.plot_y[indices]
        else:
            x = []
            y = []

        if hasattr(self, "selection_sc"):
            self.selection_sc.setData(x=x, y=y)

        if hasattr(self, "selected_count_label"):
            total = len(self.selected_indices)
            if total == 0:
                self.selected_count_label.setText("Selected points: 0")
            else:
                region_counts = self._selection_region_counts()
                summary = ", ".join(
                    f"{label}: {count}" for label, count in list(region_counts.items())[:4]
                )
                if len(region_counts) > 4:
                    summary += ", ..."
                self.selected_count_label.setText(f"Selected points: {total} ({summary})")


    def _selection_region_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for idx in sorted(self.selected_indices):
            label = self._region_name(idx)
            counts[label] = counts.get(label, 0) + 1
        return counts


    def _on_pg_hover(self, event):
        """Show compact Qt tooltip when hovering over a data point."""
        now = time.monotonic()
        if now - self._last_hover_time < 0.035:
            return
        self._last_hover_time = now

        mapped = self._event_to_plot_pos(event)
        if mapped is None:
            if self._last_hover_idx is not None:
                QToolTip.hideText()
                self._last_hover_idx = None
            return

        x, y, local_pos = mapped
        idx = self._nearest_point_index(local_pos, x, y)
        if idx is not None:
            if idx == self._last_hover_idx:
                return
            self._last_hover_idx = idx
            px, py = self.plot_x[idx], self.plot_y[idx]
            region_name = self._region_name(idx)

            if self.original_df is not None and idx < len(self.original_df) and self.original_xy_cols is not None:
                row = self.original_df.iloc[idx]
                x_col, y_col = self.original_xy_cols
                x_val = self._format_value(row[x_col])
                y_val = self._format_value(row[y_col])

                text = (
                    f"Index: {idx}\n"
                    f"Region: {region_name}\n"
                    f"{x_col}: {x_val}\n"
                    f"{y_col}: {y_val}"
                )
            else:
                text = (
                    f"Index: {idx}\n"
                    f"Region: {region_name}\n"
                    f"{self.x_label}: {px:.2f}\n"
                    f"{self.y_label}: {py:.2f}"
                )

            if idx in self.selected_indices:
                text += "\nSelected: yes"

            QToolTip.showText(self.pg_plot.viewport().mapToGlobal(local_pos), text, self.pg_plot.viewport())
        else:
            if self._last_hover_idx is not None:
                QToolTip.hideText()
                self._last_hover_idx = None
