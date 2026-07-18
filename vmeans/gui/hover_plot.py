
import html
import time
import numpy as np
from typing import Dict, Any, Iterable, List

from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QWidget, QGroupBox, QComboBox, QPushButton, QTextEdit,
    QPlainTextEdit, QCheckBox, QToolTip, QRubberBand
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent, QRect, QSize, QPointF

import pyqtgraph as pg

from scipy.spatial import cKDTree

from vmeans.ai_client import MODEL_PRESETS, ask_ai
from vmeans.colors import get_colors_for_centers


class RegionLegendSample(pg.graphicsItems.LegendItem.ItemSample):
    """Legend marker that emits clicks without hiding its scatter item."""

    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            self.sigClicked.emit(self.item)


class RegionLegendLabel(pg.LabelItem):
    sigClicked = pyqtSignal(object)

    def __init__(self, item, text, **kwargs):
        super().__init__(text, **kwargs)
        self.plot_item = item
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            self.sigClicked.emit(self.plot_item)


class RegionLegend(pg.LegendItem):
    """Legend whose markers and labels both act as region-focus buttons."""

    def __init__(self, **kwargs):
        super().__init__(sampleType=RegionLegendSample, **kwargs)

    def addItem(self, item, name):
        label = RegionLegendLabel(
            item,
            name,
            color=self.opts['labelTextColor'],
            justify='left',
            size=self.opts['labelTextSize'],
        )
        sample = RegionLegendSample(item)
        sample.sigClicked.connect(self.sigSampleClicked)
        label.sigClicked.connect(self.sigSampleClicked)
        self.items.append((sample, label))
        self._addItemToLayout(sample, label)
        self.updateSize()


class HoverPlotSelectionMixin:
    def _draw_scatter(self):
        """Draw a clean Cartesian scatter colored by region using Qt-native graphics."""
        self.pg_plot.clear()
        self.pg_scatter_items = []
        self.pg_plot.plotItem.legend = None
        legend = RegionLegend(offset=(-12, 12), labelTextSize='9pt')
        legend.setParentItem(self.pg_view)
        self.pg_plot.plotItem.legend = legend
        legend.sigSampleClicked.connect(self._on_region_legend_clicked)
        self.region_item_ids = {}
        self.region_legend_labels = {}

        # A scene item has no OS tooltip timeout: it remains visible for as
        # long as the pointer stays on the same point.
        self.hover_card = pg.TextItem(
            color='#202124',
            anchor=(0, 1),
            border=pg.mkPen('#30343b', width=1.2),
            fill=pg.mkBrush(255, 255, 255, 238),
            ensureInBounds=True,
        )
        self.hover_card.setZValue(1000)
        self.hover_card.hide()
        self.pg_plot.addItem(self.hover_card, ignoreBounds=True)

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
                tip=None,
                name=label,
            )
            item.sigClicked.connect(self._on_pg_points_clicked)
            self.pg_plot.addItem(item)
            self.pg_scatter_items.append(item)
            self.region_item_ids[id(item)] = rid
            self.region_legend_labels[rid] = legend.items[-1][1]

        self.region_focus_sc = pg.ScatterPlotItem(
            size=13,
            pxMode=True,
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen('#111111', width=2.0),
        )
        self.pg_plot.addItem(self.region_focus_sc)

        self.selection_sc = pg.ScatterPlotItem(
            size=14,
            pxMode=True,
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen('#ff8c00', width=2.2),
        )
        self.pg_plot.addItem(self.selection_sc)
        self._update_region_focus()
        self._set_equal_plot_range()

    def _on_region_legend_clicked(self, item):
        """Toggle an outline around every point in the clicked region."""
        item.setVisible(True)
        rid = self.region_item_ids.get(id(item))
        if rid is None:
            return
        self.focused_region_id = None if self.focused_region_id == rid else rid
        self._sync_region_focus_combo()
        self._update_region_focus()

    def _on_region_focus_changed(self, combo_index: int):
        """Focus a region selected from the explicit cluster control."""
        self.focused_region_id = self.region_focus_combo.itemData(combo_index)
        self._update_region_focus()

    def _sync_region_focus_combo(self):
        combo = getattr(self, 'region_focus_combo', None)
        if combo is None:
            return
        target = 0
        for index in range(combo.count()):
            if combo.itemData(index) == self.focused_region_id:
                target = index
                break
        combo.blockSignals(True)
        combo.setCurrentIndex(target)
        combo.blockSignals(False)

    def _update_region_focus(self):
        rid = self.focused_region_id
        if rid is None:
            indices = np.array([], dtype=int)
        else:
            indices = np.where(self.region_ids == rid)[0]

        if hasattr(self, 'region_focus_sc'):
            self.region_focus_sc.setData(
                x=self.plot_x[indices] if len(indices) else [],
                y=self.plot_y[indices] if len(indices) else [],
            )

        for region_id, label_item in getattr(self, 'region_legend_labels', {}).items():
            label = (
                self.region_labels[region_id]
                if 0 <= region_id < len(self.region_labels)
                else f"Region {region_id + 1}"
            )
            label_item.setText(f"▶ {label}" if region_id == rid else label)

        if hasattr(self, 'info_label'):
            if rid is None:
                self.info_label.setText(
                    f"Points: {len(self.points)}  |  Regions: {self.data['n_regions']}  |  "
                    "Click a legend region to outline all its points"
                )
            else:
                label = self._region_name(int(indices[0])) if len(indices) else f"Region {rid + 1}"
                self.info_label.setText(
                    f"Focused: {label}  |  {len(indices)} points  |  "
                    "Other regions remain visible; click again to clear"
                )


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

    def _cluster_sample_indices(self, idx: int, max_samples: int = 4) -> List[int]:
        """Return the hovered row plus stable examples from the same region."""
        rid = int(self.region_ids[idx])
        peers = np.where(self.region_ids == rid)[0]
        others = peers[peers != idx]
        if len(others) > max_samples - 1:
            positions = np.linspace(0, len(others) - 1, max_samples - 1, dtype=int)
            others = others[positions]
        return [idx, *[int(i) for i in others[:max_samples - 1]]]

    def _point_tooltip_text(self, idx: int) -> str:
        """Build one consistent tooltip with multiple same-cluster samples."""
        region_name = self._region_name(idx)
        region_size = int(np.sum(self.region_ids == self.region_ids[idx]))
        lines = [f"Index: {idx}", f"Region: {region_name} ({region_size} samples)"]
        sample_indices = self._cluster_sample_indices(idx)

        if self.has_original_coords:
            x_col, y_col = self.original_xy_cols
            row = self.original_df.iloc[idx]
            lines.extend([
                f"{x_col}: {self._format_value(row[x_col])}",
                f"{y_col}: {self._format_value(row[y_col])}",
                "Other samples from this cluster:",
            ])
            for sample_idx in sample_indices[1:]:
                sample = self.original_df.iloc[sample_idx]
                lines.append(
                    f"  #{sample_idx}: {x_col}={self._format_value(sample[x_col])}; "
                    f"{y_col}={self._format_value(sample[y_col])}"
                )
        else:
            lines.extend([
                f"{self.x_label}: {self.plot_x[idx]:.2f}",
                f"{self.y_label}: {self.plot_y[idx]:.2f}",
                "Other samples from this cluster:",
            ])
            for sample_idx in sample_indices[1:]:
                lines.append(
                    f"  #{sample_idx}: x={self.plot_x[sample_idx]:.2f}; "
                    f"y={self.plot_y[sample_idx]:.2f}"
                )
        if len(sample_indices) == 1:
            lines.append("  No additional samples in this cluster")
        if idx in self.selected_indices:
            lines.append("Selected: yes")
        return "\n".join(lines)

    def _show_hover_card(self, idx: int) -> None:
        """Show the card in the least obstructive of four nearby quadrants."""
        x = float(self.plot_x[idx])
        y = float(self.plot_y[idx])
        self.hover_card.setText(self._point_tooltip_text(idx))

        point_scene = self.pg_view.mapViewToScene(QPointF(x, y))
        plot_rect = self.pg_view.sceneBoundingRect().adjusted(8, 8, -8, -8)
        card_rect = self.hover_card.boundingRect()
        width = max(float(card_rect.width()), 1.0)
        height = max(float(card_rect.height()), 1.0)
        gap = 14.0

        # Ordered to prefer the right-hand side when obstruction scores tie.
        candidates = [
            ((0, 1), gap, -gap),
            ((0, 0), gap, gap),
            ((1, 1), -gap, -gap),
            ((1, 0), -gap, gap),
        ]
        # Search the whole plot as well, so distant empty space is preferred to
        # covering a dense cluster beside the hovered point.
        grid_left = float(plot_rect.left())
        grid_top = float(plot_rect.top())
        max_left = max(float(plot_rect.right()) - width, grid_left)
        max_top = max(float(plot_rect.bottom()) - height, grid_top)
        for left in np.linspace(grid_left, max_left, 9):
            for top in np.linspace(grid_top, max_top, 7):
                candidates.append(((0, 0), float(left) - float(point_scene.x()),
                                   float(top) - float(point_scene.y())))
        scene_points = [
            self.pg_view.mapViewToScene(QPointF(float(px), float(py)))
            for px, py in self._plot_points
        ]
        legend = getattr(self.pg_plot.plotItem, 'legend', None)
        legend_rect = legend.sceneBoundingRect() if legend is not None else None

        best = None
        for preference, (anchor, dx, dy) in enumerate(candidates):
            pos_x = float(point_scene.x()) + dx
            pos_y = float(point_scene.y()) + dy
            left = pos_x - anchor[0] * width
            top = pos_y - anchor[1] * height
            right = left + width
            bottom = top + height

            covered = sum(
                left - 4 <= float(point.x()) <= right + 4
                and top - 4 <= float(point.y()) <= bottom + 4
                for point in scene_points
            )
            overflow = (
                max(float(plot_rect.left()) - left, 0.0)
                + max(right - float(plot_rect.right()), 0.0)
                + max(float(plot_rect.top()) - top, 0.0)
                + max(bottom - float(plot_rect.bottom()), 0.0)
            )
            legend_overlap = 0
            if legend_rect is not None:
                legend_overlap = not (
                    right < float(legend_rect.left())
                    or left > float(legend_rect.right())
                    or bottom < float(legend_rect.top())
                    or top > float(legend_rect.bottom())
                )

            nearest_x = min(max(float(point_scene.x()), left), right)
            nearest_y = min(max(float(point_scene.y()), top), bottom)
            distance = (
                (nearest_x - float(point_scene.x())) ** 2
                + (nearest_y - float(point_scene.y())) ** 2
            )

            score = (overflow > 0, overflow, covered, legend_overlap, distance, preference)
            if best is None or score < best[0]:
                best = (score, anchor, pos_x, pos_y)

        _, anchor, pos_x, pos_y = best
        card_position = self.pg_view.mapSceneToView(QPointF(pos_x, pos_y))
        self.hover_card.setAnchor(anchor)
        self.hover_card.setPos(card_position)
        self.hover_card.show()

    def _hide_hover_card(self) -> None:
        card = getattr(self, 'hover_card', None)
        if card is not None:
            card.hide()


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
                self._hide_hover_card()
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
                self._hide_hover_card()
                self._last_hover_idx = None
            return

        x, y, local_pos = mapped
        idx = self._nearest_point_index(local_pos, x, y)
        if idx is not None:
            if idx == self._last_hover_idx:
                return
            self._last_hover_idx = idx
            self._show_hover_card(idx)
        else:
            if self._last_hover_idx is not None:
                self._hide_hover_card()
                self._last_hover_idx = None
