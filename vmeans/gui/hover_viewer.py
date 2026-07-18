
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

from .hover_data import RECURSIVE_HOVER_FRAMES, _extract_region_data
from .hover_ai import HoverAIMixin, AIChatWorker
from .hover_plot import HoverPlotSelectionMixin


class OriginalValueAxis(pg.AxisItem):
    """Render analysis-space positions as values in the original data units."""

    def __init__(self, orientation: str, scale: float = 1.0, offset: float = 0.0):
        super().__init__(orientation=orientation)
        self.value_scale = float(scale)
        self.value_offset = float(offset)

    def tickStrings(self, values, scale, spacing):
        original_values = [value * self.value_scale + self.value_offset for value in values]
        magnitude = max((abs(value) for value in original_values), default=0.0)
        decimals = 0 if magnitude >= 20 else 1
        return [f"{value:.{decimals}f}" for value in original_values]


class HoverScatterDialog(HoverAIMixin, HoverPlotSelectionMixin, QDialog):
    """Non-modal dialog with hover, point selection, and AI feedback."""

    MAX_SELECTED_ROWS = 120
    MAX_RAW_CONTEXT_ROWS = 80
    MAX_CHAT_CONTEXT_TURNS = 8
    MAX_FULL_ROWS = 300
    MAX_PROMPT_CHARS = 30000

    def __init__(self, frame_name: str, payload: Dict[str, Any],
                 original_df=None, original_xy_cols=None,
                 dataset_label: str | None = None,
                 analysis_settings: Dict[str, Any] | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔍 Hover Scatter Viewer")
        self.setMinimumSize(1220, 780)
        self.resize(1320, 820)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._app_event_filter_installed = False

        self.frame_name = frame_name
        self.payload = payload or {}
        self.dataset_label = dataset_label or "Unknown dataset/source"
        self.analysis_settings = analysis_settings or {}
        self.ai_worker = None
        self.chat_turns: List[tuple[str, str]] = []
        self.selected_indices = set()
        self.focused_region_id = None
        self._mouse_press = None
        self._last_hover_idx = None
        self._last_hover_time = 0.0
        self._rubber_origin = None
        self._last_drag_local_pos = None
        self._selection_before_drag = set()

        self.data = _extract_region_data(frame_name, payload)
        self.points = self.data['points']
        self.region_ids = self.data['region_ids']
        self.colors_hex = self.data['colors_hex']
        self.region_labels = self.data['region_labels']

        if self.points is None:
            raise ValueError("Hover details require point coordinates, but this frame has none.")
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] < 2 or len(self.points) == 0:
            raise ValueError("Hover details require a non-empty Nx2 point array.")

        self.region_ids = np.asarray(self.region_ids, dtype=int)
        if len(self.region_ids) != len(self.points):
            self.region_ids = np.zeros(len(self.points), dtype=int)

        self.original_df = original_df
        self.original_xy_cols = original_xy_cols

        self.has_original_coords = (
            self.original_df is not None
            and self.original_xy_cols is not None
            and len(self.original_df) == len(self.points)
        )

        # Match the animation's geometry: it displays the analysed coordinates
        # relative to the parent centroid, represented there as theta/r.  Raw
        # Excel values remain available through original_df for hover details
        # and AI context, but no longer replace the analysed plotting space.
        centroid = self.data.get('centroid')
        if centroid is None:
            centroid = np.mean(self.points[:, :2], axis=0)
        centroid = np.asarray(centroid, dtype=float).reshape(-1)
        if centroid.size < 2 or not np.all(np.isfinite(centroid[:2])):
            centroid = np.mean(self.points[:, :2], axis=0)

        centred_points = self.points[:, :2] - centroid[:2]
        self.plot_x = centred_points[:, 0]
        self.plot_y = centred_points[:, 1]
        self.axis_scales = np.ones(2, dtype=float)
        self.axis_offsets = centroid[:2].astype(float)
        if self.original_xy_cols is not None:
            x_col, y_col = self.original_xy_cols
            self.x_label = str(x_col)
            self.y_label = str(y_col)
            if self.has_original_coords and self.analysis_settings.get('standardized'):
                raw_xy = self.original_df.loc[:, [x_col, y_col]].to_numpy(dtype=float)
                raw_means = np.mean(raw_xy, axis=0)
                raw_scales = np.std(raw_xy, axis=0, ddof=0)
                raw_scales = np.where(raw_scales > 0, raw_scales, 1.0)
                self.axis_scales = raw_scales
                self.axis_offsets = raw_means + centroid[:2] * raw_scales
        else:
            self.x_label = "X"
            self.y_label = "Y"

        self._plot_points = np.column_stack([self.plot_x, self.plot_y])
        self._hover_tree = cKDTree(self._plot_points)
        self._setup_ui()


    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        n_pts = len(self.points)
        n_reg = self.data['n_regions']
        self.info_label = QLabel(
            f"Points: {n_pts}  |  Regions: {n_reg}  |  Hover, click, or drag-select points"
        )
        self.info_label.setStyleSheet("color: #555; font-size: 12px; padding: 4px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        content = QHBoxLayout()
        content.setSpacing(10)
        layout.addLayout(content, 1)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        focus_row = QHBoxLayout()
        focus_label = QLabel("Focus cluster:")
        focus_label.setStyleSheet("font-weight: 600; color: #333;")
        self.region_focus_combo = QComboBox()
        self.region_focus_combo.setMinimumWidth(190)
        self.region_focus_combo.addItem("All regions", None)
        for rid in sorted(set(int(v) for v in self.region_ids)):
            label = (
                self.region_labels[rid]
                if 0 <= rid < len(self.region_labels)
                else f"Region {rid + 1}"
            )
            count = int(np.sum(self.region_ids == rid))
            self.region_focus_combo.addItem(f"{label} ({count} points)", rid)
        self.region_focus_combo.setToolTip(
            "Choose a region to outline all of its points; other regions remain visible."
        )
        self.region_focus_combo.currentIndexChanged.connect(self._on_region_focus_changed)
        focus_row.addStretch()
        focus_row.addWidget(focus_label)
        focus_row.addWidget(self.region_focus_combo)
        focus_row.addStretch()
        plot_layout.addLayout(focus_row)

        pg.setConfigOptions(antialias=False)
        axis_items = {
            'bottom': OriginalValueAxis(
                'bottom', scale=self.axis_scales[0], offset=self.axis_offsets[0]
            ),
            'left': OriginalValueAxis(
                'left', scale=self.axis_scales[1], offset=self.axis_offsets[1]
            ),
        }
        self.pg_plot = pg.PlotWidget(background='w', axisItems=axis_items)
        self.pg_plot.setMouseTracking(True)
        self.pg_plot.viewport().setMouseTracking(True)
        self.pg_plot.viewport().installEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True
        self.pg_plot.showGrid(x=True, y=True, alpha=0.22)
        self.pg_plot.showAxis('bottom', True)
        self.pg_plot.showAxis('left', True)
        self.pg_plot.setLabel('bottom', self.x_label)
        self.pg_plot.setLabel('left', self.y_label)
        self.pg_plot.setTitle("Final Analysis - Hover, Select, Ask AI")
        self.pg_view = self.pg_plot.getPlotItem().vb
        self.pg_view.setAspectLocked(True, ratio=1)
        plot_layout.addWidget(self.pg_plot, 1)

        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.pg_plot.viewport())
        self.rubber_band.setStyleSheet(
            "QRubberBand { border: 1px dashed #ff8c00; background: rgba(255, 140, 0, 38); }"
        )

        hint = QLabel(
            "Selection: click/drag to replace; hold Shift/Ctrl/Command to extend; modifier-click removes; blank click/right-click/Esc clears."
        )
        hint.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_layout.addWidget(hint)
        content.addWidget(plot_panel, 3)

        self.ai_panel = self._build_ai_panel()
        content.addWidget(self.ai_panel, 1)

        self._draw_scatter()
        self._update_selection_display()
