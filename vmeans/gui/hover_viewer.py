
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

        if self.has_original_coords:
            x_col, y_col = self.original_xy_cols
            self.plot_x = self.original_df[x_col].values.astype(float)
            self.plot_y = self.original_df[y_col].values.astype(float)
            self.x_label = str(x_col)
            self.y_label = str(y_col)
        else:
            self.plot_x = self.points[:, 0]
            self.plot_y = self.points[:, 1]
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

        pg.setConfigOptions(antialias=False)
        self.pg_plot = pg.PlotWidget(background='w')
        self.pg_plot.setMouseTracking(True)
        self.pg_plot.viewport().setMouseTracking(True)
        self.pg_plot.viewport().installEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True
        self.pg_plot.showGrid(x=True, y=True, alpha=0.22)
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
