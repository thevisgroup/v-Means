
import sys
import os
import tempfile
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

_MPL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "viskmean_matplotlib_cache")
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox, QSlider, QCheckBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea, QSplitter,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QProgressBar, QFrame, QSizePolicy, QGridLayout, QStackedWidget,
    QHeaderView, QSpacerItem, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QFont, QPalette, QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from vmeans.core_analysis import (
    compute_better_center, detect_gradient_boundaries,
    detect_dynamic_centers, segment_points_by_region
)
from vmeans.data import generate_structured_points, convert_to_polar
from vmeans.segment import segment_points_by_theta
from vmeans.interface import PlotOptions
from vmeans.animation import build_enhanced_visible_frames, StepFrame

from .constants import *
from .models import AnalysisResult
from .data_preview import DataPreviewDialog, clean_data, normalize_column_name, adjust_combo_popup_width
from .canvas import FixedCanvas
from vmeans.gui.hover_viewer import HoverScatterDialog

USE_COLORED_SILHOUETTES = True
if USE_COLORED_SILHOUETTES:
    from vmeans.rendering.colored import draw_frame_matplotlib, get_frame_description, _reset_global_ranges, RenderOptions
else:
    from vmeans.rendering.base import draw_frame_matplotlib, get_frame_description, _reset_global_ranges, RenderOptions

from .step_animation_ui import StepAnimationUiMixin
from .step_animation_build import StepAnimationBuildMixin
from .step_animation_render import StepAnimationRenderMixin

class StepAnimationTab(StepAnimationUiMixin, StepAnimationBuildMixin, StepAnimationRenderMixin, QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames: List[StepFrame] = []
        # Removed self.cached_figures - no longer pre-caching
        self.current_step = 0
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.advance_frame)
        # Removed self.preload_thread - no longer using background thread
        self.uploaded_df = None
        self.data_points = None
        self.original_rows_df = None  # Original row values for hover viewer
        self.original_xy_cols = None  # Original (x_col, y_col) names
        self.hover_dataset_label = "Unknown dataset/source"
        self.hover_analysis_settings = {}

        # New: Global range control (ensures consistent animation scaling)
        self.global_rmax = None
        self.global_centroid = None

        # New: Store canvas layout reference for hard replacement
        self.canvas_layout = None

        # Only center the viewport on first show or when size changes, not every frame
        self._last_canvas_size = None
        self._need_center_viewport = True

        self.setup_ui()
