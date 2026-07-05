
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
from .constants import CANVAS_MIN_SIZE

class FixedCanvas(FigureCanvas):
    """Fixed-size Matplotlib canvas widget"""

    def __init__(self, fig: Figure, parent=None):
        super().__init__(fig)
        self.setParent(parent)
        self.setMinimumSize(CANVAS_MIN_SIZE, CANVAS_MIN_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
