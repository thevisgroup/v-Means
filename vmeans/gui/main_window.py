
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

from .standard_tab import StandardAnalysisTab
from .step_animation_tab import StepAnimationTab

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 Visible Silhouettes Algorithm")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Title
        title_label = QLabel("🔍 Visible Silhouettes Algorithm")
        title_label.setStyleSheet("""
            font-size: 26px;
            font-weight: bold;
            padding: 12px;
            color: #333;
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                border-radius: 4px;
                background: white;
            }
            QTabBar::tab {
                padding: 12px 25px;
                font-size: 14px;
                font-weight: bold;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:!selected {
                background-color: #e0e0e0;
            }
            QTabBar::tab:hover:!selected {
                background-color: #c0c0c0;
            }
        """)

        # Add tabs
        self.standard_tab = StandardAnalysisTab()
        self.animation_tab = StepAnimationTab()

        self.tab_widget.addTab(self.standard_tab, "📊 Standard Analysis")
        self.tab_widget.addTab(self.animation_tab, "🎬 Step Animation")

        main_layout.addWidget(self.tab_widget, 1)

        # Status bar
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("color: #666;")

    def keyPressEvent(self, event):
        """Forward keyboard events to the currently active tab."""
        current_tab = self.tab_widget.currentWidget()
        if current_tab is self.animation_tab:
            # Let StepAnimationTab handle it
            self.animation_tab.keyPressEvent(event)
            if event.isAccepted():
                return
        super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Set palette for modern look
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(250, 250, 250))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(33, 33, 33))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(33, 33, 33))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 175, 80))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(33, 33, 33))
    app.setPalette(palette)

    # Global stylesheet
    app.setStyleSheet("""
        QSpinBox, QDoubleSpinBox {
            padding: 5px 10px;
            font-size: 13px;
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 3px;
            color: #333;
            min-width: 80px;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            width: 20px;
        }
        QComboBox {
            padding: 6px 10px;
            font-size: 13px;
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 3px;
            color: #333;
            min-height: 20px;
        }
        QComboBox QAbstractItemView {
            background-color: white;
            selection-background-color: #e8f5e9;
            selection-color: black;
            border: 1px solid #ddd;
            outline: none;
        }
        QComboBox QAbstractItemView::item {
            min-height: 35px;
            padding: 5px 15px;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 13px;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 12px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            background-color: #fafafa;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
