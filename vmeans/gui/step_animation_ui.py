
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

class StepAnimationUiMixin:
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # === Left Panel - Controls ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        # --- Data Source Group ---
        data_group = QGroupBox("📁 Data Source")
        data_layout = QVBoxLayout(data_group)
        data_layout.setContentsMargins(10, 5, 10, 10)

        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Generated Data", "Upload File"])
        self.data_source_combo.currentIndexChanged.connect(self.on_data_source_changed)
        data_layout.addWidget(self.data_source_combo)

        # Generated data options
        self.gen_widget = QWidget()
        gen_layout = QVBoxLayout(self.gen_widget)
        gen_layout.setContentsMargins(0, 5, 0, 0)

        self.anim_structure_combo = QComboBox()
        self.anim_structure_combo.addItems([
            "blobs", "quadrants", "varied_blobs", "flower", "spiral",
            "cross", "ring", "concentric_circles", "anisotropic_blobs", "moons", "aggregation","zahn_compound"
        ])
        gen_layout.addWidget(self.anim_structure_combo)

        data_layout.addWidget(self.gen_widget)

        # Upload file options (hidden initially)
        self.upload_widget = QWidget()
        upload_layout = QVBoxLayout(self.upload_widget)
        upload_layout.setContentsMargins(0, 5, 0, 0)

        self.anim_file_btn = QPushButton("Select File")
        self.anim_file_btn.clicked.connect(self.load_animation_file)
        self.anim_file_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 8px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        upload_layout.addWidget(self.anim_file_btn)

        self.anim_file_label = QLabel("No file selected")
        self.anim_file_label.setStyleSheet("color: gray;")
        upload_layout.addWidget(self.anim_file_label)

        # Preview button
        self.anim_preview_btn = QPushButton("🔍 Preview & Statistics")
        self.anim_preview_btn.clicked.connect(self.show_anim_data_preview)
        self.anim_preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.anim_preview_btn.hide()
        upload_layout.addWidget(self.anim_preview_btn)

        self.anim_col_widget = QWidget()
        anim_col_layout = QVBoxLayout(self.anim_col_widget)
        anim_col_layout.setContentsMargins(0, 5, 0, 0)
        anim_col_layout.setSpacing(3)

        # X Column (vertical layout - label above combobox)
        anim_col_layout.addWidget(QLabel("X Column:"))
        self.anim_x_combo = QComboBox()
        self.anim_x_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        anim_col_layout.addWidget(self.anim_x_combo)

        # Y Column
        anim_col_layout.addWidget(QLabel("Y Column:"))
        self.anim_y_combo = QComboBox()
        self.anim_y_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        anim_col_layout.addWidget(self.anim_y_combo)

        self.anim_col_widget.hide()
        upload_layout.addWidget(self.anim_col_widget)

        # Data Cleaning Options for Animation
        self.anim_cleaning_widget = QWidget()
        anim_cleaning_layout = QVBoxLayout(self.anim_cleaning_widget)
        anim_cleaning_layout.setContentsMargins(0, 5, 0, 0)

        self.anim_remove_outliers_cb = QCheckBox("Remove Outliers (3σ)")
        self.anim_remove_outliers_cb.setChecked(False)
        anim_cleaning_layout.addWidget(self.anim_remove_outliers_cb)

        self.anim_standardize_cb = QCheckBox("Standardize Data")
        self.anim_standardize_cb.setChecked(True)  # Default ON
        anim_cleaning_layout.addWidget(self.anim_standardize_cb)

        self.anim_cleaning_widget.hide()
        upload_layout.addWidget(self.anim_cleaning_widget)

        self.upload_widget.hide()
        data_layout.addWidget(self.upload_widget)

        left_layout.addWidget(data_group)

        # --- Algorithm Settings Group ---
        algo_group = QGroupBox("⚙️ Algorithm Settings")
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.setContentsMargins(10, 5, 10, 10)

        seg_layout = QHBoxLayout()
        seg_layout.addWidget(QLabel("Segments:"))
        self.anim_segments_spin = QSpinBox()
        self.anim_segments_spin.setRange(8, 180)
        self.anim_segments_spin.setValue(60)
        seg_layout.addWidget(self.anim_segments_spin)
        seg_layout.addStretch()
        algo_layout.addLayout(seg_layout)

        grad_layout = QHBoxLayout()
        grad_layout.addWidget(QLabel("Gradient:"))
        self.anim_gradient_spin = QDoubleSpinBox()
        self.anim_gradient_spin.setRange(0.05, 0.8)
        self.anim_gradient_spin.setSingleStep(0.05)
        self.anim_gradient_spin.setDecimals(2)
        self.anim_gradient_spin.setValue(0.25)
        grad_layout.addWidget(self.anim_gradient_spin)
        grad_layout.addStretch()
        algo_layout.addLayout(grad_layout)

        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Center:"))
        self.anim_center_combo = QComboBox()
        self.anim_center_combo.addItems(["centroid", "auto", "geometric"])
        self.anim_center_combo.setMinimumWidth(100)
        center_layout.addWidget(self.anim_center_combo)
        center_layout.addStretch()
        algo_layout.addLayout(center_layout)

        # Recursion control: depth 1 = child, depth 2 = child + grandchild.
        recursion_layout = QHBoxLayout()
        self.anim_recursion_cb = QCheckBox("Enable Recursion")
        self.anim_recursion_cb.setChecked(True)
        recursion_layout.addWidget(self.anim_recursion_cb)
        recursion_layout.addWidget(QLabel("Depth:"))
        self.anim_recursion_depth_spin = QSpinBox()
        self.anim_recursion_depth_spin.setRange(1, 2)
        self.anim_recursion_depth_spin.setValue(1)
        self.anim_recursion_depth_spin.setToolTip(
            "1 = child level, 2 = child and grandchild levels"
        )
        recursion_layout.addWidget(self.anim_recursion_depth_spin)
        recursion_layout.addStretch()
        self.anim_recursion_cb.toggled.connect(
            self.anim_recursion_depth_spin.setEnabled
        )
        algo_layout.addLayout(recursion_layout)

        left_layout.addWidget(algo_group)

        # --- Playback Controls Group ---
        playback_group = QGroupBox("🎬 Playback")
        playback_layout = QVBoxLayout(playback_group)
        playback_layout.setContentsMargins(10, 5, 10, 10)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed (ms):"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(100, 2000)
        self.speed_slider.setValue(500)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(200)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("500")
        self.speed_label.setMinimumWidth(40)
        speed_layout.addWidget(self.speed_label)
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(str(v)))
        playback_layout.addLayout(speed_layout)

        left_layout.addWidget(playback_group)

        # --- Display Settings Group (NEW) ---
        display_group = QGroupBox("🎨 Display Settings")
        display_layout = QVBoxLayout(display_group)
        display_layout.setContentsMargins(10, 5, 10, 10)

        # Point Size slider - default smaller (30%)
        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(QLabel("Point Size:"))
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setRange(5, 150)  # 5% to 150%
        self.point_size_slider.setValue(30)  # 30% = smaller default
        self.point_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.point_size_slider.setTickInterval(20)
        point_size_layout.addWidget(self.point_size_slider)
        self.point_size_label = QLabel("30%")
        self.point_size_label.setMinimumWidth(45)
        point_size_layout.addWidget(self.point_size_label)
        self.point_size_slider.valueChanged.connect(
            lambda v: self.point_size_label.setText(f"{v}%"))
        display_layout.addLayout(point_size_layout)

        # Silhouette Thickness slider
        silhouette_layout = QHBoxLayout()
        silhouette_layout.addWidget(QLabel("Silhouette:"))
        self.silhouette_slider = QSlider(Qt.Orientation.Horizontal)
        self.silhouette_slider.setRange(50, 300)  # 0.5x to 3.0x
        self.silhouette_slider.setValue(200)  # 2.0x = more emphasized
        self.silhouette_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.silhouette_slider.setTickInterval(50)
        silhouette_layout.addWidget(self.silhouette_slider)
        self.silhouette_label = QLabel("2.0x")
        self.silhouette_label.setMinimumWidth(45)
        silhouette_layout.addWidget(self.silhouette_label)
        self.silhouette_slider.valueChanged.connect(
            lambda v: self.silhouette_label.setText(f"{v/100:.1f}x"))
        display_layout.addLayout(silhouette_layout)

        # Chart Size slider - directly controls chart size
        chart_size_layout = QHBoxLayout()
        chart_size_layout.addWidget(QLabel("Chart Size:"))
        self.chart_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.chart_size_slider.setRange(300, 1200)  # 300px to 1200px
        self.chart_size_slider.setValue(650)  # Default 650px
        self.chart_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.chart_size_slider.setTickInterval(100)
        chart_size_layout.addWidget(self.chart_size_slider)
        self.chart_size_label = QLabel("650px")
        self.chart_size_label.setMinimumWidth(50)
        chart_size_layout.addWidget(self.chart_size_label)
        self.chart_size_slider.valueChanged.connect(self.on_chart_size_changed)
        display_layout.addLayout(chart_size_layout)

        # Zoom slider - controls view scale (smaller = zoom in on main data)
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(50, 150)  # 50% to 150%
        self.zoom_slider.setValue(96)  # Tighter fit while retaining radial headroom
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(25)
        zoom_layout.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        zoom_layout.addWidget(self.zoom_label)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        display_layout.addLayout(zoom_layout)

        # Final View Options - ONLY Show Silhouettes option
        final_view_label = QLabel("Final Frame:")
        final_view_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        display_layout.addWidget(final_view_label)

        self.show_silhouettes_cb = QCheckBox("Show Silhouettes")
        self.show_silhouettes_cb.setChecked(False)  # Default OFF - final frame shows only colored points
        self.show_silhouettes_cb.setToolTip("Show boundary outlines around clusters on final frame")
        display_layout.addWidget(self.show_silhouettes_cb)

        left_layout.addWidget(display_group)

        # --- Build Button ---
        self.build_btn = QPushButton("🔄 Build Animation")
        self.build_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.build_btn.clicked.connect(self.build_animation)
        left_layout.addWidget(self.build_btn)

        # Export Final Frame button
        self.export_btn = QPushButton("💾 Export Final Frame")
        self.export_btn.setMinimumHeight(36)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.export_btn.clicked.connect(self.export_final_frame)
        self.export_btn.setEnabled(False)  # Disabled until animation is built
        left_layout.addWidget(self.export_btn)

        # Hover Scatter Viewer button (only visible on last frame)
        self.hover_btn = QPushButton("🔍 Hover Details")
        self.hover_btn.setMinimumHeight(36)
        self.hover_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.hover_btn.clicked.connect(self.open_hover_viewer)
        self.hover_btn.hide()
        self.hover_btn.setEnabled(False)
        left_layout.addWidget(self.hover_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        left_layout.addWidget(self.progress_bar)

        # Frame info
        self.frame_info_label = QLabel("No animation loaded")
        self.frame_info_label.setStyleSheet("color: gray; font-style: italic;")
        left_layout.addWidget(self.frame_info_label)

        left_layout.addStretch()

        # === Right Panel - Animation Display ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        right_layout.setSpacing(5)  # Minimal spacing

        # Frame slider at top
        slider_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(self.frame_slider)
        self.frame_counter = QLabel("0/0")
        self.frame_counter.setMinimumWidth(60)
        self.frame_counter.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        slider_layout.addWidget(self.frame_counter)
        right_layout.addLayout(slider_layout)

        # Playback buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(5)

        self.first_btn = QPushButton("⏮️")
        self.first_btn.setMinimumHeight(32)
        self.first_btn.clicked.connect(self.go_first)
        btn_layout.addWidget(self.first_btn)

        self.prev_btn = QPushButton("⬅️")
        self.prev_btn.setMinimumHeight(32)
        self.prev_btn.clicked.connect(self.prev_frame)
        btn_layout.addWidget(self.prev_btn)

        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.setMinimumHeight(32)
        self.play_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
            }
        """)
        self.play_btn.clicked.connect(self.toggle_play)
        btn_layout.addWidget(self.play_btn)

        self.next_btn = QPushButton("➡️")
        self.next_btn.setMinimumHeight(32)
        self.next_btn.clicked.connect(self.next_frame_manual)
        btn_layout.addWidget(self.next_btn)

        self.last_btn = QPushButton("⏭️")
        self.last_btn.setMinimumHeight(32)
        self.last_btn.clicked.connect(self.go_last)
        btn_layout.addWidget(self.last_btn)

        right_layout.addLayout(btn_layout)

        # Canvas container with ScrollArea for unlimited size
        self.canvas_container = QWidget()
        self.canvas_container.setStyleSheet("background-color: white;")
        # Set initial size matching the default chart_size_slider value
        initial_size = 650
        self.canvas_container.setFixedSize(initial_size, initial_size)

        self.canvas_layout = QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        # Key point: center_viewport() must NOT run every frame or it causes jitter
        self.canvas_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Initially create empty Canvas
        self.canvas = FigureCanvas(Figure(figsize=(5, 5), dpi=FIGURE_DPI))
        self.canvas_layout.addWidget(self.canvas, 0, Qt.AlignmentFlag.AlignCenter)

        # GIF player label (hidden initially)
        self.gif_label = QLabel()
        self.gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gif_label.setStyleSheet("background-color: white;")
        self.gif_label.hide()
        self.canvas_layout.addWidget(self.gif_label, 0, Qt.AlignmentFlag.AlignCenter)

        # QMovie for GIF playback
        self.gif_movie = None
        self.gif_path = None
        self.child_gif_path = None
        self.grandchild_gif_path = None
        self.gif_playing = False
        self.gif_finished_triggered = False
        self.gif_total_frames = 0
        self.gif_frame_counter = 0
        self._resume_after_gif = False

        # Live animation state
        self._animation_playing = False
        self._animation_timer = None
        self._animation_frame = None
        self._animation_current = 0
        self._animation_total = 30
        self._resume_after_animation = False
        self._animation_completed = False

        # Wrap container in ScrollArea for unlimited canvas size
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas_container)
        # Disable auto-resize; we control the container size manually
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setStyleSheet("border: none; background-color: white;")

        right_layout.addWidget(self.scroll_area, 1)

        # Wrap left panel in scroll area for small screens
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidget(left_panel)
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.left_scroll.setMinimumWidth(340)
        self.left_scroll.setMaximumWidth(500)
        self.left_scroll.setStyleSheet("border: none;")

        # --- Collapse/Expand Toggle Button ---
        self.collapse_btn = QPushButton("◀")
        self.collapse_btn.setFixedWidth(20)
        self.collapse_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.collapse_btn.setToolTip("Collapse / Expand control panel")
        self.collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.collapse_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
                color: #666;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
                color: #333;
            }
        """)
        self.collapse_btn.clicked.connect(self.toggle_left_panel)
        self._left_panel_visible = True

        # Add panels to main layout
        main_layout.addWidget(self.left_scroll)
        main_layout.addWidget(self.collapse_btn)
        main_layout.addWidget(right_panel, 1)


    def toggle_left_panel(self):
        """Toggle left control panel visibility"""
        self._left_panel_visible = not self._left_panel_visible
        self.left_scroll.setVisible(self._left_panel_visible)
        self.collapse_btn.setText("◀" if self._left_panel_visible else "▶")
        self.collapse_btn.setToolTip(
            "Collapse control panel" if self._left_panel_visible else "Expand control panel"
        )


    def on_data_source_changed(self, index):
        """Handle data source change"""
        if index == 0:  # Generated
            self.gen_widget.show()
            self.upload_widget.hide()
        else:  # Upload
            self.gen_widget.hide()
            self.upload_widget.show()


    def show_anim_data_preview(self):
        """Show data preview for animation tab"""
        if self.uploaded_df is not None:
            dialog = DataPreviewDialog(self.uploaded_df, self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Warning", "No data loaded")
