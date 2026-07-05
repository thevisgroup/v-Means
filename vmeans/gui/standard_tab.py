
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
from .constants import _default_points_for_dataset
from .models import AnalysisResult
from .data_preview import DataPreviewDialog, clean_data, normalize_column_name, adjust_combo_popup_width
from .canvas import FixedCanvas

class StandardAnalysisTab(QWidget):
    """Standard Analysis Tab - matches web version functionality"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = None
        self.source_label = ""
        self.analysis_result = None
        self.uploaded_df = None
        self.show_overlay = False

        self.setup_ui()

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

        # File upload
        file_layout = QHBoxLayout()
        self.file_btn = QPushButton("Upload CSV/TXT/Excel")
        self.file_btn.setMinimumWidth(140)
        self.file_btn.clicked.connect(self.load_file)
        self.file_btn.setStyleSheet("""
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
        file_layout.addWidget(self.file_btn)
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.file_label, 1)
        data_layout.addLayout(file_layout)

        # Column selection (hidden initially)
        self.column_widget = QWidget()
        column_layout = QVBoxLayout(self.column_widget)
        column_layout.setContentsMargins(0, 5, 0, 0)
        column_layout.setSpacing(3)

        # X Column (vertical layout - label above combobox)
        column_layout.addWidget(QLabel("X Column:"))
        self.x_col_combo = QComboBox()
        self.x_col_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        column_layout.addWidget(self.x_col_combo)

        # Y Column
        column_layout.addWidget(QLabel("Y Column:"))
        self.y_col_combo = QComboBox()
        self.y_col_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        column_layout.addWidget(self.y_col_combo)

        # Preview button
        self.preview_btn = QPushButton("🔍 Preview & Statistics")
        self.preview_btn.clicked.connect(self.show_data_preview)
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 6px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        column_layout.addWidget(self.preview_btn)

        self.column_widget.hide()
        data_layout.addWidget(self.column_widget)

        # --- Data Cleaning Group ---
        self.cleaning_widget = QWidget()
        cleaning_layout = QVBoxLayout(self.cleaning_widget)
        cleaning_layout.setContentsMargins(0, 10, 0, 0)

        cleaning_label = QLabel("🧹 Data Cleaning Options")
        cleaning_label.setStyleSheet("font-weight: bold; color: #555;")
        cleaning_layout.addWidget(cleaning_label)

        # Remove outliers checkbox
        self.remove_outliers_cb = QCheckBox("Remove Outliers")
        self.remove_outliers_cb.setChecked(False)
        self.remove_outliers_cb.stateChanged.connect(self.on_outlier_toggle)
        cleaning_layout.addWidget(self.remove_outliers_cb)

        # Outlier threshold slider
        self.outlier_widget = QWidget()
        outlier_layout = QHBoxLayout(self.outlier_widget)
        outlier_layout.setContentsMargins(20, 0, 0, 0)
        outlier_layout.addWidget(QLabel("Threshold (σ):"))
        self.outlier_spin = QDoubleSpinBox()
        self.outlier_spin.setRange(2.0, 5.0)
        self.outlier_spin.setSingleStep(0.5)
        self.outlier_spin.setValue(3.0)
        self.outlier_spin.setDecimals(1)
        outlier_layout.addWidget(self.outlier_spin)
        outlier_layout.addStretch()
        self.outlier_widget.hide()
        cleaning_layout.addWidget(self.outlier_widget)

        # Standardize checkbox
        self.standardize_cb = QCheckBox("Standardize Data (z-score)")
        self.standardize_cb.setChecked(True)  # Default ON
        cleaning_layout.addWidget(self.standardize_cb)

        # Apply button
        self.apply_cols_btn = QPushButton("✓ Apply & Load Data")
        self.apply_cols_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.apply_cols_btn.clicked.connect(self.apply_columns)
        cleaning_layout.addWidget(self.apply_cols_btn)

        self.cleaning_widget.hide()
        data_layout.addWidget(self.cleaning_widget)

        # Separator
        sep_label = QLabel("─" * 30)
        sep_label.setStyleSheet("color: lightgray;")
        sep_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(sep_label)

        # Generated data
        data_layout.addWidget(QLabel("Or select generated structure:"))
        self.structure_combo = QComboBox()
        self.structure_combo.addItems([
            "blobs", "quadrants", "varied_blobs", "flower", "anisotropic_blobs",
            "cross", "ring", "concentric_circles", "moons", "spiral", "aggregation","zahn_compound"
        ])
        self.structure_combo.setMinimumWidth(200)
        data_layout.addWidget(self.structure_combo)

        left_layout.addWidget(data_group)

        # --- Algorithm Settings Group ---
        algo_group = QGroupBox("⚙️ Algorithm Settings")
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.setContentsMargins(10, 5, 10, 10)

        # Center method
        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Center Method:"))
        self.center_combo = QComboBox()
        self.center_combo.addItems(["centroid", "auto", "geometric"])
        self.center_combo.setMinimumWidth(120)
        center_layout.addWidget(self.center_combo)
        center_layout.addStretch()
        algo_layout.addLayout(center_layout)

        # Segments
        seg_layout = QHBoxLayout()
        seg_layout.addWidget(QLabel("Segments:"))
        self.segments_spin = QSpinBox()
        self.segments_spin.setRange(8, 180)
        self.segments_spin.setValue(60)
        seg_layout.addWidget(self.segments_spin)
        seg_layout.addStretch()
        algo_layout.addLayout(seg_layout)

        # Gradient threshold
        grad_layout = QHBoxLayout()
        grad_layout.addWidget(QLabel("Gradient Threshold:"))
        self.gradient_spin = QDoubleSpinBox()
        self.gradient_spin.setRange(0.05, 0.8)  # same range as Step Animation
        self.gradient_spin.setSingleStep(0.05)
        self.gradient_spin.setDecimals(2)
        self.gradient_spin.setValue(0.25)
        grad_layout.addWidget(self.gradient_spin)
        grad_layout.addStretch()
        algo_layout.addLayout(grad_layout)

        left_layout.addWidget(algo_group)

        # --- Visualization Options Group ---
        vis_group = QGroupBox("🎨 Visualization Options")
        vis_layout = QVBoxLayout(vis_group)
        vis_layout.setContentsMargins(10, 5, 10, 10)

        self.show_segments_cb = QCheckBox("Show Segment Bars")
        self.show_segments_cb.setChecked(True)
        vis_layout.addWidget(self.show_segments_cb)

        self.show_gradient_cb = QCheckBox("Show Gradient Lines")
        self.show_gradient_cb.setChecked(True)
        vis_layout.addWidget(self.show_gradient_cb)

        self.show_edges_cb = QCheckBox("Show d_min/d_max Edges")
        self.show_edges_cb.setChecked(True)
        vis_layout.addWidget(self.show_edges_cb)

        self.show_scatter_cb = QCheckBox("Show Distance Scatter")
        self.show_scatter_cb.setChecked(True)
        vis_layout.addWidget(self.show_scatter_cb)

        self.show_cartesian_cb = QCheckBox("Show Cartesian Analysis")
        self.show_cartesian_cb.setChecked(True)
        vis_layout.addWidget(self.show_cartesian_cb)

        left_layout.addWidget(vis_group)

        # --- Action Buttons ---
        btn_group = QGroupBox("🚀 Actions")
        btn_layout = QVBoxLayout(btn_group)
        btn_layout.setContentsMargins(10, 5, 10, 10)

        self.run_btn = QPushButton("▶️ Run Standard Analysis")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_btn.clicked.connect(lambda: self.run_analysis(recursive=False))
        btn_layout.addWidget(self.run_btn)

        self.recursive_btn = QPushButton("🔄 Run Recursive Analysis")
        self.recursive_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.recursive_btn.clicked.connect(lambda: self.run_analysis(recursive=True))
        btn_layout.addWidget(self.recursive_btn)

        self.overlay_btn = QPushButton("✨ Preview Overlay")
        self.overlay_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.overlay_btn.clicked.connect(self.toggle_overlay)
        self.overlay_btn.setEnabled(False)
        btn_layout.addWidget(self.overlay_btn)

        left_layout.addWidget(btn_group)
        left_layout.addStretch()

        # === Right Panel - Visualization ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # Status label
        self.status_label = QLabel("Ready. Configure parameters and click 'Run Analysis'")
        self.status_label.setStyleSheet("""
            color: #666; 
            font-style: italic; 
            padding: 8px;
            background-color: #f5f5f5;
            border-radius: 4px;
        """)
        right_layout.addWidget(self.status_label)

        # Scroll area for plots
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #fafafa;
            }
        """)

        self.plot_container = QWidget()
        self.plot_layout = QGridLayout(self.plot_container)
        self.plot_layout.setSpacing(20)
        self.plot_layout.setContentsMargins(15, 15, 15, 15)
        scroll.setWidget(self.plot_container)

        right_layout.addWidget(scroll, 1)

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

    def on_outlier_toggle(self, state):
        """Show/hide outlier threshold slider"""
        self.outlier_widget.setVisible(state == Qt.CheckState.Checked.value)

    def show_data_preview(self):
        """Show data preview dialog with statistics"""
        if self.uploaded_df is not None:
            dialog = DataPreviewDialog(self.uploaded_df, self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Warning", "No data loaded")

    def load_file(self):
        """Load CSV, TXT, or Excel file (supports SIPU benchmark format)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "Data Files (*.csv *.txt *.xlsx *.xls);;CSV Files (*.csv);;TXT Files (*.txt);;Excel Files (*.xlsx *.xls)"
        )

        if file_path:
            try:
                if file_path.endswith('.txt'):
                    # SIPU benchmark format: tab/whitespace-separated, no header
                    try:
                        # Try tab-separated first
                        self.uploaded_df = pd.read_csv(
                            file_path,
                            sep='\t',
                            header=None,
                            engine='python'
                        )
                    except:
                        # Fallback: any whitespace separator
                        self.uploaded_df = pd.read_csv(
                            file_path,
                            sep=r'\s+',
                            header=None,
                            engine='python'
                        )

                    # Auto-generate column names based on number of columns
                    n_cols = len(self.uploaded_df.columns)
                    if n_cols == 2:
                        self.uploaded_df.columns = ['X', 'Y']
                    elif n_cols == 3:
                        self.uploaded_df.columns = ['X', 'Y', 'Label']
                    elif n_cols == 4:
                        self.uploaded_df.columns = ['X', 'Y', 'Z', 'Label']
                    else:
                        self.uploaded_df.columns = [f'Col{i}' for i in range(n_cols)]

                elif file_path.endswith('.csv'):
                    # Try different encodings
                    for encoding in ['utf-8', 'gbk', 'latin1', 'iso-8859-1']:
                        try:
                            self.uploaded_df = pd.read_csv(file_path, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise ValueError("Unable to decode CSV file")
                else:
                    self.uploaded_df = pd.read_excel(file_path)

                self.file_label.setText(os.path.basename(file_path))
                self.file_label.setStyleSheet("color: green;")

                # Populate column combos
                numeric_cols = self.uploaded_df.select_dtypes(include=[np.number]).columns.tolist()

                # Clean column names (remove newlines for proper display)
                clean_cols = [normalize_column_name(col) for col in numeric_cols]

                # Store mapping: clean name -> original name
                self.col_name_map = dict(zip(clean_cols, numeric_cols))

                self.x_col_combo.clear()
                self.y_col_combo.clear()
                self.x_col_combo.addItems(clean_cols)
                self.y_col_combo.addItems(clean_cols)

                if len(clean_cols) >= 2:
                    self.y_col_combo.setCurrentIndex(1)

                # Auto-adjust popup width to fit longest column name
                adjust_combo_popup_width(self.x_col_combo)
                adjust_combo_popup_width(self.y_col_combo)

                self.column_widget.show()
                self.cleaning_widget.show()

                self.status_label.setText(
                    f"Loaded {len(self.uploaded_df)} rows, {len(numeric_cols)} numeric columns. "
                    "Select columns and click 'Apply'"
                )
                self.status_label.setStyleSheet(
                    "color: #1976D2; font-style: italic; padding: 8px; "
                    "background-color: #E3F2FD; border-radius: 4px;"
                )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def apply_columns(self):
        """Apply selected columns with data cleaning"""
        if self.uploaded_df is None:
            return

        x_col_display = self.x_col_combo.currentText()
        y_col_display = self.y_col_combo.currentText()

        if not x_col_display or not y_col_display:
            QMessageBox.warning(self, "Warning", "Please select both X and Y columns")
            return

        # Get original column names (may contain newlines)
        x_col = self.col_name_map.get(x_col_display, x_col_display)
        y_col = self.col_name_map.get(y_col_display, y_col_display)

        # Apply data cleaning
        points, cleaned_df, messages = clean_data(
            self.uploaded_df,
            x_col, y_col,
            remove_outliers=self.remove_outliers_cb.isChecked(),
            outlier_std=self.outlier_spin.value(),
            standardize=self.standardize_cb.isChecked()
        )

        if points is not None:
            self.points = points
            self.source_label = f"{x_col_display} vs {y_col_display}"

            # Show all messages
            message_text = " | ".join(messages)
            self.status_label.setText(message_text)
            self.status_label.setStyleSheet(
                "color: green; font-style: italic; padding: 8px; "
                "background-color: #e8f5e9; border-radius: 4px;"
            )
        else:
            QMessageBox.critical(self, "Error", "\n".join(messages))

    def get_plot_options(self) -> PlotOptions:
        """Get current plot options"""
        return PlotOptions(
            show_dmin_dmax_comparison=True,
            draw_dmin_dmax_edges=self.show_edges_cb.isChecked(),
            draw_dmax_dot_plot=True,
            draw_segment_bars=self.show_segments_cb.isChecked(),
            show_gradient_lines=self.show_gradient_cb.isChecked(),
            gradient_threshold_ratio=self.gradient_spin.value(),
            show_distance_angle_scatter=self.show_scatter_cb.isChecked(),
            show_cartesian_analysis=self.show_cartesian_cb.isChecked()
        )

    def run_analysis(self, recursive: bool = False):
        """Run the analysis"""
        # Get data points
        if self.points is None or len(self.points) == 0:
            structure = self.structure_combo.currentText()
            self.points = generate_structured_points(
                structure, _default_points_for_dataset(structure)
            )
            self.source_label = f"Generated: {structure}"

        self.status_label.setText(f"Running {'recursive' if recursive else 'standard'} analysis...")
        self.status_label.setStyleSheet("color: blue; font-style: italic; padding: 8px; background-color: #e3f2fd; border-radius: 4px;")
        QApplication.processEvents()

        try:
            self.clear_plots()

            segments = self.segments_spin.value()
            center_method = self.center_combo.currentText()
            plot_options = self.get_plot_options()

            # Core analysis
            centroid = compute_better_center(self.points, method=center_method)
            r, theta = convert_to_polar(self.points, centroid)
            max_r = np.max(r) if len(r) > 0 else 1.0

            segment_points, theta_edges = segment_points_by_theta(r, theta, s=segments)
            dmax_list = [max(seg) if seg else 0.0 for seg in segment_points]
            dmin_list = [min(seg) if seg else 0.0 for seg in segment_points]

            # Gradient detection
            gradient_boundaries = []
            if plot_options.show_gradient_lines:
                gradient_boundaries = detect_gradient_boundaries(
                    dmax_list, plot_options.gradient_threshold_ratio, global_max=max_r
                )

            # Dynamic center detection
            additional_centers, gradient_angles, center_labels = [], [], []
            if gradient_boundaries:
                additional_centers, gradient_angles, center_labels = detect_dynamic_centers(
                    self.points, r, theta, gradient_boundaries, theta_edges, segment_points
                )

            # Store result
            self.analysis_result = AnalysisResult(
                points=self.points,
                centroid=centroid,
                r=r,
                theta=theta,
                segment_points=segment_points,
                theta_edges=theta_edges,
                dmax_list=dmax_list,
                dmin_list=dmin_list,
                gradient_boundaries=gradient_boundaries,
                additional_centers=additional_centers,
                gradient_angles=gradient_angles,
                center_labels=center_labels
            )

            # Recursive analysis
            if recursive:
                self.analysis_result.sub_analyses = []
                if additional_centers and gradient_angles:
                    regions = segment_points_by_region(self.points, theta, gradient_angles, additional_centers)

                    for i, (center, label) in enumerate(zip(additional_centers, center_labels)):
                        if i < len(regions) and len(regions[i]) > 10:
                            sub_points = self.points[regions[i]]
                            sub_centroid = compute_better_center(sub_points, method='centroid')
                            sub_r, sub_theta = convert_to_polar(sub_points, sub_centroid)
                            sub_segments = segments // 2
                            sub_seg_points, sub_theta_edges = segment_points_by_theta(sub_r, sub_theta, s=sub_segments)
                            sub_dmax = [max(seg) if seg else 0.0 for seg in sub_seg_points]

                            sub_gradient_boundaries = detect_gradient_boundaries(
                                sub_dmax, plot_options.gradient_threshold_ratio,
                                global_max=max_r  # use parent max_r for consistency with Step Animation
                            )

                            self.analysis_result.sub_analyses.append({
                                'label': label,
                                'points': sub_points,
                                'centroid': sub_centroid,
                                'r': sub_r,
                                'theta': sub_theta,
                                'segment_points': sub_seg_points,
                                'theta_edges': sub_theta_edges,
                                'dmax_list': sub_dmax,
                                'gradient_boundaries': sub_gradient_boundaries
                            })

            # Create visualizations
            self.create_plots(recursive, plot_options)

            self.overlay_btn.setEnabled(recursive and len(self.analysis_result.sub_analyses or []) > 0)
            self.status_label.setText(f"Analysis complete: {len(self.points)} points, {segments} segments")
            self.status_label.setStyleSheet("color: green; font-style: italic; padding: 8px; background-color: #e8f5e9; border-radius: 4px;")

        except Exception as e:
            import traceback
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-style: italic; padding: 8px; background-color: #ffebee; border-radius: 4px;")
            print(traceback.format_exc())

    def clear_plots(self):
        """Clear all plots from the layout"""
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item.widget():
                widget = item.widget()
                widget.setParent(None)
                widget.deleteLater()

    def create_plots(self, recursive: bool, plot_options: PlotOptions):
        """Create analysis plots"""
        from vmeans.plot import (
            draw_segment_structure_with_dynamic_centers,
            draw_dmin_dmax_edges,
            draw_cartesian_x_distance_scatter,
            draw_cartesian_x_based_analysis,
            draw_segment_distance_dot_plot,
            draw_recursive_overlay
        )

        result = self.analysis_result

        # Row 0: Main polar plot + Distance scatter
        if self.show_overlay and recursive and result.sub_analyses:
            fig1 = draw_recursive_overlay({
                'points': result.points,
                'centroid': result.centroid,
                'sub_analyses': result.sub_analyses
            }, self.source_label)
        else:
            fig1 = draw_segment_structure_with_dynamic_centers(
                result.r, result.theta, result.segment_points, result.theta_edges,
                np.max(result.r), result.centroid,
                points=result.points,
                title=f"Polar: {self.source_label}",
                show_segments=plot_options.draw_segment_bars,
                gradient_boundaries=result.gradient_boundaries,
                detect_centers_func=detect_dynamic_centers
            )

        canvas1 = FixedCanvas(fig1)
        self.plot_layout.addWidget(canvas1, 0, 0)

        if plot_options.show_distance_angle_scatter:
            fig2 = draw_cartesian_x_distance_scatter(
                result.points, title=f"Distance Scatter"
            )
            canvas2 = FixedCanvas(fig2)
            self.plot_layout.addWidget(canvas2, 0, 1)

        # Row 1: d_min/d_max edges + Cartesian analysis
        col = 0
        if plot_options.draw_dmin_dmax_edges:
            fig3 = draw_dmin_dmax_edges(
                result.theta_edges, result.dmin_list, result.dmax_list,
                title="d_min/d_max Edges"
            )
            canvas3 = FixedCanvas(fig3)
            self.plot_layout.addWidget(canvas3, 1, col)
            col += 1

        if plot_options.show_cartesian_analysis:
            fig4 = draw_cartesian_x_based_analysis(
                result.points, segments=self.segments_spin.value(),
                gradient_threshold_ratio=plot_options.gradient_threshold_ratio if plot_options.show_gradient_lines else None,
                title=f"Cartesian X-based"
            )
            canvas4 = FixedCanvas(fig4)
            self.plot_layout.addWidget(canvas4, 1, col)
            col += 1

        # Row 2: Segment distance dot plot
        dot_result = draw_segment_distance_dot_plot(result.segment_points, result.theta_edges)
        if dot_result:
            fig5, d_max_val, d_min_val, empty_count = dot_result
            canvas5 = FixedCanvas(fig5)
            self.plot_layout.addWidget(canvas5, 2, 0, 1, 2)

        # Row 3+: Recursive subplots
        if recursive and result.sub_analyses and not self.show_overlay:
            header = QLabel("🎯 Recursive Analysis (Independent Regions)")
            header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
            self.plot_layout.addWidget(header, 3, 0, 1, 2)

            row = 4
            col = 0
            for sub in result.sub_analyses:
                fig_sub = draw_segment_structure_with_dynamic_centers(
                    sub['r'], sub['theta'], sub['segment_points'], sub['theta_edges'],
                    np.max(sub['r']) if len(sub['r']) > 0 else 1.0, sub['centroid'],
                    points=sub['points'],
                    title=f"Cluster {sub['label']}",
                    gradient_boundaries=sub.get('gradient_boundaries', []),
                    detect_centers_func=detect_dynamic_centers
                )
                canvas_sub = FixedCanvas(fig_sub)
                self.plot_layout.addWidget(canvas_sub, row, col)
                col += 1
                if col > 1:
                    col = 0
                    row += 1

    def toggle_overlay(self):
        """Toggle overlay view"""
        self.show_overlay = not self.show_overlay
        if self.analysis_result:
            self.clear_plots()
            self.create_plots(recursive=True, plot_options=self.get_plot_options())
