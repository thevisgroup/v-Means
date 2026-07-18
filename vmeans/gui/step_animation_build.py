
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
from vmeans.gui.hover_viewer import HoverScatterDialog

USE_COLORED_SILHOUETTES = True
if USE_COLORED_SILHOUETTES:
    from vmeans.rendering.colored import draw_frame_matplotlib, get_frame_description, _reset_global_ranges, RenderOptions
else:
    from vmeans.rendering.base import draw_frame_matplotlib, get_frame_description, _reset_global_ranges, RenderOptions

class StepAnimationBuildMixin:
    def load_animation_file(self):
        """Load file for animation (supports SIPU benchmark TXT format)"""
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

                self.anim_file_label.setText(os.path.basename(file_path))
                self.anim_file_label.setStyleSheet("color: green;")

                numeric_cols = self.uploaded_df.select_dtypes(include=[np.number]).columns.tolist()

                # Clean column names (remove newlines for proper display)
                clean_cols = [normalize_column_name(col) for col in numeric_cols]

                # Store mapping: clean name -> original name
                self.anim_col_name_map = dict(zip(clean_cols, numeric_cols))

                self.anim_x_combo.clear()
                self.anim_y_combo.clear()
                self.anim_x_combo.addItems(clean_cols)
                self.anim_y_combo.addItems(clean_cols)

                if len(clean_cols) >= 2:
                    self.anim_y_combo.setCurrentIndex(1)

                # Auto-adjust popup width to fit longest column name
                adjust_combo_popup_width(self.anim_x_combo)
                adjust_combo_popup_width(self.anim_y_combo)

                self.anim_col_widget.show()
                self.anim_preview_btn.show()
                self.anim_cleaning_widget.show()

                # Force layout refresh so ScrollArea updates properly
                from PyQt6.QtWidgets import QApplication
                QApplication.processEvents()
                self.upload_widget.adjustSize()
                # Find parent widget and refresh it too
                parent = self.upload_widget.parentWidget()
                while parent:
                    parent.adjustSize()
                    parent = parent.parentWidget()
                    if isinstance(parent, QScrollArea):
                        break

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")


    def build_animation(self):
        """Build animation frames (Data Only, No Pre-rendering)"""
        self.build_btn.setEnabled(False)
        self.progress_bar.hide()  # No progress bar needed
        self.hover_btn.hide()
        self.hover_btn.setEnabled(False)

        try:
            # Get data points
            if self.data_source_combo.currentIndex() == 0:  # Generated
                structure = self.anim_structure_combo.currentText()
                self.data_points = generate_structured_points(
                    structure, _default_points_for_dataset(structure)
                )
                self.original_rows_df = None  # No original df for generated data
                self.original_xy_cols = None
                self.hover_dataset_label = f"Generated Data / {structure}"
            else:  # Uploaded
                if self.uploaded_df is None:
                    QMessageBox.warning(self, "Warning", "Please select a file first")
                    self.build_btn.setEnabled(True)
                    return

                x_col_display = self.anim_x_combo.currentText()
                y_col_display = self.anim_y_combo.currentText()

                if not x_col_display or not y_col_display:
                    QMessageBox.warning(self, "Warning", "Please select X and Y columns")
                    self.build_btn.setEnabled(True)
                    return

                # Get original column names (may contain newlines)
                x_col = self.anim_col_name_map.get(x_col_display, x_col_display)
                y_col = self.anim_col_name_map.get(y_col_display, y_col_display)

                # Apply cleaning
                points, cleaned_df_result, messages = clean_data(
                    self.uploaded_df,
                    x_col, y_col,
                    remove_outliers=self.anim_remove_outliers_cb.isChecked(),
                    outlier_std=3.0,
                    standardize=self.anim_standardize_cb.isChecked()
                )

                if points is None:
                    QMessageBox.critical(self, "Error", "\n".join(messages))
                    self.build_btn.setEnabled(True)
                    return

                self.data_points = points

                # Save original (pre-standardization) row values for hover viewer
                # cleaned_df_result.index = surviving row indices from uploaded_df
                self.original_rows_df = self.uploaded_df.loc[cleaned_df_result.index].reset_index(drop=True)
                self.original_xy_cols = (x_col, y_col)  # Original column names for axis labels
                self.hover_dataset_label = f"Uploaded Data / X={x_col}, Y={y_col}"

                self.frame_info_label.setText(" | ".join(messages))

            self.hover_analysis_settings = {
                "segments": self.anim_segments_spin.value(),
                "gradient_threshold_ratio": self.anim_gradient_spin.value(),
                "center_method": self.anim_center_combo.currentText(),
                "recursion_enabled": self.anim_recursion_cb.isChecked(),
                "max_recursion_depth": self.anim_recursion_depth_spin.value(),
                "standardized": (
                    self.data_source_combo.currentIndex() != 0
                    and self.anim_standardize_cb.isChecked()
                ),
            }

            # Reset global state
            _reset_global_ranges()

            # Generate StepFrame data list (millisecond-level)
            self.frames = build_enhanced_visible_frames(
                points=self.data_points,
                segments=self.anim_segments_spin.value(),
                center_method=self.anim_center_combo.currentText(),
                gradient_threshold_ratio=self.anim_gradient_spin.value(),
                enable_recursion=self.anim_recursion_cb.isChecked(),
                max_recursion_depth=self.anim_recursion_depth_spin.value(),
                circle_animation_frames=1,  # Only 1 frame, we'll use GIF instead
                export_final_silhouettes=False
            )

            # Calculate the baseline coordinate range for top-level frames.
            # Child / grandchild steps compute their own larger extents at render time,
            # so we avoid shrinking the overview frames just to reserve space for
            # later recursive circles.
            self.global_rmax = 0
            self.global_centroid = None

            # First try to get r values from first frame (raw data)
            if self.frames and 'r' in self.frames[0].payload:
                r_values = np.asarray(self.frames[0].payload['r'], dtype=float)
                # Filter out NaN and Inf values for robustness
                r_values = r_values[np.isfinite(r_values)]
                if r_values.size > 0:
                    self.global_rmax = float(np.max(r_values))  # Use max for consistency with static frames
                else:
                    self.global_rmax = 1.0  # Fallback for empty/all-NaN data
            else:
                # Fallback: use max of rmax values
                for frame in self.frames:
                    if 'rmax' in frame.payload:
                        rmax_val = frame.payload['rmax']
                        if np.isfinite(rmax_val):
                            self.global_rmax = max(self.global_rmax, rmax_val)
                if self.global_rmax == 0:
                    self.global_rmax = 1.0  # Safety fallback

            for frame in self.frames:
                if 'centroid' in frame.payload and self.global_centroid is None:
                    self.global_centroid = frame.payload['centroid']

            # ========================================
            # Sync slider parameters to the GIF generation module
            # ========================================
            from vmeans.rendering import base as pyqt_matplotlib_vis

            # Point size multiplier
            point_multiplier = self.point_size_slider.value() / 100.0
            pyqt_matplotlib_vis._GLOBAL_POINT_MULTIPLIER = point_multiplier

            # Zoom / field-of-view
            zoom_factor = self.zoom_slider.value() / 100.0
            pyqt_matplotlib_vis._GLOBAL_ZOOM = zoom_factor

            # Global rmax -- keep GIF and static frames on the same scale
            pyqt_matplotlib_vis._GLOBAL_RMAX = self.global_rmax

            # Compute chart offset to visually center asymmetric data
            if self.frames and 'r' in self.frames[0].payload and 'theta' in self.frames[0].payload:
                from vmeans.rendering.base import _compute_axes_shift
                r_vals = np.asarray(self.frames[0].payload['r'], dtype=float)
                theta_vals = np.asarray(self.frames[0].payload['theta'], dtype=float)
                _compute_axes_shift(r_vals, theta_vals)

            # Chart size -- figsize must match static frames; resolution
            # comes from bumping up the DPI instead
            chart_px = self.chart_size_slider.value()
            dpr = self.devicePixelRatioF()  # typically 2.0 on Retina displays
            chart_inches = chart_px / FIGURE_DPI  # keep figsize consistent with static frames
            current_figsize = (chart_inches, chart_inches)

            # Stash dpr for playback
            self._gif_dpr = dpr

            # Higher DPI for crisp GIF output (actual pixels = figsize * dpi)
            gif_dpi = int(FIGURE_DPI * dpr)
            self._gif_dpi = gif_dpi

            print(f"🔄 Syncing to GIF: Points={point_multiplier:.2f}x, Zoom={zoom_factor:.2f}x, Rmax={self.global_rmax:.2f}, Size={chart_inches:.2f}in @ {gif_dpi}dpi (DPR={dpr:.1f})")

            # ========================================
            # Use live animation instead of GIF files.
            # Renders directly on the canvas -- no file I/O.
            # ========================================
            dmax_frame = None
            for f in self.frames:
                if f.name == 'dmax_circle':
                    dmax_frame = f
                    break

            if dmax_frame is not None:
                # Replace dmax_circle_only and dmax_circle_animating with live anim.
                # Keep dmax_circle (Step 4, the straight line) as-is!
                new_frames = []
                animation_added = False
                for f in self.frames:
                    if f.name == 'dmax_circle_only' or f.name == 'dmax_circle_animating':
                        # Swap in animation frames for Step 5
                        if not animation_added:
                            new_frames.append(StepFrame('live_circle_animation', {
                                'r': dmax_frame.payload['r'],
                                'theta': dmax_frame.payload['theta'],
                                'rmax': dmax_frame.payload['rmax'],
                                'dmax_idx': dmax_frame.payload['dmax_idx'],
                                'display_rmax': dmax_frame.payload.get('display_rmax'),
                                'n_frames': 30,
                                'animation_type': 'parent_circle'
                            }, meta={'step': 5, 'description': 'Drawing d_max Silhouette (Animation)'}))
                            animation_added = True
                        # Skip these frames
                    else:
                        new_frames.append(f)

                self.frames = new_frames

            # ========================================
            # Live animation for child circles too
            # ========================================
            child_circle_frame = None
            for f in self.frames:
                if f.name == 'parallel_child_dmax_circle':
                    child_circle_frame = f
                    break

            if child_circle_frame is not None:
                child_analyses = child_circle_frame.payload.get('child_analyses', [])
                parent_centroid = child_circle_frame.payload.get('parent_centroid')
                parent_r = child_circle_frame.payload.get('parent_r')


                if child_analyses and parent_centroid is not None and parent_r is not None:
                    # Replace original frame with live animation
                    new_frames = []
                    animation_added = False
                    for f in self.frames:
                        if f.name == 'parallel_child_dmax_circle':
                            if not animation_added:
                                live_payload = dict(child_circle_frame.payload)
                                live_payload.update({
                                    'n_frames': 30,
                                    'animation_type': 'child_circle',
                                    'level_label': 'Child',
                                    'step_label': '14.5'
                                })
                                new_frames.append(StepFrame(
                                    'live_child_circle_animation',
                                    live_payload,
                                    meta={
                                        'step': 14.5,
                                        'description': 'Child - Drawing d_max Circles (Animation)'
                                    }
                                ))
                                animation_added = True
                            # Drop the original frame
                        else:
                            new_frames.append(f)

                    self.frames = new_frames

            # ========================================
            # Live animation for grandchild circles
            # ========================================
            grandchild_circle_frame = next(
                (f for f in self.frames if f.name == 'parallel_grandchild_dmax_circle'),
                None
            )
            if grandchild_circle_frame is not None:
                grandchild_analyses = grandchild_circle_frame.payload.get('child_analyses', [])
                parent_centroid = grandchild_circle_frame.payload.get('parent_centroid')
                parent_r = grandchild_circle_frame.payload.get('parent_r')

                if grandchild_analyses and parent_centroid is not None and parent_r is not None:
                    new_frames = []
                    animation_added = False
                    for f in self.frames:
                        if f.name == 'parallel_grandchild_dmax_circle':
                            if not animation_added:
                                live_payload = dict(grandchild_circle_frame.payload)
                                live_payload.update({
                                    'n_frames': 30,
                                    'animation_type': 'child_circle',
                                    'level_label': 'Grandchild',
                                    'step_label': '15.5'
                                })
                                new_frames.append(StepFrame('live_child_circle_animation', live_payload, meta={
                                    'step': '15.5',
                                    'description': 'Grandchild - Drawing d_max Silhouettes (Animation)'
                                }))
                                animation_added = True
                        else:
                            new_frames.append(f)
                    self.frames = new_frames

            # Initialize player
            self.frame_slider.setRange(0, len(self.frames) - 1)
            self.frame_slider.setValue(0)

            # Center viewport on first build
            self._last_canvas_size = None
            self._need_center_viewport = True

            # Show first frame (this is when the first drawing happens)
            self.show_frame(0)

            self.build_btn.setEnabled(True)
            self.export_btn.setEnabled(True)  # Enable export button
            self.frame_info_label.setText(f"Ready: {len(self.frames)} frames (Live Rendering)")

            # Give focus to the frame slider so arrow keys work immediately
            self.frame_slider.setFocus()

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to build: {str(e)}")
            self.build_btn.setEnabled(True)
            self.export_btn.setEnabled(False)
