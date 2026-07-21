
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

class StepAnimationRenderMixin:
    def show_frame(self, index: int):
        """Render and display frame - Hard Replace Strategy"""
        if 0 <= index < len(self.frames):
            self.current_step = index
            frame = self.frames[index]

            # ========================================
            # Check if this is a live animation frame
            # ========================================
            if frame.name in ('live_circle_animation', 'live_child_circle_animation'):
                self.play_live_animation(frame)
                return

            # ========================================
            # Stop any playing animation and switch back to normal rendering
            # ========================================
            self.stop_live_animation()

            try:
                # ========================================
                # Step 1: Calculate chart size first (needed for container)
                # ========================================
                chart_px = self.chart_size_slider.value()
                render_w_px = round(chart_px)
                render_h_px = round(chart_px)
                fig_inches = render_h_px / FIGURE_DPI

                # Detect whether the canvas size actually changed
                new_size = (render_w_px, render_h_px)
                size_changed = (self._last_canvas_size != new_size)
                if size_changed:
                    self._last_canvas_size = new_size
                    self._need_center_viewport = True

                # ========================================
                # Freeze UI updates to prevent layout reflow
                # ========================================
                self.scroll_area.setUpdatesEnabled(False)
                self.canvas_container.setUpdatesEnabled(False)

                try:
                    # ========================================
                    # Step 3: Clean up old Canvas and Figure
                    # ========================================
                    if self.canvas is not None:
                        old_fig = self.canvas.figure
                        self.canvas_layout.removeWidget(self.canvas)
                        self.canvas.setParent(None)
                        self.canvas.hide()
                        self.canvas.deleteLater()
                        self.canvas = None
                        if old_fig is not None:
                            plt.close(old_fig)

                    # ========================================
                    # Step 4: Build RenderOptions from UI settings
                    # ========================================
                    is_final_frame = frame.name in ['parallel_child_final']
                    zoom_factor = self.zoom_slider.value() / 100.0

                    render_opts = RenderOptions(
                        point_size=self.point_size_slider.value() / 100.0,
                        silhouette_width=self.silhouette_slider.value() / 100.0,
                        show_legend=False,
                        show_silhouettes=self.show_silhouettes_cb.isChecked() if is_final_frame else True,
                        show_gradients=False,
                        show_centers=False,
                        simplified_final=is_final_frame,
                        zoom_factor=zoom_factor
                    )

                    # ========================================
                    # Step 5: Draw the frame
                    # ========================================
                    fig = draw_frame_matplotlib(
                        frame,
                        figsize=(fig_inches, fig_inches),
                        global_rmax=self.global_rmax,
                        render_options=render_opts
                    )

                    # ========================================
                    # Step 6: Create new Canvas and insert into layout
                    # ========================================
                    self.canvas = FigureCanvas(fig)
                    self.canvas.setFixedSize(render_w_px, render_h_px)
                    self.canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                    self.canvas_layout.insertWidget(0, self.canvas, 0, Qt.AlignmentFlag.AlignCenter)

                    # Lock the container size so the scroll area doesn't recalculate
                    self.canvas_container.setFixedSize(render_w_px, render_h_px)
                    self.gif_label.setFixedSize(render_w_px, render_h_px)

                finally:
                    # ========================================
                    # Unfreeze UI updates
                    # ========================================
                    self.canvas_container.setUpdatesEnabled(True)
                    self.scroll_area.setUpdatesEnabled(True)

                # Only recenter when size changes
                if self._need_center_viewport:
                    self.center_viewport()
                    self._need_center_viewport = False

            except Exception as e:
                import traceback
                print(f"Error rendering frame {index}: {e}")
                traceback.print_exc()

            # Update UI information
            self.update_frame_info(frame, index)

            # Show hover button only on last frame
            is_last = (index == len(self.frames) - 1)
            self.hover_btn.setVisible(is_last)
            self.hover_btn.setEnabled(is_last)


    def _select_hover_frame(self):
        """Find the nearest rendered frame that contains point data for hover details."""
        candidates = self.frames[:self.current_step + 1]
        preferred_names = {
            'export_silhouettes_only',
            'parallel_grandchild_final',
            'parallel_grandchild_find_matching_pairs',
            'parallel_child_final',
            'parallel_child_find_matching_pairs',
            'find_matching_gradient_pairs',
        }

        def has_points(frame):
            payload = getattr(frame, 'payload', {}) or {}
            return payload.get('all_points') is not None or payload.get('points') is not None

        # Hospital tooltips describe the original 15 V-Means clusters.  Later
        # recursive frames split a few of those clusters into 23/24 leaf
        # regions, which changes the meaning of the table and its legend.
        original_rows = getattr(self, 'original_rows_df', None)
        normalized_columns = {
            ' '.join(str(column).replace('\n', ' ').split()).lower()
            for column in getattr(original_rows, 'columns', [])
        }
        hospital_columns = {
            'primary diagnosis: summary code and description',
            'finished admission episodes',
            'mean time waited (days)',
            'mean age (years)',
        }
        if hospital_columns.issubset(normalized_columns):
            for frame in reversed(candidates):
                if frame.name == 'find_matching_gradient_pairs' and has_points(frame):
                    return frame

        for frame in reversed(candidates):
            if frame.name in preferred_names and has_points(frame):
                return frame

        for frame in reversed(candidates):
            if has_points(frame):
                return frame

        return None


    def open_hover_viewer(self):
        """Open the hover scatter viewer dialog for the last frame."""
        if not self.frames:
            return
        if self.current_step != len(self.frames) - 1:
            return
        frame = self._select_hover_frame()
        if frame is None:
            QMessageBox.warning(
                self,
                "Hover Details",
                "No point data is available for the final frame."
            )
            return

        try:
            self.hover_dialog = HoverScatterDialog(
                frame.name, frame.payload,
                original_df=getattr(self, 'original_rows_df', None),
                original_xy_cols=getattr(self, 'original_xy_cols', None),
                dataset_label=getattr(self, 'hover_dataset_label', None),
                analysis_settings=getattr(self, 'hover_analysis_settings', None),
                parent=self
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Hover Details",
                f"Could not open hover details for this frame:\n{exc}"
            )
            return

        self.hover_dialog.show()
        self.hover_dialog.raise_()
        self.hover_dialog.activateWindow()


    def on_chart_size_changed(self, value):
        """Handle chart size slider change"""
        self.chart_size_label.setText(f"{value}px")
        # Re-render current frame if animation is loaded
        if hasattr(self, 'frames') and self.frames and len(self.frames) > 0:
            self.show_frame(self.current_step)


    def on_zoom_changed(self, value):
        """Handle zoom slider change"""
        self.zoom_label.setText(f"{value}%")
        # Re-render current frame if animation is loaded
        if hasattr(self, 'frames') and self.frames and len(self.frames) > 0:
            self.show_frame(self.current_step)


    def center_viewport(self):
        """Auto-center the scroll area viewport on the chart (horizontally), align to top (vertically)

        Should only be called on first show or when size changes -- never per-frame!
        """
        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()

        # Horizontal: center
        h_bar.setValue(h_bar.maximum() // 2)
        # Vertical: align to TOP to ensure title is visible
        v_bar.setValue(0)


    def export_final_frame(self):
        """Export the final frame as a high-resolution PNG with silhouettes and legend

        Uses the gradient_lines frame (Step 15.10 or 14.10) but WITHOUT gradient lines.
        """
        if not hasattr(self, 'frames') or not self.frames:
            QMessageBox.warning(self, "Warning", "No animation loaded. Please build animation first.")
            return

        # Prefer the dedicated export frame because it contains the deepest
        # recursive leaf composition (including untouched Child leaves).
        gradient_frame = None

        gradient_frame = next(
            (frame for frame in reversed(self.frames)
             if frame.name == 'export_silhouettes_only'),
            None
        )

        # Fallback for older frame sequences without a dedicated export frame.
        if gradient_frame is None:
            for frame in reversed(self.frames):
                if 'grandchild_gradient_lines' in frame.name:
                    gradient_frame = frame
                    break

        if gradient_frame is None:
            for frame in reversed(self.frames):
                if 'child_gradient_lines' in frame.name:
                    gradient_frame = frame
                    break

        if gradient_frame is None:
            # Fallback to last frame
            gradient_frame = self.frames[-1]


        # Ask user for save location
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Final Frame",
            "final_analysis.png",
            "PNG Images (*.png);;All Files (*)"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Import required modules — route the export through the
            # coloured-silhouette module when the switch is on so the
            # exported figure matches the on-screen preview.
            if USE_COLORED_SILHOUETTES:
                from vmeans.rendering.colored import draw_export_silhouettes_only
            else:
                from vmeans.rendering.base import draw_export_silhouettes_only
            from vmeans.rendering.base import RenderOptions
            from matplotlib.figure import Figure

            # Build render options - ENABLE legend for export
            render_opts = RenderOptions()
            render_opts.point_size = self.point_size_slider.value() / 100.0
            render_opts.silhouette_width = self.silhouette_slider.value() / 100.0
            render_opts.show_legend = True  # ENABLE legend!
            render_opts.show_silhouettes = True
            render_opts.show_gradients = False
            render_opts.show_centers = False  # No X markers
            render_opts.simplified_final = True

            # Check if we have child_analyses in the payload
            payload = gradient_frame.payload
            if 'child_analyses' in payload:
                # Use chart size from slider (convert px to inches)
                chart_px = self.chart_size_slider.value()
                export_size_inches = max(8, chart_px / 100.0)  # At least 8 inches, scaled from slider

                # Use the new export function with legend
                # 300 DPI for paper / presentation quality
                fig = draw_export_silhouettes_only(payload, gradient_frame.meta, figsize=(export_size_inches, export_size_inches), opts=render_opts)
                fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                print(f"Exported with draw_export_silhouettes_only: {file_path} (size={export_size_inches}in, dpi=300)")
            else:
                # Fallback to old method
                # 300 DPI export
                from vmeans.rendering.base import export_final_frame_hd
                export_final_frame_hd(
                    gradient_frame,
                    output_path=file_path,
                    figsize=(12, 12),
                    dpi=300,
                    global_rmax=self.global_rmax,
                    render_options=render_opts
                )

            QMessageBox.information(self, "Success", f"Final frame exported to:\n{file_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")


    def _fit_gif_keep_aspect(self):
        """Keep GIF dimensions consistent with static frames."""
        from PyQt6.QtCore import QSize

        # Bail out if there's no valid movie object
        if not getattr(self, "gif_movie", None) or not self.gif_movie.isValid():
            return

        # Use chart_size_slider value directly (logical pixels)
        chart_px = int(self.chart_size_slider.value())

        # Sync all dimensions
        self.canvas_container.setFixedSize(chart_px, chart_px)
        self.gif_label.setFixedSize(chart_px, chart_px)
        self.gif_movie.setScaledSize(QSize(chart_px, chart_px))

        self.gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


    def resizeEvent(self, event):
        """Re-fit the GIF when the window is resized."""
        super().resizeEvent(event)
        # Don't recalculate during GIF playback (causes visible jitter)
        if getattr(self, "gif_playing", False):
            return
        # QTimer ensures layout has settled before we measure
        QTimer.singleShot(0, self._fit_gif_keep_aspect)


    def play_gif_frame(self, frame):
        """Play GIF animation for circle drawing"""
        from PyQt6.QtGui import QMovie

        gif_path = frame.payload.get('gif_path', self.gif_path)
        if not gif_path or not os.path.exists(gif_path):
            print(f"GIF file not found: {gif_path}")
            self.advance_frame()  # Skip to next frame
            return

        # Remember whether auto-play was active, then pause
        self._resume_after_gif = self.is_playing
        if self.timer.isActive():
            self.timer.stop()

        # Hide canvas, show GIF label
        if self.canvas:
            self.canvas.hide()
        self.gif_label.show()

        # Stop any existing movie
        if self.gif_movie:
            self.gif_movie.stop()
            try:
                self.gif_movie.frameChanged.disconnect(self.on_gif_frame_changed)
            except:
                pass
            self.gif_movie.deleteLater()

        # Create and start new movie
        self.gif_movie = QMovie(gif_path)
        self.gif_label.setMovie(self.gif_movie)

        # Cache all frames for smooth playback
        self.gif_movie.setCacheMode(QMovie.CacheMode.CacheAll)

        # Jump to frame 0 so QMovie reports correct metadata
        self.gif_movie.jumpToFrame(0)

        # Frame count (can be -1 if the format doesn't report it)
        self.gif_total_frames = self.gif_movie.frameCount()
        self.gif_frame_counter = 0  # manual counter as fallback

        # Wire up the frameChanged signal
        self.gif_movie.frameChanged.connect(self.on_gif_frame_changed)

        self.gif_playing = True
        self.gif_finished_triggered = False

        # Sync container dimensions
        chart_px = int(self.chart_size_slider.value())
        self.canvas_container.setFixedSize(chart_px, chart_px)
        self.gif_label.setFixedSize(chart_px, chart_px)

        # Fit the GIF into the viewport
        self._fit_gif_keep_aspect()
        QTimer.singleShot(0, self._fit_gif_keep_aspect)

        self.gif_movie.start()

        # Update UI info
        self.update_frame_info(frame, self.current_step)


    def on_gif_frame_changed(self, frame_num):
        """Called when GIF frame changes - detect end of animation"""
        if not self.gif_playing or not self.gif_movie or self.gif_finished_triggered:
            return

        self.gif_frame_counter += 1

        # Detect whether we've reached the last frame.
        # If frameCount is valid, compare against it;
        # otherwise detect the wrap-around back to frame 0.
        is_last_frame = False

        if self.gif_total_frames > 0:
            # frameCount is known
            if frame_num >= self.gif_total_frames - 1:
                is_last_frame = True
        else:
            # frameCount unknown -- detect loop restart
            if frame_num == 0 and self.gif_frame_counter > 1:
                is_last_frame = True

        if is_last_frame:
            self.gif_finished_triggered = True
            self.gif_movie.stop()
            QTimer.singleShot(100, self.on_gif_finished)


    def on_gif_finished(self):
        """Called when GIF animation finishes"""
        if not self.gif_playing:
            return

        self.gif_playing = False

        # Disconnect signal to prevent any further triggers
        if self.gif_movie:
            try:
                self.gif_movie.frameChanged.disconnect(self.on_gif_frame_changed)
            except:
                pass

        # Restore previous playback state
        if getattr(self, '_resume_after_gif', False):
            # Resume auto-play: advance one frame and restart the timer
            self.advance_frame()
            self.timer.start(self.speed_slider.value())
        else:
            # Not auto-playing -- just show the next frame
            next_idx = (self.current_step + 1) % len(self.frames)
            self.show_frame(next_idx)


    def stop_gif(self):
        """Stop GIF playback and switch back to canvas view"""
        if self.gif_movie:
            self.gif_movie.stop()
            try:
                self.gif_movie.frameChanged.disconnect(self.on_gif_frame_changed)
            except:
                pass

        self.gif_playing = False
        self.gif_label.hide()


    def play_live_animation(self, frame):
        """Play live animation directly on Canvas - no GIF needed!"""
        # Remember auto-play state, then pause
        self._resume_after_animation = self.is_playing
        if self.timer.isActive():
            self.timer.stop()

        # Stash animation parameters
        self._animation_frame = frame
        self._animation_current = 0
        self._animation_total = frame.payload.get('n_frames', 30)
        self._animation_type = frame.payload.get('animation_type', 'parent_circle')

        # Animation timer (40ms interval = ~25 fps)
        if self._animation_timer is None:
            self._animation_timer = QTimer()
            self._animation_timer.timeout.connect(self._render_animation_frame)

        self._animation_playing = True

        # Render the first frame
        self._render_animation_frame()

        # Start the animation loop
        self._animation_timer.start(40)

        # Update UI info
        self.update_frame_info(frame, self.current_step)
        self.frame_info_label.setText(f"🎬 Live Animation: {self._animation_type}")


    def stop_live_animation(self):
        """Stop live animation"""
        if hasattr(self, '_animation_timer') and self._animation_timer is not None and self._animation_timer.isActive():
            self._animation_timer.stop()

        self._animation_playing = False

        # If auto-play was active before, resume it
        if hasattr(self, '_resume_after_animation') and self._resume_after_animation:
            self._resume_after_animation = False
            # Brief delay so the current frame finishes rendering first
            QTimer.singleShot(100, self._resume_playback_after_animation)


    def _resume_playback_after_animation(self):
        """Resume playback after animation finishes"""
        if self.is_playing and not self.timer.isActive():
            self.timer.start(self.speed_slider.value())


    def _render_animation_frame(self):
        """Render a single animation frame on Canvas"""
        if not hasattr(self, '_animation_frame') or not self._animation_playing:
            return

        frame = self._animation_frame
        progress = self._animation_current / max(1, self._animation_total - 1)

        try:
            # Grab render parameters
            chart_px = self.chart_size_slider.value()
            render_px = round(chart_px)
            fig_inches = render_px / FIGURE_DPI
            figsize = (fig_inches, fig_inches)

            # Dispatch to the right renderer based on animation type
            if self._animation_type == 'parent_circle':
                fig = self._render_parent_circle_frame(frame.payload, progress, figsize)
            elif self._animation_type == 'child_circle':
                fig = self._render_child_circle_frame(frame.payload, progress, figsize)
            else:
                fig = None

            if fig is not None:
                # Show the rendered figure on the canvas directly
                self._display_figure_on_canvas(fig, render_px)

        except Exception as e:
            import traceback
            print(f"Animation frame error: {e}")
            print(traceback.format_exc())

        # Advance progress
        self._animation_current += 1

        # Check if finished
        if self._animation_current >= self._animation_total:
            self._animation_timer.stop()
            self._animation_playing = False

            # Short delay before moving to the next frame
            QTimer.singleShot(200, self._on_animation_finished)


    def _on_animation_finished(self):
        """Called when the animation finishes -- stays on the last frame."""
        # Render the final frame (progress=1.0) and stop.
        # The user can press next-frame to advance past it.
        self._animation_completed = True

        # Restore the auto-play timer if it was running before
        if hasattr(self, '_resume_after_animation') and self._resume_after_animation:
            self._resume_after_animation = False
            if self.is_playing:
                self.timer.start(self.speed_slider.value())


    def _display_figure_on_canvas(self, fig, render_px):
        """Display a matplotlib Figure on the Canvas"""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

        # Create a fresh canvas
        new_canvas = FigureCanvas(fig)
        new_canvas.setFixedSize(render_px, render_px)

        # Swap out the old canvas
        if self.canvas:
            old_canvas = self.canvas
            self.canvas_container.layout().replaceWidget(old_canvas, new_canvas)
            old_canvas.deleteLater()
        else:
            self.canvas_container.layout().insertWidget(0, new_canvas)

        self.canvas = new_canvas
        self.canvas.show()

        # Sync container size
        self.canvas_container.setFixedSize(render_px, render_px)


    def _render_parent_circle_frame(self, payload, progress, figsize):
        """Render parent circle animation frame"""
        import numpy as np
        from vmeans.rendering.base import (
            _create_figure_with_panel, _configure_polar_ax, _fill_info_panel,
            _get_display_rmax, _get_point_size, _get_center_size, _get_highlight_size,
            COLOR_DATA_POINTS_GRAY, COLOR_CENTER, COLOR_DMAX, COLOR_SILHOUETTE,
            SILHOUETTE_LINE_WIDTH, CHILD_SILHOUETTE_LINE_WIDTH
        )

        r = payload['r']
        theta = payload['theta']
        rmax = payload['rmax']
        dmax_idx = payload['dmax_idx']

        # Use global rmax
        global_rmax = payload.get(
            'display_rmax',
            getattr(self, 'global_rmax', rmax)
        )
        display_rmax = _get_display_rmax(global_rmax)

        # Create figure
        fig, ax, ax_info = _create_figure_with_panel(figsize)
        ax.set_facecolor('white')

        # Static elements
        ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(), alpha=0.5,
                   label='Points', zorder=2)
        ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(),
                   label='Center', zorder=10)
        ax.scatter([theta[dmax_idx]], [r[dmax_idx]], c=COLOR_DMAX, s=_get_highlight_size(),
                   label=f'd={rmax:.1f}', zorder=15)
        ax.plot([0, theta[dmax_idx]], [0, r[dmax_idx]], COLOR_DMAX, linewidth=3, alpha=0.8, zorder=8)

        # Arc starts at the d_max point and sweeps counter-clockwise
        start_angle = theta[dmax_idx]  # starting angle
        sweep_angle = 2 * np.pi * progress  # how far we've swept
        end_angle = start_angle + sweep_angle

        # Arc sample points
        n_points = max(2, int(100 * progress))
        angles = np.linspace(start_angle, end_angle, n_points)
        radii = [rmax] * len(angles)

        ax.plot(angles, radii, COLOR_SILHOUETTE, linewidth=SILHOUETTE_LINE_WIDTH,
                linestyle='-', label='Silhouette', zorder=5)

        # Pen marker at the end of the arc
        if len(angles) > 0:
            ax.plot([angles[-1]], [radii[-1]], 'o', color='white', markersize=10,
                    markeredgecolor=COLOR_SILHOUETTE, markeredgewidth=3, zorder=20)

        _configure_polar_ax(ax, display_rmax, show_grid=True)

        # Title
        pct = int(progress * 100)
        title_text = f'Drawing d_max Silhouette ({pct}%)' if pct < 100 else 'Draw d_max Silhouette (Complete)'
        ax.set_title(f'Step 5\n{title_text}', fontsize=8, fontweight="bold", pad=0, y=1.05)

        # Right-side info panel
        handles, labels = ax.get_legend_handles_labels()
        _fill_info_panel(ax_info, handles=handles, labels=labels)

        return fig


    def _render_child_circle_frame(self, payload, progress, figsize):
        """Render child circle animation frame"""
        import numpy as np
        from vmeans.rendering.base import (
            _create_figure_with_panel, _configure_polar_ax, _fill_info_panel,
            _get_display_rmax, _get_point_size, _get_center_size,
            _draw_recursive_context,
            COLOR_DATA_POINTS_GRAY, COLOR_CENTER, COLOR_DMAX, COLOR_SILHOUETTE,
            SILHOUETTE_LINE_WIDTH, CHILD_SILHOUETTE_LINE_WIDTH, get_colors_for_centers
        )
        from vmeans.data import convert_to_polar

        child_analyses = payload['child_analyses']
        parent_centroid = payload['parent_centroid']
        parent_r = payload['parent_r']
        level_label = payload.get('level_label', 'Child')
        step_label = payload.get('step_label', '14.5')

        if not child_analyses:
            return None

        # Display range (same calculation as static frames)
        from vmeans.rendering.base import _get_child_max_r
        child_max_r = payload.get('display_rmax')
        if child_max_r is None:
            child_max_r = _get_child_max_r(
                np.array(parent_r), child_analyses, parent_centroid
            )
        display_rmax = _get_display_rmax(child_max_r)

        # Create figure
        fig, ax, ax_info = _create_figure_with_panel(figsize)
        ax.set_facecolor('white')
        if payload.get('context_child_analyses'):
            _draw_recursive_context(
                ax,
                payload,
                include_labels=False,
                include_silhouettes=True
            )

        # The correct key is 'center_global'
        child_centers = [ca['center_global'] for ca in child_analyses]
        colors = get_colors_for_centers(np.array(child_centers), origin=parent_centroid)

        # Draw each child's data points and animated arc
        for idx, ca in enumerate(child_analyses):
            child_points = ca['points']
            child_center = ca['center_global']
            child_rmax = ca['rmax']
            child_r = ca['r']
            child_theta = ca['theta']
            color = colors[idx]

            # Convert to parent coordinate system
            r_child, theta_child = convert_to_polar(child_points, parent_centroid)

            # Draw points
            ax.scatter(theta_child, r_child, c=color, s=_get_point_size(), alpha=0.5, zorder=2)

            # Arc starts at the child's local d_max angle
            dmax_idx = np.argmax(child_r)
            dmax_angle_local = child_theta[dmax_idx]

            # Sweep counter-clockwise from d_max
            start_angle = dmax_angle_local
            sweep_angle = 2 * np.pi * progress
            end_angle = start_angle + sweep_angle
            angles_local = np.linspace(start_angle, end_angle, max(2, int(50 * progress)))

            arc_theta, arc_r = [], []
            for angle_local in angles_local:
                local_x = child_rmax * np.cos(angle_local)
                local_y = child_rmax * np.sin(angle_local)
                global_point = np.array([child_center[0] + local_x, child_center[1] + local_y])
                r_pt, theta_pt = convert_to_polar(np.array([global_point]), parent_centroid)
                arc_theta.append(theta_pt[0])
                arc_r.append(r_pt[0])

            # Arc colour: child's own colour when the coloured-silhouette
            # switch is on (recursion step requirement), otherwise the
            # original uniform blue.
            arc_color = color if USE_COLORED_SILHOUETTES else COLOR_SILHOUETTE
            ax.plot(arc_theta, arc_r, color=arc_color, linewidth=CHILD_SILHOUETTE_LINE_WIDTH,
                    linestyle='-', zorder=5)

            # d_max marker at the arc start
            dmax_local_x = child_rmax * np.cos(dmax_angle_local)
            dmax_local_y = child_rmax * np.sin(dmax_angle_local)
            dmax_global = np.array([child_center[0] + dmax_local_x, child_center[1] + dmax_local_y])
            dmax_r_pt, dmax_theta_pt = convert_to_polar(np.array([dmax_global]), parent_centroid)

            # Line from center to d_max
            child_r_parent, child_theta_parent = convert_to_polar(np.array([child_center]), parent_centroid)
            ax.plot([child_theta_parent[0], dmax_theta_pt[0]], [child_r_parent[0], dmax_r_pt[0]],
                    color='orange', linewidth=3, alpha=0.8, zorder=8)

            ax.scatter([dmax_theta_pt[0]], [dmax_r_pt[0]], c='orange', s=_get_point_size() * 2,
                       edgecolors='black', linewidth=1, zorder=12, marker='o')

        # Draw center star markers
        for idx, ca in enumerate(child_analyses):
            child_center = ca['center_global']
            center_r, center_theta = convert_to_polar(np.array([child_center]), parent_centroid)
            color = colors[idx]
            ax.scatter([center_theta[0]], [center_r[0]], c=color, s=_get_center_size(),
                       marker='*', edgecolors='black', linewidth=0.5, zorder=10,
                       label=f'C{idx+1}')

        _configure_polar_ax(ax, display_rmax, show_grid=True)

        pct = int(progress * 100)
        title_text = (
            f'{level_label} - Drawing d_max Silhouettes ({pct}%)'
            if pct < 100
            else f'{level_label} - d_max Silhouettes (Complete)'
        )
        ax.set_title(
            f'Step {step_label}\n{title_text}',
            fontsize=8, fontweight="bold", pad=0, y=1.05
        )

        handles, labels = ax.get_legend_handles_labels()
        _fill_info_panel(ax_info, handles=handles, labels=labels)

        return fig


    def update_frame_info(self, frame, index):
        """Update frame info labels"""
        self.frame_counter.setText(f"{index + 1}/{len(self.frames)}")
        # Note: Step info is shown in the chart title, no separate label needed

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)


    def on_slider_changed(self, value):
        """Handle slider movement"""
        self.show_frame(value)
        if self.is_playing:
            self.toggle_play()


    def toggle_play(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_btn.setText("⏸️ Pause")
            self.timer.start(self.speed_slider.value())
        else:
            self.play_btn.setText("▶️ Play")
            self.timer.stop()


    def advance_frame(self):
        """Advance to next frame (called by timer)"""
        if len(self.frames) > 0:
            next_idx = (self.current_step + 1) % len(self.frames)
            self.show_frame(next_idx)


    def next_frame_manual(self):
        """Manual next frame button"""
        if len(self.frames) > 0:
            next_idx = (self.current_step + 1) % len(self.frames)
            self.show_frame(next_idx)


    def prev_frame(self):
        """Go to previous frame"""
        if len(self.frames) > 0:
            prev_idx = (self.current_step - 1) % len(self.frames)
            self.show_frame(prev_idx)


    def go_first(self):
        """Go to first frame"""
        if len(self.frames) > 0:
            self.show_frame(0)
            if self.is_playing:
                self.toggle_play()


    def go_last(self):
        """Go to last frame"""
        if len(self.frames) > 0:
            self.show_frame(len(self.frames) - 1)
            if self.is_playing:
                self.toggle_play()


    def keyPressEvent(self, event):
        """Arrow keys always control frames, regardless of which widget has focus."""
        key = event.key()
        if key == Qt.Key.Key_Right:
            self.next_frame_manual()
            event.accept()
        elif key == Qt.Key.Key_Left:
            self.prev_frame()
            event.accept()
        elif key == Qt.Key.Key_Home:
            self.go_first()
            event.accept()
        elif key == Qt.Key.Key_End:
            self.go_last()
            event.accept()
        elif key == Qt.Key.Key_Space:
            self.toggle_play()
            event.accept()
        else:
            super().keyPressEvent(event)
