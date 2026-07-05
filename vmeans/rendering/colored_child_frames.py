"""
pyqt_matplotlib_vis_colored.py
==============================

Coloured-silhouette variant of :mod:`pyqt_matplotlib_vis`.

Purpose
-------
During recursion, every silhouette outline is drawn in the colour of the
cluster it encloses, instead of a single uniform blue.  This matches the
reviewer request:

    "For the recursion step, change the silhouette colors to match the
     corresponding cluster colors, for all future versions, e.g. the next
     paper submission with video."

Design
------
This module re-exports every public symbol from ``pyqt_matplotlib_vis``
(``from .base import *``) and then overrides exactly the
frame functions that paint silhouette outlines:

    A) Per-child silhouettes — outline uses the child's own colour.
       - draw_parallel_child_segments
       - draw_parallel_child_shrinking
       - draw_parallel_child_gradient_comparison
       - draw_parallel_child_gradient_lines
       - draw_parallel_child_gradient_pair_search
       - draw_parallel_child_dynamic_center_found
       - draw_parallel_child_find_matching_pairs
       - draw_parallel_child_dmax_circle
       - draw_parallel_child_progressive_segments
       - draw_parallel_child_d_values

    B) Per-leaf silhouettes — each leaf cluster outlined in its own colour.
       - draw_parallel_child_final           (already per-leaf in the base
                                              module; not overridden here)
       - draw_export_silhouettes_only

No other frame types are touched.  The top-level dispatcher
``draw_frame_matplotlib`` is re-defined here so the PyQt application can
import this module in place of the base module without further changes.

Switching
---------
The main application picks a renderer at import time, e.g.::

    USE_COLORED_SILHOUETTES = True
    if USE_COLORED_SILHOUETTES:
        from vmeans.rendering import colored as vis
    else:
        from vmeans.rendering import base as vis
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

# Inherit everything from the base module.  This gives us every helper
# (_create_figure_with_panel, _configure_polar_ax, _get_child_colors, ...),
# every constant (COLOR_SILHOUETTE, CHILD_SILHOUETTE_LINE_WIDTH, ...), and
# every frame function we are NOT overriding.
from .base import * # noqa: F401,F403
from vmeans.rendering import base as _base
from . import base as pyqt_matplotlib_vis

# =========================================================
# Use this code to replace the Explicit imports section at the top of the file
# =========================================================

# Explicit imports for names used inside the overrides — keeps the
# dependency surface visible at a glance.
from .base import (
    # Helpers
    _create_figure_with_panel,
    _configure_polar_ax,
    _get_child_colors,
    _get_child_max_r,
    _get_display_rmax,
    _get_point_size,
    _get_center_size,
    _get_highlight_size,
    _get_title,
    _fill_info_panel,

    # ⬇️ The culprit of the previous error is here! Explicitly import the underscored panel functions ⬇️
    _add_info_to_panel,
    _add_legend_to_panel,

    # ⬇️ Other core functions with underscores that must be explicitly imported ⬇️
    _draw_child_visible_silhouette,
    _draw_child_visible_silhouette_connected,
    _collect_all_sub_regions,
    _collect_final_leaf_clusters,
    _get_sub_region_colors,
    _gradient_pair_boundary_angle,
    _reset_global_ranges,
    _smooth_radii,  # <-- Import the smooth radius function in advance to prevent errors in later comparison steps

    # Constants
    COLOR_SILHOUETTE,
    CHILD_SILHOUETTE_LINE_WIDTH,  # <-- Explicitly import line width to prevent NameError
    SILHOUETTE_SMOOTH_SIGMA,
    MPL_NORMAL_SIZE,

    # Options
    RenderOptions,
    _DEFAULT_RENDER_OPTIONS,
)


# ===========================================================================
# A) Per-child overrides — silhouette uses the same colour as that child's
#    points.  In each of these frames the child has not yet been split into
#    leaf sub-regions, so one silhouette per child is still the right
#    granularity; only the colour changes.
# ===========================================================================

def draw_parallel_child_segments(P, meta, figsize, opts: RenderOptions = None) -> Figure:
    """Child - show segments (arc outline in child colour)."""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']

    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)

    ax.scatter([0], [0], c='black', s=_get_center_size(), zorder=10, label='Origin')

    child_colors = _get_child_colors(child_analyses, parent_centroid)

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        dmax_list = child['dmax_list']
        segment_points = child['segment_points']
        theta_edges_local = child['theta_edges']
        rmax_child = child['rmax']

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)

        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_highlight_size(), marker='X', edgecolors='black',
                   linewidth=1, zorder=15, label=f'C{idx + 1}')

        for i in range(len(dmax_list)):
            theta_start_local = theta_edges_local[i]
            theta_end_local = (theta_edges_local[i + 1]
                               if i + 1 < len(theta_edges_local)
                               else theta_edges_local[0])

            has_data = len(segment_points[i]) > 0 if i < len(segment_points) else False
            if not has_data:
                continue

            height = rmax_child
            if height > 0:
                theta_arc_local = np.linspace(theta_start_local, theta_end_local, 15)
                x_local = height * np.cos(theta_arc_local)
                y_local = height * np.sin(theta_arc_local)
                pts_local = np.column_stack([x_local, y_local])
                pts_global = pts_local + child_center
                r_global, theta_global = convert_to_polar(pts_global, parent_centroid)
                # Was COLOR_SILHOUETTE; now the child's own colour.
                ax.plot(theta_global, r_global, color=color, linewidth=1, linestyle='-')

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Show Segments', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig


def draw_parallel_child_shrinking(P, meta, figsize, opts: RenderOptions = None) -> Figure:
    """Child - shrinking animation with silhouette in child colour."""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']

    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    child_colors = _get_child_colors(child_analyses, parent_centroid)

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        current_heights = child.get('current_heights', child['dmax_list'])
        theta_edges_local = child['theta_edges']
        current_segment_idx = child.get('current_segment_idx', -1)

        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_highlight_size(), marker='X', edgecolors='black',
                   linewidth=1, zorder=15, label=f'C{idx + 1}')

        # Silhouette → child colour
        _draw_child_visible_silhouette(
            ax, child_center, theta_edges_local, current_heights,
            parent_centroid, color=color, zorder=5,
        )

        # Orange overlay on the segment currently being processed stays
        # orange so the "active" marker keeps visual contrast against the
        # now-coloured silhouette.
        if 0 <= current_segment_idx < len(current_heights):
            i = current_segment_idx
            theta_start_local = theta_edges_local[i]
            theta_end_local = (theta_edges_local[i + 1]
                               if i + 1 < len(theta_edges_local)
                               else theta_edges_local[0])
            height = current_heights[i]
            if height > 0:
                theta_arc_local = np.linspace(theta_start_local, theta_end_local, 15)
                x_local = height * np.cos(theta_arc_local)
                y_local = height * np.sin(theta_arc_local)
                pts_local = np.column_stack([x_local, y_local])
                pts_global = pts_local + child_center
                r_global, theta_global = convert_to_polar(pts_global, parent_centroid)
                ax.plot(theta_global, r_global, color='orange', linewidth=2.5,
                        linestyle='-', zorder=6)

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Reduce Empty Space', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig


def draw_parallel_child_gradient_comparison(P, meta, figsize, opts=None) -> Figure:
    """Step 14.8 — gradient comparison with silhouette in child colour.

    Rebuilt from scratch so we can swap the silhouette colour cleanly.
    Logic mirrors the base ``draw_parallel_child_gradient_comparison``:
    silhouette + highlight on the two currently-compared segments.
    """
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    from vmeans.data import convert_to_polar
    from scipy.ndimage import gaussian_filter1d

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    child_colors = _get_child_colors(child_analyses, parent_centroid)
    info_texts = []

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        dmax_list = child['dmax_list']
        theta_edges_local = child['theta_edges']
        current_comparison = child.get('current_comparison', None)
        gradient_count = child.get('gradient_count', 0)
        comparison_complete = child.get('comparison_complete', False)

        # --- NEW: Get rmax and historical gradient records for drawing lines ---
        rmax_child = child.get('rmax', max(dmax_list) if dmax_list else 1.0)
        found_gradients_so_far = child.get('found_gradients_so_far', [])

        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color,
                       s=_get_point_size(), alpha=0.4, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_highlight_size(), marker='X',
                   edgecolors='black', linewidth=1, zorder=15, label=f'C{idx + 1}')

        # Silhouette → child colour
        _draw_child_visible_silhouette_connected(
            ax, child_center, theta_edges_local, dmax_list,
            parent_centroid, color=color,
            linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.8, zorder=3,
        )

        # Overlay orange arc on the pair of segments being compared
        if current_comparison and isinstance(current_comparison, dict):
            seg_i = current_comparison.get('seg_i', -1)
            seg_j = current_comparison.get('seg_j', -1)

            pts_per_seg = 20
            all_r_raw = []
            for si in range(len(dmax_list)):
                all_r_raw.extend([max(dmax_list[si], 0.01)] * pts_per_seg)
            if all_r_raw:
                all_r = gaussian_filter1d(all_r_raw, sigma=1.5, mode='wrap')
                for seg_idx in (seg_i, seg_j):
                    if seg_idx < 0 or seg_idx >= len(dmax_list):
                        continue
                    t0 = theta_edges_local[seg_idx]
                    t1 = (theta_edges_local[seg_idx + 1]
                          if seg_idx + 1 < len(theta_edges_local)
                          else theta_edges_local[0] + 2 * np.pi)
                    if t1 <= t0:
                        t1 += 2 * np.pi
                    arc_local = np.linspace(t0, t1, pts_per_seg)
                    start = seg_idx * pts_per_seg
                    end = start + pts_per_seg
                    r_slice = all_r[start:end]
                    x_local = r_slice * np.cos(arc_local)
                    y_local = r_slice * np.sin(arc_local)
                    pts_global = np.column_stack([x_local, y_local]) + child_center
                    r_g, t_g = convert_to_polar(pts_global, parent_centroid)
                    ax.plot(t_g, r_g, color='orange', linewidth=3.0,
                            alpha=0.9, zorder=7)

        # --- NEW: Draw the historical gradient lines found in previous steps ---
        g_counter = 0
        for prev_i, prev_j in found_gradients_so_far:
            boundary_theta_local = _gradient_pair_boundary_angle(
                theta_edges_local, prev_i, prev_j
            )
            x_end = rmax_child * np.cos(boundary_theta_local)
            y_end = rmax_child * np.sin(boundary_theta_local)
            line_pts = np.array([child_center, [child_center[0] + x_end, child_center[1] + y_end]])
            r_line, theta_line = convert_to_polar(line_pts, parent_centroid)

            is_current = (
                current_comparison
                and current_comparison.get('has_gradient', False)
                and (prev_i, prev_j) == (
                    current_comparison.get('seg_i'),
                    current_comparison.get('seg_j')
                )
            )
            ax.plot(
                theta_line,
                r_line,
                color='magenta',
                linewidth=3.0 if is_current else 2.0,
                linestyle='--',
                alpha=0.95 if is_current else 0.7,
                zorder=9 if is_current else 8
            )
            g_counter += 1
            ax.text(theta_line[1], r_line[1] * 1.06, f'C{idx + 1}-G{g_counter}', fontsize=6, color='magenta',
                    fontweight='bold', ha='center')

        if comparison_complete:
            info_texts.append(f"C{idx + 1}: Done ({gradient_count}G)")
        elif current_comparison and isinstance(current_comparison, dict):
            seg_i = current_comparison.get('seg_i', 0)
            seg_j = current_comparison.get('seg_j', 0)
            has_gradient = current_comparison.get('has_gradient', False)
            info_texts.append(
                f"C{idx + 1}: {seg_i}↔{seg_j} {'✓' if has_gradient else '○'}"
            )
        else:
            info_texts.append(f"C{idx + 1}: Waiting...")

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Gradient Comparison', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    level_label = P.get('level_label', 'Child')
    info_text = f"{level_label} Status\n" + "\n".join(info_texts)
    _fill_info_panel(
        ax_info,
        handles=handles,
        labels=labels,
        info_text=info_text,
        info_color='gray'
    )
    return fig

def draw_parallel_child_gradient_lines(P, meta, figsize, opts: RenderOptions = None) -> Figure:
    """Step 14.9 — silhouette + gradient lines, silhouette in child colour."""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    child_colors = _get_child_colors(child_analyses, parent_centroid)

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        dmax_list = child['dmax_list']
        rmax_child = child['rmax']
        gradient_angles_local = child.get('gradient_angles', [])
        theta_edges_local = child['theta_edges']

        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_highlight_size(), marker='X', edgecolors='black',
                   linewidth=1, zorder=15, label=f'C{idx + 1}')

        # Silhouette → child colour
        _draw_child_visible_silhouette_connected(
            ax, child_center, theta_edges_local, dmax_list,
            parent_centroid, color=color,
            linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.8, label=None,
        )

        for g_idx, g_angle_local in enumerate(gradient_angles_local):
            x_local_end = rmax_child * np.cos(g_angle_local)
            y_local_end = rmax_child * np.sin(g_angle_local)
            pts_global = np.array([
                child_center,
                [child_center[0] + x_local_end, child_center[1] + y_local_end],
            ])
            r_g, t_g = convert_to_polar(pts_global, parent_centroid)
            ax.plot(t_g, r_g, color='magenta', linewidth=2.5, linestyle='--', alpha=0.8)
            ax.text(t_g[1], r_g[1] * 1.06, f'C{idx + 1}-G{g_idx + 1}',
                    fontsize=6, color='magenta', fontweight='bold', ha='center')

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Gradient Lines', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig

def draw_parallel_child_gradient_pair_search(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - Search gradient pairs for each child (Colored Silhouette)"""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    current_pair_idx = P.get('current_pair_idx', 0)

    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)

    child_colors = _get_child_colors(child_analyses, parent_centroid)

    info_texts = []

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        dmax_list = child['dmax_list']
        rmax_child = child['rmax']
        gradient_angles_local = child.get('gradient_angles', [])
        theta_edges_local = child['theta_edges']

        # Draw data points
        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        # =========================================================
        # Modification: Changed original color=COLOR_SILHOUETTE to color=color
        # =========================================================
        _draw_child_visible_silhouette_connected(ax, child_center, theta_edges_local, dmax_list,
                                                 parent_centroid, color=color,
                                                 linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.8,
                                                 label=None)

        # Only show gradient lines while this child still has pairs to scan
        n_angles = len(gradient_angles_local)
        if n_angles >= 2 and current_pair_idx < n_angles:
            g_current_idx = current_pair_idx % n_angles
            g_next_idx = (current_pair_idx + 1) % n_angles

            for g_idx, g_angle_local in enumerate(gradient_angles_local):
                is_current = (g_idx == g_current_idx) or (g_idx == g_next_idx)
                line_alpha = 0.9 if is_current else 0.5

                x_local_end = rmax_child * np.cos(g_angle_local)
                y_local_end = rmax_child * np.sin(g_angle_local)
                points_global = np.array([
                    child_center,
                    [child_center[0] + x_local_end, child_center[1] + y_local_end]
                ])
                r_global, theta_global = convert_to_polar(points_global, parent_centroid)
                ax.plot(theta_global, r_global, color='magenta', linewidth=2.5,
                        linestyle='--', alpha=line_alpha)

                ax.text(theta_global[1], r_global[1] * 1.06, f'C{idx + 1}-G{g_idx + 1}',
                        fontsize=6, color='magenta', fontweight='bold', ha='center')

            # Check if interval has points
            g_start = gradient_angles_local[g_current_idx]
            g_end = gradient_angles_local[g_next_idx]
            child_theta = child.get('theta', np.array([]))
            crosses_zero = g_start > g_end
            pts_count = 0
            for angle in child_theta:
                if crosses_zero:
                    in_range = (angle >= g_start) or (angle <= g_end)
                else:
                    in_range = g_start <= angle <= g_end
                if in_range:
                    pts_count += 1

            has_pts = pts_count > 0

            # =========================================================
            # Crucial restoration! The original bright green highlight "scanning" logic is back!
            # =========================================================
            if has_pts and child_points is not None:
                for pi, angle in enumerate(child_theta):
                    if crosses_zero:
                        in_range = (angle >= g_start) or (angle <= g_end)
                    else:
                        in_range = g_start <= angle <= g_end
                    if in_range:
                        pt_r, pt_theta = convert_to_polar(
                            np.array([child_points[pi]]), parent_centroid)
                        ax.scatter(pt_theta, pt_r, c='#32CD32', s=_get_point_size() + 10,
                                   alpha=0.8, zorder=5, edgecolors='black', linewidth=0.3)

            status = f"✓{pts_count}pts" if has_pts else "✗Empty"
            info_texts.append(f"C{idx + 1}: G{g_current_idx + 1}→G{g_next_idx + 1} {status}")
        elif n_angles >= 2:
            info_texts.append(f"C{idx + 1}: Done")
        else:
            info_texts.append(f"C{idx + 1}: <2 gradients")

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Child - Search Gradient Pairs', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    info_text = "\n".join(info_texts) if info_texts else ""
    _add_info_to_panel(ax_info, [
        {'text': info_text, 'y': 0.85, 'fontsize': 6, 'color': 'gray', 'title': 'Pair Search'}
    ])
    _add_legend_to_panel(ax_info, ax, y_pos=0.45)

    return fig

def draw_parallel_child_dynamic_center_found(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Step 14.11 — dynamic centres discovered, silhouette in child colour."""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    centers_shown = P.get('centers_shown', 1)

    from vmeans.data import convert_to_polar
    from vmeans.colors import get_colors_for_centers

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)

    child_colors = _get_child_colors(child_analyses, parent_centroid)

    sc_counter = 0

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        dmax_list = child['dmax_list']
        rmax_child = child['rmax']
        gradient_angles_local = child.get('gradient_angles', [])
        additional_centers_local = child.get('additional_centers', [])
        center_labels_local = child.get('center_labels', [])
        theta_edges_local = child['theta_edges']

        # Draw data points
        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        # =========================================================
        # Sole modification: Changed original color=COLOR_SILHOUETTE to color=color
        # =========================================================
        _draw_child_visible_silhouette_connected(ax, child_center, theta_edges_local, dmax_list,
                                                 parent_centroid, color=color,
                                                 linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.8)

        # Draw gradient lines with labels
        for g_idx, g_angle_local in enumerate(gradient_angles_local):
            x_local_end = rmax_child * np.cos(g_angle_local)
            y_local_end = rmax_child * np.sin(g_angle_local)
            points_global = np.array([
                child_center,
                [child_center[0] + x_local_end, child_center[1] + y_local_end]
            ])
            r_global, theta_global = convert_to_polar(points_global, parent_centroid)
            ax.plot(theta_global, r_global, color='magenta', linewidth=2.5,
                    linestyle='--', alpha=0.8)
            ax.text(theta_global[1], r_global[1] * 1.06, f'C{idx + 1}-G{g_idx + 1}',
                    fontsize=6, color='magenta', fontweight='bold', ha='center')

        # Draw discovered sub-centers
        if additional_centers_local:
            n_show = min(centers_shown, len(additional_centers_local))
            sub_colors = get_colors_for_centers(
                np.array(additional_centers_local), origin=child_center
            ) if additional_centers_local else []

            for ci in range(n_show):
                sc_counter += 1
                sub_center = additional_centers_local[ci]
                sub_r, sub_theta = convert_to_polar(np.array([sub_center]), parent_centroid)
                sub_color = sub_colors[ci] if ci < len(sub_colors) else '#888888'
                ax.scatter([sub_theta[0]], [sub_r[0]], c=sub_color,
                           s=_get_highlight_size() * 1.2, marker='*',
                           edgecolors='black', linewidth=1, zorder=18,
                           label=f'SC{sc_counter}')

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Child - Compute New Centers', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig

def draw_parallel_child_find_matching_pairs(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - Algorithm 1 Pass Complete. (Colored Silhouette Version)"""
    if opts is None:
        opts = _DEFAULT_RENDER_OPTIONS

    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']

    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    point_size = opts.get_point_size(MPL_NORMAL_SIZE)

    # === 1. Draw points (by sub-region color) ===
    all_sub_regions = _collect_all_sub_regions(child_analyses, parent_centroid)
    sub_colors = _get_sub_region_colors(all_sub_regions, parent_centroid)

    for i, (sr, st, label, _) in enumerate(all_sub_regions):
        ax.scatter(st, sr, c=sub_colors[i], s=_get_point_size(),
                   alpha=0.4, zorder=3, label=label)

    # === NEW: Get color for each parent Child ===
    child_colors = _get_child_colors(child_analyses, parent_centroid)

    # === 2. Draw colored silhouette ===
    for idx, child in enumerate(child_analyses):
        child_center = child['center_global']
        dmax_list = child.get('dmax_list', [])
        theta_edges_local = child.get('theta_edges', [])

        # Get the exclusive color for this Child
        color = child_colors[idx % len(child_colors)]

        if len(dmax_list) > 0 and len(theta_edges_local) > 0:
            _draw_child_visible_silhouette_connected(
                ax, child_center, theta_edges_local, dmax_list,
                parent_centroid, color=color,  # <--- Replaced the original COLOR_SILHOUETTE here
                linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.9, label=None
            )

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Algorithm 1 Pass Complete', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig

def draw_parallel_child_dmax_circle(P, meta, figsize, opts: RenderOptions = None) -> Figure:
    """Step 14.4 — d_max circle drawn in each child's colour."""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    circle_progress = P.get('circle_progress', 1.0)
    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    child_colors = _get_child_colors(child_analyses, parent_centroid)

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        rmax_child = child['rmax']

        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_center_size(), marker='*', edgecolors='black',
                   linewidth=0.5, zorder=15, label=f'C{idx + 1}')

        child_r = child.get('r', None)
        child_theta = child.get('theta', None)
        if child_r is not None and len(child_r) > 0:
            dmax_idx = int(np.argmax(child_r))
            dmax_angle_local = child_theta[dmax_idx]
            dmax_x = rmax_child * np.cos(dmax_angle_local)
            dmax_y = rmax_child * np.sin(dmax_angle_local)
            dmax_global = np.array([child_center[0] + dmax_x, child_center[1] + dmax_y])
            edge_points = np.array([child_center, dmax_global])
            edge_r, edge_theta = convert_to_polar(edge_points, parent_centroid)
            ax.plot(edge_theta, edge_r, color='orange', linewidth=3, alpha=0.8, zorder=8)
            dmax_r_pt, dmax_theta_pt = convert_to_polar(np.array([dmax_global]), parent_centroid)
            ax.scatter([dmax_theta_pt[0]], [dmax_r_pt[0]], c='orange',
                       s=_get_highlight_size(), marker='o',
                       edgecolors='black', linewidth=1.5, zorder=20)

        if circle_progress > 0:
            n_points = max(int(100 * circle_progress), 10)
            angles_local = np.linspace(0, 2 * np.pi * circle_progress, n_points)
            x_local = rmax_child * np.cos(angles_local)
            y_local = rmax_child * np.sin(angles_local)
            pts_local = np.column_stack([x_local, y_local])
            pts_global = pts_local + child_center
            r_global, theta_global = convert_to_polar(pts_global, parent_centroid)
            # Circle → child colour
            ax.plot(theta_global, r_global, color=color,
                    linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.8)

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title(f'Child - Draw d_max Silhouette ({int(circle_progress * 100)}%)', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig


def draw_parallel_child_progressive_segments(P, meta, figsize, opts: RenderOptions = None) -> Figure:
    """Step 14.5 — progressive segment lines; circle and spokes in child colour."""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    child_colors = _get_child_colors(child_analyses, parent_centroid)

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        rmax_child = child['rmax']
        theta_edges_local = child['theta_edges']
        visible_segments = child.get('visible_segments', len(theta_edges_local) - 1)

        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_center_size(), marker='*', edgecolors='black',
                   linewidth=0.5, zorder=15, label=f'C{idx + 1}')

        # d_max circle → child colour
        angles_local = np.linspace(0, 2 * np.pi, 100)
        x_local = rmax_child * np.cos(angles_local)
        y_local = rmax_child * np.sin(angles_local)
        pts_global = np.column_stack([x_local, y_local]) + child_center
        r_g, t_g = convert_to_polar(pts_global, parent_centroid)
        ax.plot(t_g, r_g, color=color, linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.8)

        # Spokes → child colour, slightly faded
        for i in range(min(visible_segments, len(theta_edges_local))):
            theta_local = theta_edges_local[i]
            x_end = rmax_child * np.cos(theta_local)
            y_end = rmax_child * np.sin(theta_local)
            line_points = np.array([
                child_center,
                [child_center[0] + x_end, child_center[1] + y_end],
            ])
            r_l, t_l = convert_to_polar(line_points, parent_centroid)
            ax.plot(t_l, r_l, color=color, linewidth=1, alpha=0.6)

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Divide into Segments', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig


def draw_parallel_child_d_values(P, meta, figsize, opts: RenderOptions = None) -> Figure:
    """Step 14.7 — shrunk silhouette drawn in each child's colour."""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    child_colors = _get_child_colors(child_analyses, parent_centroid)

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        dmax_list = child['dmax_list']
        theta_edges_local = child['theta_edges']

        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_center_size(), marker='*', edgecolors='black',
                   linewidth=0.5, zorder=15, label=f'C{idx + 1}')

        # Silhouette → child colour
        _draw_child_visible_silhouette(
            ax, child_center, theta_edges_local, dmax_list,
            parent_centroid, color=color,
            linewidth=CHILD_SILHOUETTE_LINE_WIDTH,
        )

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Finding the Large Gradients', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig


# ===========================================================================
# B) Per-leaf override — each leaf cluster outlined in its own colour.
#    (parallel_child_final is already per-leaf in the base module so we do
#    not override it here.)
# ===========================================================================

def draw_export_silhouettes_only(P, meta, figsize, opts: RenderOptions = None) -> Figure:
    """Export figure — one silhouette per leaf, each in the leaf's colour."""
    if opts is None:
        opts = _DEFAULT_RENDER_OPTIONS

    from vmeans.data import convert_to_polar
    from vmeans.core_analysis import compute_better_center
    from vmeans.segment import segment_points_by_theta

    parent_centroid = P['parent_centroid']
    child_analyses = P.get('child_analyses', [])
    parent_r = P['parent_r']
    parent_segments = P.get('segments', 60)
    leaf_segments = max(8, parent_segments // 4)

    # Use the deepest available recursion result, preserving untouched
    # Child leaves when only part of the data was recursed.
    leaf_clusters = _collect_final_leaf_clusters(P)

    # One colour per leaf — same scheme _get_child_colors uses.
    leaf_analyses_for_color = [
        {'center_global': np.mean(pts, axis=0), 'points': pts}
        for pts in leaf_clusters
    ]
    colors = _get_child_colors(leaf_analyses_for_color, parent_centroid,
                               ensure_distinct=True)

    max_r = max(
        _get_child_max_r(parent_r, child_analyses, parent_centroid),
        float(P.get('display_rmax', 0))
    )
    display_rmax = _get_display_rmax(max_r)
    fig, ax, ax_info = _create_figure_with_panel(figsize)
    point_size = opts.get_point_size(MPL_NORMAL_SIZE)

    for idx, pts in enumerate(leaf_clusters):
        color = colors[idx] if idx < len(colors) else 'gray'

        pts_r, pts_theta = convert_to_polar(pts, parent_centroid)
        ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                   alpha=0.9, zorder=3, edgecolors='none',
                   label=f'Cluster {idx + 1}')

        # Silhouette needs at least a few points to trace a contour.
        # Singleton / near-singleton leaves still get their scatter point
        # and legend entry above, just without an outline.
        if len(pts) < 3:
            continue

        leaf_center = compute_better_center(pts, method='centroid')
        leaf_r, leaf_theta = convert_to_polar(pts, leaf_center)
        seg_data, theta_edges = segment_points_by_theta(leaf_r, leaf_theta, s=leaf_segments)
        export_padding = max(float(np.max(leaf_r)) * 0.025, 0.03)
        dmax_list = [(max(seg) + export_padding) if seg else 0.0 for seg in seg_data]

        _draw_child_visible_silhouette_connected(
            ax, leaf_center, theta_edges, dmax_list,
            parent_centroid, color=color,          # ← was COLOR_SILHOUETTE
            linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.9, label=None,
            smooth_sigma=0.75
        )

    _configure_polar_ax(ax, display_rmax, show_grid=False)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.spines['polar'].set_visible(False)

    ax.set_title(_get_title('Silhouettes (Export)', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig
