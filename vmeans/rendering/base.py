import os
import numpy as np
from matplotlib.figure import Figure

from matplotlib.gridspec import GridSpec


# ============================================================================
# GridSpec-based layout system for the plot + info panel
# ============================================================================

def _create_figure_with_panel(figsize=(8, 8), plot_ratio=0.72):
    """
    Create a Figure with a right-side info panel.

    The polar axis is forced to be square (non-square frames leave
    ugly whitespace). Axis position is fixed -- no auto-shifting.
    """
    fig = Figure(figsize=figsize, dpi=RENDER_DPI, facecolor='white')

    # Right-side panel dimensions
    info_x0, info_w = 0.88, 0.12
    info_y0, info_h = 0.04, 0.84

    # Usable area for the main plot (right edge = info panel left edge)
    plot_x0 = 0.06  # 6% left margin
    plot_y0 = 0.08  # 8% bottom margin, room for tick labels
    plot_x1 = info_x0  # right edge at 88%
    plot_y1 = 0.92  # 8% top margin, room for title

    avail_w = plot_x1 - plot_x0  # 0.82
    avail_h = plot_y1 - plot_y0  # 0.84

    # Polar axis must be square; pick the shorter side of the available area
    side = min(avail_w, avail_h)

    # Center the square region inside the available area
    base_x = plot_x0 + (avail_w - side) * 0.5
    base_y = plot_y0 + (avail_h - side) * 0.5

    ax_plot = fig.add_axes([base_x, base_y, side, side], projection='polar')
    ax_plot.set_facecolor('white')

    # Right-side info panel
    ax_info = fig.add_axes([info_x0, info_y0, info_w, info_h])
    ax_info.axis('off')
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    return fig, ax_plot, ax_info


def _fill_info_panel(ax_info, handles=None, labels=None, info_text=None, info_color='gray'):
    """
    Populate the right-side info panel with legend handles and/or text.
    A small inset margin keeps bboxes from being clipped.
    """
    # Legend in the upper-right corner (5% inset so the frame isn't clipped)
    if handles and labels:
        ax_info.legend(
            handles, labels,
            loc='upper right',
            bbox_to_anchor=(0.95, 1.0),  # slight inset to avoid edge clipping
            fontsize=6,
            frameon=True,
            framealpha=0.9,
            borderpad=0.4
        )

    # Info block in the lower-right corner (5% inset for same reason)
    if info_text:
        ax_info.text(
            0.95, 0.05,  # inset from panel edge
            info_text,
            va='bottom',
            ha='right',
            multialignment='right',  # align multi-line text to the right
            fontsize=6,
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor='white',
                alpha=0.95,
                edgecolor=info_color,
                linewidth=1.0
            ),
            transform=ax_info.transAxes
        )


def _create_figure_simple(figsize=(8, 8)):
    """
    Create a simple Figure. Still uses the panel layout, but the
    info panel is left empty.
    """
    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)
    ax.set_facecolor('white')
    # Fill info panel (extend as needed)
    _fill_info_panel(ax_info, )
    return fig, ax


def _add_info_to_panel(ax_info, info_blocks):
    """
    Place multiple info blocks in the info panel.

    Args:
        ax_info: The info panel Axes.
        info_blocks: List of dicts, each with:
            {
                'text': str,           # content
                'y': float,            # vertical position (0-1, bottom to top)
                'fontsize': int,       # font size
                'color': str,          # border color
                'title': str           # optional heading
            }
    """
    for block in info_blocks:
        text = block.get('text', '')
        y = block.get('y', 0.5)
        fontsize = block.get('fontsize', 6)  # 6pt works well for dense info
        edge_color = block.get('color', 'gray')
        title = block.get('title', None)

        # Optional heading above the info box
        if title:
            ax_info.text(0.95, y + 0.08, title, fontsize=fontsize + 1,
                         fontweight='bold', va='bottom', ha='right',
                         transform=ax_info.transAxes)

        # The info box itself
        ax_info.text(0.95, y, text, fontsize=fontsize,
                     va='top', ha='right',
                     multialignment='right',  # right-align multi-line text
                     transform=ax_info.transAxes,
                     bbox=dict(boxstyle='round,pad=0.2',
                               facecolor='white', alpha=0.95,
                               edgecolor=edge_color, linewidth=1.0))


def _add_legend_to_panel(ax_info, ax_plot, y_pos=0.95):
    """Copy the plot legend into the info panel (right-aligned, with margin)."""
    handles, labels = ax_plot.get_legend_handles_labels()
    if handles:
        ax_info.legend(handles, labels, loc='upper right',
                       bbox_to_anchor=(0.95, y_pos), fontsize=6,  # inset to avoid clipping
                       framealpha=0.9, borderaxespad=0)


from matplotlib.patches import Circle
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import color module for angle-based coloring
from vmeans.colors import (
    get_cluster_colors,
    get_colors_for_centers,
    get_cluster_palettes,
    get_color_by_angle  # kept for backward compatibility
)

# GIF output directory
GIF_OUTPUT_DIR = './GIF'

# Import StepFrame type
try:
    from vmeans.animation import StepFrame
except ImportError:
    from dataclasses import dataclass as dc_dataclass, field


    @dc_dataclass(frozen=True)
    class StepFrame:
        name: str
        payload: Dict[str, Any]
        meta: Optional[Dict[str, Any]] = field(default_factory=dict)

CHART_SIZE = 8  # inches
RENDER_DPI = 150  # Higher DPI for sharper rendering

# Point sizes -- kept small so silhouettes stay readable
MPL_NORMAL_SIZE = 25  # small enough so silhouette outlines stay visible
MPL_CENTER_SIZE = 55
MPL_HIGHLIGHT_SIZE = 55

# Font sizes -- tuned for recursive mode where legend entries pile up
FONT_SIZE_TITLE = 10  # title in the middle, unaffected by panel
FONT_SIZE_LEGEND = 6  # smaller to handle many entries in recursive mode
FONT_SIZE_LABEL = 6  # smaller to prevent text overflow in the panel

# Global point size multiplier (set by RenderOptions)
_GLOBAL_POINT_MULTIPLIER = 1.0

# Minimum point size to prevent negative/zero values
MIN_POINT_SIZE = 1.0


def _get_point_size(base_size=MPL_NORMAL_SIZE, reduction=0):
    """Get current point size with global multiplier applied

    Args:
        base_size: Base point size (default: MPL_NORMAL_SIZE)
        reduction: Amount to subtract from the result (will be scaled by multiplier)

    Returns:
        Point size, guaranteed to be at least MIN_POINT_SIZE
    """
    size = base_size * _GLOBAL_POINT_MULTIPLIER - reduction * _GLOBAL_POINT_MULTIPLIER
    return max(size, MIN_POINT_SIZE)


def _get_center_size(reduction=0):
    """Get center point size with global multiplier applied"""
    size = MPL_CENTER_SIZE * _GLOBAL_POINT_MULTIPLIER - reduction * _GLOBAL_POINT_MULTIPLIER
    return max(size, MIN_POINT_SIZE)


def _get_highlight_size(reduction=0):
    """Get highlight point size with global multiplier applied"""
    size = MPL_HIGHLIGHT_SIZE * _GLOBAL_POINT_MULTIPLIER - reduction * _GLOBAL_POINT_MULTIPLIER
    return max(size, MIN_POINT_SIZE)


# Colors - MODIFIED: Data points now BLACK
COLOR_DATA_POINTS = '#000000'  # black (was lightblue)
COLOR_DATA_POINTS_GRAY = '#555555'  # darker gray for better visibility
COLOR_CENTER = '#ff0000'  # red
COLOR_DMAX = '#ffa500'  # orange
COLOR_SILHOUETTE = '#2196F3'  # Blue color for silhouette
SILHOUETTE_LINE_WIDTH = 4.0  # Parent silhouette linewidth
CHILD_SILHOUETTE_LINE_WIDTH = 2.0  # Child/grandchild silhouette = parent - 2
SILHOUETTE_SMOOTH_SIGMA = 1.5      # Gaussian sigma for corner rounding (0 = sharp, 1.5 = default, 3.0 = very smooth)
COLOR_SEGMENT_LINE = '#4682b4'  # steelblue
COLOR_GRID = '#d3d3d3'  # lightgray


# ============================================================================
# Smoothing helper for corner rounding
# ============================================================================

def _smooth_radii(r_values, sigma=None):
    """
    Apply Gaussian smoothing to radius values for corner rounding.
    Uses wrap mode so the closed path stays continuous at the seam.

    Args:
        r_values: array of radius values along the silhouette path
        sigma: Gaussian standard deviation (larger = rounder corners)

    Returns:
        smoothed radius array (same length)
    """
    if sigma is None:
        sigma = SILHOUETTE_SMOOTH_SIGMA
    if sigma <= 0 or len(r_values) < 5:
        return np.array(r_values, dtype=float)

    r = np.array(r_values, dtype=float)

    # Build Gaussian kernel
    kernel_size = max(int(sigma * 6), 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    x = np.arange(kernel_size) - half
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    # Wrap-pad for circular (closed) path
    pad = min(half, len(r) - 1)
    r_padded = np.concatenate([r[-pad:], r, r[:pad]])
    r_smooth = np.convolve(r_padded, kernel, mode='same')
    r_smooth = r_smooth[pad:pad + len(r)]

    # Ensure no negative radii
    r_smooth = np.maximum(r_smooth, 0.0)

    return r_smooth


def _gradient_pair_boundary_angle(theta_edges, seg_i, seg_j):
    """Return the shared edge angle for an ordered adjacent segment pair."""
    if len(theta_edges) < 2:
        return 0.0

    segment_count = len(theta_edges) - 1
    if segment_count > 0 and (seg_i + 1) % segment_count == seg_j:
        edge_index = seg_j if seg_j != 0 else segment_count
    else:
        edge_index = max(seg_i, seg_j) % len(theta_edges)
    return theta_edges[edge_index]


def _compute_segment_dmax(r, theta, theta_edges):
    """
    Compute the d_max (maximum radius) within each angular segment.

    Args:
        r: radial distances of all points
        theta: angular positions of all points
        theta_edges: boundary angles between segments

    Returns:
        segment_dmax: list of max-radius values, one per segment
    """
    n_segments = len(theta_edges) - 1
    segment_dmax = []

    for i in range(n_segments):
        theta_start = theta_edges[i]
        theta_end = theta_edges[i + 1] if i + 1 < len(theta_edges) else theta_edges[0] + 2 * np.pi

        # Find points in this segment
        if theta_end > theta_start:
            mask = (theta >= theta_start) & (theta < theta_end)
        else:
            # Crosses 0/2π boundary
            mask = (theta >= theta_start) | (theta < theta_end)

        segment_r = r[mask]
        if len(segment_r) > 0:
            segment_dmax.append(float(np.max(segment_r)))
        else:
            segment_dmax.append(0.0)

    return segment_dmax


def _draw_visible_silhouette(ax, theta_edges, segment_dmax, color=None, linewidth=None, alpha=0.9, zorder=5,
                             label='Silhouette', smooth_sigma=None):
    """
    Draw the visible silhouette as a connected, closed outline with
    smoothed corners.  Each segment sits at its d_max radius; adjacent
    segments are joined and the whole path is closed.

    Args:
        ax: polar matplotlib Axes
        theta_edges: boundary angles
        segment_dmax: d_max for each segment
        color: line color (default blue)
        linewidth: line width
        alpha: opacity
        zorder: draw order
        label: legend label
        smooth_sigma: Gaussian sigma for corner rounding (None = use default)
    """
    if color is None:
        color = COLOR_SILHOUETTE
    if linewidth is None:
        linewidth = SILHOUETTE_LINE_WIDTH
    if smooth_sigma is None:
        smooth_sigma = SILHOUETTE_SMOOTH_SIGMA

    if len(segment_dmax) == 0:
        return

    n_segments = len(segment_dmax)

    # Build a single connected path
    all_theta = []
    all_r = []

    for i in range(n_segments):
        theta_start = theta_edges[i]
        theta_end = theta_edges[i + 1] if i + 1 < len(theta_edges) else theta_edges[0] + 2 * np.pi
        height = max(segment_dmax[i], 0.01)  # keep degenerate arcs small

        # Guard against non-monotonic edges (float rounding or unsorted input)
        if theta_end <= theta_start:
            theta_end += 2 * np.pi

        n_pts = 20
        theta_arc = np.linspace(theta_start, theta_end, n_pts)
        r_arc = np.full(n_pts, height)

        all_theta.extend(theta_arc)
        all_r.extend(r_arc)

    if len(all_theta) == 0:
        return

    all_theta = np.array(all_theta)
    all_r = np.array(all_r)

    # Smooth BEFORE closing: wrap-mode sees a clean circular array,
    # so both ends are treated symmetrically and the seam is gap-free.
    if smooth_sigma > 0:
        all_r = _smooth_radii(all_r, sigma=smooth_sigma)

    # Close AFTER smoothing using all_theta[0] + 2*pi to keep the angle
    # sequence monotonically increasing and avoid a backward connecting line.
    all_theta = np.append(all_theta, all_theta[0] + 2 * np.pi)
    all_r = np.append(all_r, all_r[0])

    ax.plot(all_theta, all_r, color, linewidth=linewidth, linestyle='-',
            label=label, alpha=alpha, zorder=zorder)


def _draw_connected_silhouette_with_gradients(ax, theta_edges, segment_dmax, gradient_angles=None,
                                              color=None, linewidth=None, alpha=0.9, zorder=5, label='Silhouette',
                                              smooth_sigma=None):
    """
    Draw a connected silhouette with smoothed corners.  Arcs are joined
    together, the path is closed, and corners are rounded by a Gaussian
    filter.  At each gradient angle a spoke line is drawn from the
    silhouette back to the center.

    Args:
        ax: polar matplotlib Axes
        theta_edges: boundary angles
        segment_dmax: d_max per segment
        gradient_angles: angles where spokes should be drawn to the center
        color: line color (default blue)
        linewidth: line width
        alpha: opacity
        zorder: draw order
        label: legend label
        smooth_sigma: Gaussian sigma for corner rounding (None = use default)
    """
    if color is None:
        color = COLOR_SILHOUETTE
    if linewidth is None:
        linewidth = SILHOUETTE_LINE_WIDTH
    if gradient_angles is None:
        gradient_angles = []
    if smooth_sigma is None:
        smooth_sigma = SILHOUETTE_SMOOTH_SIGMA

    if len(segment_dmax) == 0:
        return

    n_segments = len(segment_dmax)

    # Build the complete connected silhouette path
    all_theta = []
    all_r = []

    for i in range(n_segments):
        theta_start = theta_edges[i]
        theta_end = theta_edges[i + 1] if i + 1 < len(theta_edges) else theta_edges[0] + 2 * np.pi
        height = max(segment_dmax[i], 0.01)

        # Guard against non-monotonic edges (float rounding or unsorted input)
        if theta_end <= theta_start:
            theta_end += 2 * np.pi

        theta_arc = np.linspace(theta_start, theta_end, 30)
        r_arc = np.full(30, height)
        all_theta.extend(theta_arc)
        all_r.extend(r_arc)

    if len(all_theta) == 0:
        return

    all_theta = np.array(all_theta)
    all_r = np.array(all_r)

    # Smooth BEFORE closing (same reason as _draw_visible_silhouette)
    if smooth_sigma > 0:
        all_r = _smooth_radii(all_r, sigma=smooth_sigma)

    # Close AFTER smoothing, keeping angle sequence monotonically increasing
    all_theta = np.append(all_theta, all_theta[0] + 2 * np.pi)
    all_r = np.append(all_r, all_r[0])

    ax.plot(all_theta, all_r, color, linewidth=linewidth, linestyle='-',
            label=label, alpha=alpha, zorder=zorder)

    # Draw spoke lines from the silhouette to the center at each gradient angle
    for g_angle in gradient_angles:
        r_at_gradient = 0
        for i, t_edge in enumerate(theta_edges):
            if abs(t_edge - g_angle) < 0.01 or abs(t_edge - g_angle - 2 * np.pi) < 0.01:
                left_idx = (i - 1) % n_segments
                right_idx = i % n_segments
                r_at_gradient = max(segment_dmax[left_idx] if left_idx < len(segment_dmax) else 0,
                                    segment_dmax[right_idx] if right_idx < len(segment_dmax) else 0)
                break

        if r_at_gradient > 0:
            ax.plot([g_angle, g_angle], [0, r_at_gradient], color,
                    linewidth=linewidth, linestyle='-', alpha=alpha, zorder=zorder - 1)


def _draw_child_visible_silhouette(ax, child_center, theta_edges_local, segment_dmax, parent_centroid,
                                   color=None, linewidth=None, alpha=0.9, zorder=5, label=None,
                                   smooth_sigma=None):
    """
    Draw a child cluster's visible silhouette as a connected, smoothed
    outline in parent polar coordinates.
    """
    from vmeans.data import convert_to_polar

    if color is None:
        color = COLOR_SILHOUETTE
    if linewidth is None:
        linewidth = CHILD_SILHOUETTE_LINE_WIDTH
    if smooth_sigma is None:
        smooth_sigma = SILHOUETTE_SMOOTH_SIGMA

    if len(segment_dmax) == 0:
        return

    n_segments = len(segment_dmax)

    # Build connected path in local coordinates
    all_x = []
    all_y = []

    for i in range(n_segments):
        theta_start_local = theta_edges_local[i]
        theta_end_local = theta_edges_local[i + 1] if i + 1 < len(theta_edges_local) else theta_edges_local[0] + 2 * np.pi
        height = max(segment_dmax[i], 0.01)

        # Guard against non-monotonic edges (float rounding or unsorted input)
        if theta_end_local <= theta_start_local:
            theta_end_local += 2 * np.pi

        n_pts = 20
        theta_arc_local = np.linspace(theta_start_local, theta_end_local, n_pts)
        x_local = height * np.cos(theta_arc_local)
        y_local = height * np.sin(theta_arc_local)

        all_x.extend(x_local)
        all_y.extend(y_local)

    if len(all_x) == 0:
        return

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Smooth in local polar coordinates BEFORE closing, then convert back to Cartesian
    if smooth_sigma > 0:
        local_r = np.sqrt(all_x**2 + all_y**2)
        local_theta = np.arctan2(all_y, all_x)
        local_r = _smooth_radii(local_r, sigma=smooth_sigma)
        all_x = local_r * np.cos(local_theta)
        all_y = local_r * np.sin(local_theta)

    # Close AFTER smoothing
    all_x = np.append(all_x, all_x[0])
    all_y = np.append(all_y, all_y[0])

    # Convert to parent polar coordinates
    points_global = np.column_stack([all_x, all_y]) + child_center
    r_global, theta_global = convert_to_polar(points_global, parent_centroid)

    # Preserve continuity when the outline crosses the polar 0/2π seam.
    # Without this, wrapped angles create long radial chords across the plot.
    theta_global = np.unwrap(theta_global)

    ax.plot(theta_global, r_global, color=color, linewidth=linewidth, linestyle='-',
            label=label, alpha=alpha, zorder=zorder)


def _draw_child_visible_silhouette_connected(ax, child_center, theta_edges_local, segment_dmax, parent_centroid,
                                             color=None, linewidth=None, alpha=0.9, zorder=5, label=None,
                                             smooth_sigma=None):
    """
    Draw a child cluster's silhouette as a connected, smoothed outline
    in parent polar coordinates.  Used in steps 14.10 / 15.10.
    """
    from vmeans.data import convert_to_polar

    if color is None:
        color = COLOR_SILHOUETTE
    if linewidth is None:
        linewidth = CHILD_SILHOUETTE_LINE_WIDTH
    if smooth_sigma is None:
        smooth_sigma = SILHOUETTE_SMOOTH_SIGMA

    if len(segment_dmax) == 0:
        return

    n_segments = len(segment_dmax)
    all_x = []
    all_y = []

    for i in range(n_segments):
        theta_start_local = theta_edges_local[i]
        theta_end_local = theta_edges_local[i + 1] if i + 1 < len(theta_edges_local) else theta_edges_local[0] + 2 * np.pi
        height = max(segment_dmax[i], 0.01)

        # Guard against non-monotonic edges (float rounding or unsorted input)
        if theta_end_local <= theta_start_local:
            theta_end_local += 2 * np.pi

        theta_arc_local = np.linspace(theta_start_local, theta_end_local, 20)
        x_local = height * np.cos(theta_arc_local)
        y_local = height * np.sin(theta_arc_local)

        all_x.extend(x_local)
        all_y.extend(y_local)

    if len(all_x) == 0:
        return

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Smooth in local polar coordinates BEFORE closing
    if smooth_sigma > 0:
        local_r = np.sqrt(all_x**2 + all_y**2)
        local_theta = np.arctan2(all_y, all_x)
        local_r = _smooth_radii(local_r, sigma=smooth_sigma)
        all_x = local_r * np.cos(local_theta)
        all_y = local_r * np.sin(local_theta)

    # Close AFTER smoothing
    all_x = np.append(all_x, all_x[0])
    all_y = np.append(all_y, all_y[0])

    # Convert closed path to parent polar coordinates
    points_global = np.column_stack([all_x, all_y]) + child_center
    r_global, theta_global = convert_to_polar(points_global, parent_centroid)

    # Preserve the connected outline across the polar 0/2π seam.  Wrapped
    # angles otherwise produce long radial chords across the final frame.
    theta_global = np.unwrap(theta_global)

    ax.plot(theta_global, r_global, color=color, linewidth=linewidth, linestyle='-',
            label=label, alpha=alpha, zorder=zorder)


DEFAULT_SILHOUETTE_WIDTH = 1.5

# ============================================================================
# Shared layout constants
# ============================================================================
LAYOUT_LEFT = 0.02
LAYOUT_RIGHT = 0.90  # main plot right edge
LAYOUT_BOTTOM = 0.04
LAYOUT_TOP = 0.88  # main plot top edge (leaves room for title)
LEGEND_ANCHOR = (1.00, 1.00)
LEGEND_FONTSIZE = 6
INFO_X = 0.90  # info panel x-offset


# ============================================================================
# [DEPRECATED] Old layout system -- superseded by GridSpec + _fill_info_panel.
# Kept for reference only; do not call.
# ============================================================================
def _apply_standard_layout(fig, ax, legend_handles=None, legend_labels=None,
                           info_text=None, info_color='gray', show_legend=True):
    """
    [DEPRECATED] Old unified layout -- replaced by _create_figure_with_panel +
    _fill_info_panel. Do NOT call this; use the new system instead.
    """
    raise DeprecationWarning("Use _create_figure_with_panel + _fill_info_panel instead")


# ============================================================================
# Render Options - User-configurable display settings
# ============================================================================

@dataclass
class RenderOptions:
    """User-configurable rendering options for animation frames"""
    point_size: float = 1.0  # Multiplier for point size (0.5 = half, 2.0 = double)
    silhouette_width: float = 2.0  # Line width for silhouette boundaries
    show_legend: bool = True  # Show legend on final frame
    show_silhouettes: bool = True # Show silhouette outlines on final frame (default OFF)
    show_gradients: bool = False  # Show gradient lines on final frame (default OFF)
    show_centers: bool = True  # Show center markers on final frame
    simplified_final: bool = True  # Use simplified final view (colored points only)
    zoom_factor: float = 1.0  # Zoom factor (0.5 = zoom in 2x, 1.5 = zoom out)
    # Hidden option: export final frame with silhouettes only (no title, no gradients)
    export_final_silhouettes: bool = True  # Default OFF, enable for paper figures

    def get_point_size(self, base_size: float = MPL_NORMAL_SIZE) -> float:
        """Calculate actual point size"""
        return base_size * self.point_size

    def get_silhouette_width(self, base_width: float = DEFAULT_SILHOUETTE_WIDTH) -> float:
        """Calculate actual silhouette width"""
        return base_width * self.silhouette_width


# Global default render options (can be overridden per call)
_DEFAULT_RENDER_OPTIONS = RenderOptions()


# ============================================================================
# Color allocation helpers (angle-based, Paletton-style)
# ============================================================================

def _get_child_colors(child_analyses: list, parent_centroid: np.ndarray = None,
                      ensure_distinct: bool = True) -> list:
    """
    Pick colors for child clusters using a Paletton-style scheme.

    Hue comes from each cluster center's polar angle relative to
    the parent centroid.  Saturation/value scale with point count
    (more points = more saturated).  When two hues land too close
    together they get nudged apart.

    Args:
        child_analyses: list of child analysis dicts
        parent_centroid: parent cluster center (origin for angles)
        ensure_distinct: push apart colors that are too similar

    Returns:
        list of color strings, one per child
    """
    n = len(child_analyses)
    if n == 0:
        return []

    try:
        # Gather center coordinates
        centers = np.array([c['center_global'] for c in child_analyses])

        # Gather point counts (if available)
        point_counts = []
        for c in child_analyses:
            if 'points' in c and c['points'] is not None:
                point_counts.append(len(c['points']))
            elif 'n_points' in c:
                point_counts.append(c['n_points'])
            else:
                point_counts.append(100)  # sensible default

        # Delegate to the colors module
        if parent_centroid is not None:
            return get_colors_for_centers(centers, origin=parent_centroid,
                                          point_counts=point_counts,
                                          ensure_distinct=ensure_distinct)
        else:
            return get_colors_for_centers(centers, point_counts=point_counts,
                                          ensure_distinct=ensure_distinct)

    except (KeyError, TypeError):
        # fallback
        return get_cluster_colors(n, ensure_distinct=ensure_distinct)


# Global ranges (anti-jitter)
_GLOBAL_RMAX = None
_GLOBAL_CENTROID = None
_GLOBAL_ZOOM = 1.0
_FRAME_DISPLAY_RMAX = None
_GLOBAL_AXES_SHIFT = (0.0, 0.0)  # (shift_x, shift_y) in normalized coords


def _reset_global_ranges():
    """Reset global ranges"""
    global _GLOBAL_RMAX, _GLOBAL_CENTROID, _GLOBAL_ZOOM
    global _FRAME_DISPLAY_RMAX, _GLOBAL_AXES_SHIFT
    _GLOBAL_RMAX = None
    _GLOBAL_CENTROID = None
    _GLOBAL_ZOOM = 1.0
    _FRAME_DISPLAY_RMAX = None
    _GLOBAL_AXES_SHIFT = (0.0, 0.0)


def _compute_axes_shift(r, theta):
    """
    Intentionally does nothing.  Auto-shifting the polar axis to
    "center" the data turned out to cause clipping and blank regions.
    The origin must stay dead-center.
    """
    global _GLOBAL_AXES_SHIFT
    _GLOBAL_AXES_SHIFT = (0.0, 0.0)
    return


def _set_global_ranges(rmax, centroid=None, zoom_factor=None):
    """Set global ranges"""
    global _GLOBAL_RMAX, _GLOBAL_CENTROID, _GLOBAL_ZOOM
    if _GLOBAL_RMAX is None:
        _GLOBAL_RMAX = rmax
    if _GLOBAL_CENTROID is None and centroid is not None:
        _GLOBAL_CENTROID = centroid.copy()
    if zoom_factor is not None:
        _GLOBAL_ZOOM = zoom_factor


def _get_display_rmax(rmax, zoom_factor=None):
    """Get display rmax with zoom factor applied

    Args:
        rmax: Base rmax value
        zoom_factor: Zoom multiplier (0.5 = zoom in 2x, 1.5 = zoom out)
                     If None, uses global _GLOBAL_ZOOM
    """
    global _GLOBAL_RMAX, _GLOBAL_ZOOM, _FRAME_DISPLAY_RMAX
    candidates = [rmax]
    if _GLOBAL_RMAX:
        candidates.append(_GLOBAL_RMAX)
    if _FRAME_DISPLAY_RMAX:
        candidates.append(_FRAME_DISPLAY_RMAX)
    base_rmax = max(candidates)
    zoom = zoom_factor if zoom_factor is not None else _GLOBAL_ZOOM
    return base_rmax * zoom


def _get_child_max_r(parent_r, child_analyses, parent_centroid, grandchild_analyses=None):
    """Compute the true max_r needed for child steps, accounting for the outermost silhouette edges."""
    max_r = np.max(parent_r) if len(parent_r) > 0 else 1.0

    for child in child_analyses:
        child_center = child.get('center_global')
        rmax_child = child.get('rmax', 0)
        if child_center is not None and rmax_child > 0:
            dist_to_parent = np.sqrt((child_center[0] - parent_centroid[0]) ** 2 +
                                     (child_center[1] - parent_centroid[1]) ** 2)
            child_outer = dist_to_parent + rmax_child
            max_r = max(max_r, child_outer)

    return max_r


def _configure_polar_ax(ax, rmax, show_grid=False):
    """Configure polar axis - clean style: no outer frame, no angle labels, no grid"""
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, rmax * 1.05)
    ax.set_rticks([])

    # Hide outer frame
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('none')

    # Hide angle labels
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    ax.tick_params(axis='x', colors='none', labelcolor='none')

    # Grid is off by default
    if show_grid:
        ax.grid(True, alpha=0.3, color='lightgray', linestyle=':')
    else:
        ax.grid(False)


def _get_title(base_title: str, meta: dict) -> str:
    """Generate title - two lines format: Step on first line, description on second"""
    step = meta.get('step', '')
    description = meta.get('description', '')
    if step and description:
        return f"Step {step}\n{description}"
    elif step:
        return f"Step {step}\n{base_title}"
    elif description:
        return description
    return base_title


def get_frame_description(frame_name: str) -> str:
    """Get frame type description"""
    descriptions = {
        "show_raw_data": "Show raw data",
        "show_center": "Compute center point",
        "phase_transition": "Phase Transition",
        "init_center": "Compute center point",
        "cartesian_with_polar_grid": "Polar coordinate overlay",
        "highlight_dmax": "Highlight furthest point",
        "dmax_circle": "Draw edge to d_max",
        "dmax_circle_only": "Draw d_max silhouette",
        "dmax_circle_animating": "Drawing d_max silhouette...",
        "play_circle_gif": "Drawing d_max Silhouette (Animation)",
        "show_segments": "Divide into segments",
        "segment_shrinking": "Reduce Empty Space",
        "progressive_segments": "Divide region into segments",
        "progressive_shrinking": "Reduce Empty Space",
        "segmentation": "Algorithm 1 Pass Complete",
        "d_values": "Finding the Large Gradients",
        "gradient_scan_progress": "Gradient scan progress",
        "gradient_comparison": "Gradient comparison",
        "gradient_lines_final": "Final gradient lines",
        "dynamic_center_found": "Compute New Centers",
        "find_matching_gradient_pairs": "Find Matching Gradient Pairs",
        "gradient_pair_search": "Searching gradient pairs...",
        # Child cluster frame types
        "parallel_child_init": "Child clusters - Show centers",
        "parallel_child_segments": "Child clusters - Show segments",
        "parallel_child_shrinking": "Child clusters - Reduce Empty Space",
        "parallel_child_gradient_comparison": "Child clusters - Gradient comparison",
        "parallel_child_gradient_lines": "Child clusters - Gradient lines",
        "parallel_child_gradient_pair_search": "Child - Search Gradient Pairs",
        "parallel_child_dynamic_center_found": "Child - Compute New Centers",
        "parallel_child_find_matching_pairs": "Child - Algorithm 1 Pass Complete",
        "parallel_child_final": "Child clusters - Final analysis",
        # NEW: Additional child frame types
        "parallel_child_raw_data": "Recurse",
        "parallel_child_center": "Child - Compute Center Points",
        "parallel_child_highlight_dmax": "Child - Highlight d_max Points",
        "parallel_child_dmax_edge": "Child - Draw Edge to d_max",
        "parallel_child_dmax_circle": "Child - Draw d_max Silhouettes",
        "play_child_circle_gif": "Child - Drawing d_max Silhouettes (Animation)",
        "parallel_child_progressive_segments": "Child - Divide into Segments",
        "parallel_child_d_values": "Child - Finding the Large Gradients",
        # Grandchild frame types reuse the parallel child renderers.
        "parallel_grandchild_raw_data": "Recurse to Grandchild Level",
        "parallel_grandchild_center": "Grandchild - Compute Center Points",
        "parallel_grandchild_highlight_dmax": "Grandchild - Highlight d_max Points",
        "parallel_grandchild_dmax_edge": "Grandchild - Draw Edge to d_max",
        "parallel_grandchild_dmax_circle": "Grandchild - Draw d_max Silhouettes",
        "parallel_grandchild_progressive_segments": "Grandchild - Divide into Segments",
        "parallel_grandchild_shrinking": "Grandchild - Reduce Empty Space",
        "parallel_grandchild_d_values": "Grandchild - Finding the Large Gradients",
        "parallel_grandchild_gradient_comparison": "Grandchild - Gradient comparison",
        "parallel_grandchild_gradient_lines": "Grandchild - Gradient lines",
        "parallel_grandchild_gradient_pair_search": "Grandchild - Search Gradient Pairs",
        "parallel_grandchild_dynamic_center_found": "Grandchild - Compute New Centers",
        "parallel_grandchild_find_matching_pairs": "Grandchild - Algorithm 1 Pass Complete",
        "parallel_grandchild_final": "Grandchild - Final analysis",
        "no_gradient_boundaries": "No gradient boundaries",
    }
    return descriptions.get(frame_name, f"Step: {frame_name}")


# Export underscore helpers for internal split modules.
__all__ = [name for name in globals() if not name.startswith('__')]

# Import split frame renderers into this compatibility module.
# We intentionally copy underscore-prefixed helpers too: the historical
# pyqt_matplotlib_vis module exposed them internally, and the colored renderer
# imports several of those helpers.
def _merge_module_namespace(module):
    globals().update({
        name: value
        for name, value in vars(module).items()
        if not name.startswith('__')
    })


from . import parent_frames as _parent_frames
from . import child_frames as _child_frames
from . import recursive_frames as _recursive_frames
from . import export_frames as _export_frames
from . import dispatch as _dispatch

for _module in (
    _parent_frames,
    _child_frames,
    _recursive_frames,
    _export_frames,
    _dispatch,
):
    _merge_module_namespace(_module)

__all__ = [name for name in globals() if not name.startswith('__')]
