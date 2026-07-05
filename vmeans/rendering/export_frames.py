from .base import *
from . import base as _base
from .base import (
    _DEFAULT_RENDER_OPTIONS,
    _configure_polar_ax,
    _create_figure_with_panel,
    _draw_child_visible_silhouette,
    _draw_child_visible_silhouette_connected,
    _fill_info_panel,
    _get_center_size,
    _get_child_colors,
    _get_child_max_r,
    _get_display_rmax,
    _get_highlight_size,
    _get_point_size,
    _get_title,
    _set_global_ranges,
)
from .recursive_frames import (
    _collect_all_sub_regions,
    _collect_final_leaf_clusters,
    _get_sub_region_colors,
)

def draw_export_silhouettes_only(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """
    Export-quality figure: each leaf cluster gets its own silhouette.
    """
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

    # Compose the global final result. At depth 2 this keeps untouched
    # Child leaves and replaces only recursed leaves with Grandchild leaves.
    leaf_clusters = _collect_final_leaf_clusters(P)

    # === 2. Build fake analyses for coloring ===
    leaf_analyses_for_color = []
    for pts in leaf_clusters:
        leaf_analyses_for_color.append({
            'center_global': np.mean(pts, axis=0),
            'points': pts
        })
    colors = _get_child_colors(leaf_analyses_for_color, parent_centroid, ensure_distinct=True)

    # === 3. Setup figure ===
    max_r = max(
        _get_child_max_r(parent_r, child_analyses, parent_centroid),
        float(P.get('display_rmax', 0))
    )
    display_rmax = _get_display_rmax(max_r)
    fig, ax, ax_info = _create_figure_with_panel(figsize)
    point_size = opts.get_point_size(MPL_NORMAL_SIZE)

    # === 4. For each leaf: color points + compute & draw silhouette ===
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
            parent_centroid, color=COLOR_SILHOUETTE,
            linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.9, label=None,
            smooth_sigma=0.75
        )

    # Clean export styling
    _configure_polar_ax(ax, display_rmax, show_grid=False)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.spines['polar'].set_visible(False)

    ax.set_title(_get_title('Silhouettes (Export)', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)

    return fig


# ============================================================================
# Additional Parallel Child Frame Renderers
# ============================================================================

def draw_parallel_child_raw_data(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - Step 14.1: Show raw data (no center)"""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']

    from vmeans.data import convert_to_polar

    # This frame only shows the original points. Using the child silhouette
    # envelope here makes the same data shrink as soon as recursion starts.
    max_r = np.max(parent_r) if len(parent_r) > 0 else 1.0
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)

    child_colors = _get_child_colors(child_analyses, parent_centroid)

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_points = child.get('points', None)

        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.7, zorder=2, label=f'Cluster {idx + 1}')

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Recurse', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_parallel_child_center(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - Step 14.2: Show center points"""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']

    from vmeans.data import convert_to_polar

    # Child centers lie within the parent data extent; reserve the expanded
    # child envelope only for later frames that actually draw child circles.
    max_r = np.max(parent_r) if len(parent_r) > 0 else 1.0
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)

    child_colors = _get_child_colors(child_analyses, parent_centroid)

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)

        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.5, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_center_size(), marker='*', edgecolors='black',
                   linewidth=0.5, zorder=15, label=f'C{idx + 1}')

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Compute Center Points', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_parallel_child_highlight_dmax(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - Step 14.3: Highlight d_max points"""
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
        r_child = child.get('r', None)

        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_center_size(), marker='*', edgecolors='black',
                   linewidth=0.5, zorder=15, label=f'C{idx + 1}')

        if r_child is not None and len(r_child) > 0 and child_points is not None:
            dmax_idx_child = np.argmax(r_child)
            dmax_point = child_points[dmax_idx_child]
            dmax_r, dmax_theta = convert_to_polar(np.array([dmax_point]), parent_centroid)
            ax.scatter([dmax_theta[0]], [dmax_r[0]], c='orange',
                       s=_get_highlight_size(), marker='o', edgecolors='black',
                       linewidth=1.5, zorder=20)

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Highlight d_max Points', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_parallel_child_dmax_edge(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - Step 14.4: Draw edge from center to d_max"""
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
        r_child = child.get('r', None)

        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_center_size(), marker='*', edgecolors='black',
                   linewidth=0.5, zorder=15, label=f'C{idx + 1}')

        if r_child is not None and len(r_child) > 0 and child_points is not None:
            dmax_idx_child = np.argmax(r_child)
            dmax_point = child_points[dmax_idx_child]

            dmax_r, dmax_theta = convert_to_polar(np.array([dmax_point]), parent_centroid)

            ax.scatter([dmax_theta[0]], [dmax_r[0]], c='orange',
                       s=_get_highlight_size(), marker='o', edgecolors='black',
                       linewidth=1.5, zorder=20)

            ax.plot([child_theta_parent[0], dmax_theta[0]],
                    [child_r_parent[0], dmax_r[0]],
                    color='orange', linewidth=3, zorder=8)

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Draw Edge to d_max', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_parallel_child_dmax_circle(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - Step 14.4: Draw d_max circles (animated)"""
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
            dmax_idx = np.argmax(child_r)
            dmax_angle_local = child_theta[dmax_idx]
            dmax_x = rmax_child * np.cos(dmax_angle_local)
            dmax_y = rmax_child * np.sin(dmax_angle_local)
            dmax_global = np.array([child_center[0] + dmax_x, child_center[1] + dmax_y])
            edge_points = np.array([child_center, dmax_global])
            edge_r, edge_theta = convert_to_polar(edge_points, parent_centroid)
            ax.plot(edge_theta, edge_r, color='orange', linewidth=3, alpha=0.8, zorder=8)
            dmax_r_pt, dmax_theta_pt = convert_to_polar(np.array([dmax_global]), parent_centroid)
            ax.scatter([dmax_theta_pt[0]], [dmax_r_pt[0]], c='orange', s=_get_highlight_size(), marker='o',
                       edgecolors='black', linewidth=1.5, zorder=20)
        if circle_progress > 0:
            n_points = max(int(100 * circle_progress), 10)
            angles_local = np.linspace(0, 2 * np.pi * circle_progress, n_points)

            x_local = rmax_child * np.cos(angles_local)
            y_local = rmax_child * np.sin(angles_local)

            points_local = np.column_stack([x_local, y_local])
            points_global = points_local + child_center

            r_global, theta_global = convert_to_polar(points_global, parent_centroid)
            ax.plot(theta_global, r_global, color=COLOR_SILHOUETTE, linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.8)

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title(f'Child - Draw d_max Silhouette ({int(circle_progress * 100)}%)', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_parallel_child_progressive_segments(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - Step 14.5: Progressive segment lines"""
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

        # Draw d_max circle
        angles_local = np.linspace(0, 2 * np.pi, 100)
        x_local = rmax_child * np.cos(angles_local)
        y_local = rmax_child * np.sin(angles_local)
        points_local = np.column_stack([x_local, y_local])
        points_global = points_local + child_center
        r_global, theta_global = convert_to_polar(points_global, parent_centroid)
        ax.plot(theta_global, r_global, color=COLOR_SILHOUETTE, linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.8)

        # Draw segment lines progressively
        for i in range(min(visible_segments, len(theta_edges_local))):
            theta_local = theta_edges_local[i]
            x_end = rmax_child * np.cos(theta_local)
            y_end = rmax_child * np.sin(theta_local)

            line_points = np.array([
                child_center,
                [child_center[0] + x_end, child_center[1] + y_end]
            ])

            r_line, theta_line = convert_to_polar(line_points, parent_centroid)
            ax.plot(theta_line, r_line, color=COLOR_SILHOUETTE, linewidth=1, alpha=0.6)

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Divide into Segments', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_parallel_child_d_values(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - Step 14.7: Finding the Gradients"""
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
        segment_points = child['segment_points']
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

        # Draw connected + smoothed silhouette for this child
        _draw_child_visible_silhouette(ax, child_center, theta_edges_local, dmax_list,
                                       parent_centroid, color=COLOR_SILHOUETTE,
                                       linewidth=CHILD_SILHOUETTE_LINE_WIDTH)

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Finding the Large Gradients', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


# ============================================================================
# Error Handling
# ============================================================================

def draw_unknown_frame(name: str, meta: dict, figsize) -> Figure:
    """Unknown frame type -- plain text on a blank page."""
    fig = Figure(figsize=figsize, dpi=RENDER_DPI, facecolor='white')
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, f'Unknown frame type: {name}', ha='center', va='center',
            fontsize=16, color='gray')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig


def draw_error_frame(name: str, error: str, figsize) -> Figure:
    """Error frame -- displays the exception message."""
    fig = Figure(figsize=figsize, dpi=RENDER_DPI, facecolor='white')
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.6, f'Error rendering: {name}', ha='center', va='center',
            fontsize=14, color='red')
    ax.text(0.5, 0.4, error[:80], ha='center', va='center', fontsize=10, color='gray')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig


# ============================================================================
# High-Resolution Export (No Title)
# ============================================================================

def export_final_frame_hd(frame, output_path: str, figsize=(12, 12), dpi=150,
                          global_rmax=None, render_options: RenderOptions = None):
    """Export the final frame as a high-resolution PNG without title."""
    from vmeans.data import convert_to_polar
    import matplotlib.pyplot as plt

    P = frame.payload
    name = frame.name
    opts = render_options if render_options is not None else _DEFAULT_RENDER_OPTIONS

    if global_rmax:
        centroid = P.get('centroid', P.get('parent_centroid', None))
        _set_global_ranges(global_rmax, centroid)

    if name == 'parallel_child_final':
        fig = _draw_final_frame_no_title_child(P, figsize, opts)
    else:
        fig = _draw_final_frame_no_title_generic(P, figsize, opts)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                pad_inches=0.15, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Exported final frame to: {output_path}")


def _draw_final_frame_no_title_child(P: dict, figsize, opts: RenderOptions) -> Figure:
    """Draw child final frame without title"""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    point_size = opts.get_point_size(MPL_NORMAL_SIZE) * 1.5

    all_sub_regions = _collect_all_sub_regions(child_analyses, parent_centroid)
    sub_colors = _get_sub_region_colors(all_sub_regions, parent_centroid)

    for i, (sr, st, label, _) in enumerate(all_sub_regions):
        ax.scatter(st, sr, c=sub_colors[i], s=_get_point_size(),
                   alpha=0.85, zorder=2, label=label)

    _configure_polar_ax(ax, display_rmax)
    ax.set_title('')
    _fill_info_panel(ax_info)
    return fig


def _draw_final_frame_no_title_generic(P: dict, figsize, opts: RenderOptions) -> Figure:
    """Draw generic final frame without title (fallback)"""
    from vmeans.data import convert_to_polar

    r = P.get('r', P.get('parent_r', np.array([])))
    theta = P.get('theta', P.get('parent_theta', np.array([])))

    if len(r) == 0:
        fig = Figure(figsize=figsize, dpi=RENDER_DPI, facecolor='white')
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No data available for export', ha='center', va='center')
        ax.axis('off')
        return fig

    max_r = np.max(r)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)

    point_size = opts.get_point_size(MPL_NORMAL_SIZE) * 1.5
    ax.scatter(theta, r, c='steelblue', s=point_size, alpha=0.8)

    _configure_polar_ax(ax, display_rmax)
    ax.set_title('')
    _fill_info_panel(ax_info, )
    return fig


# ============================================================================
# GIF Animation Generation
# ============================================================================

def generate_circle_animation_gif(
        r: np.ndarray,
        theta: np.ndarray,
        rmax: float,
        dmax_idx: int,
        output_path: str = None,
        n_frames: int = 30,
        duration_ms: int = 50,
        figsize: tuple = (7, 7),
        global_rmax: float = None
) -> str:
    """Generate a GIF animation of the d_max circle being drawn."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.figure import Figure

    os.makedirs(GIF_OUTPUT_DIR, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(GIF_OUTPUT_DIR, 'circle_animation.gif')
    elif not os.path.dirname(output_path):
        output_path = os.path.join(GIF_OUTPUT_DIR, output_path)

    if global_rmax and _base._GLOBAL_RMAX is None:
        _base._GLOBAL_RMAX = global_rmax
    display_rmax = _get_display_rmax(rmax)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    ax.set_facecolor('white')

    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(), alpha=0.5,
               label='Points', zorder=2)
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(),
               label='Center', zorder=10)
    ax.scatter([theta[dmax_idx]], [r[dmax_idx]], c=COLOR_DMAX, s=_get_highlight_size(),
               label=f'd={rmax:.1f}', zorder=15)
    ax.plot([0, theta[dmax_idx]], [0, r[dmax_idx]], COLOR_DMAX, linewidth=3, alpha=0.8, zorder=8)

    _configure_polar_ax(ax, display_rmax, show_grid=True)

    circle_line, = ax.plot([], [], COLOR_SILHOUETTE, linewidth=SILHOUETTE_LINE_WIDTH, linestyle='-',
                           label='Silhouette', zorder=5)
    pen_marker, = ax.plot([], [], 'o', color='white', markersize=10,
                          markeredgecolor=COLOR_SILHOUETTE, markeredgewidth=3, zorder=20)
    title = ax.set_title('Step 5\nDrawing d_max Silhouette...', fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)

    start_angle = theta[dmax_idx]

    def init():
        circle_line.set_data([], [])
        pen_marker.set_data([], [])
        return circle_line, pen_marker, title

    def animate(frame):
        progress = (frame + 1) / n_frames
        sweep_angle = 2 * np.pi * progress

        arc_theta = np.linspace(start_angle, start_angle + sweep_angle, int(100 * progress) + 2)
        arc_r = [rmax] * len(arc_theta)
        circle_line.set_data(arc_theta, arc_r)

        current_angle = start_angle + sweep_angle
        pen_marker.set_data([current_angle], [rmax])

        title.set_text(f'Step 5\nDrawing d_max Silhouette ({int(progress * 100)}%)')

        return circle_line, pen_marker, title

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=duration_ms, blit=True
    )

    anim.save(output_path, writer='pillow', fps=1000 // duration_ms, dpi=RENDER_DPI)
    plt.close(fig)

    print(f"✅ GIF saved to: {output_path}")
    return output_path


def generate_child_circle_animation_gif(
        child_analyses: list,
        parent_centroid: np.ndarray,
        parent_r: np.ndarray,
        output_path: str = None,
        n_frames: int = 30,
        duration_ms: int = 50,
        figsize: tuple = (7, 7)
) -> str:
    """Generate a GIF animation of d_max circles being drawn for all child clusters."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from vmeans.data import convert_to_polar

    os.makedirs(GIF_OUTPUT_DIR, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(GIF_OUTPUT_DIR, 'child_circle_animation.gif')
    elif not os.path.dirname(output_path):
        output_path = os.path.join(GIF_OUTPUT_DIR, output_path)

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    child_colors = _get_child_colors(child_analyses, parent_centroid)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    ax.set_facecolor('white')

    child_data = []
    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        rmax_child = child['rmax']
        r_child = child.get('r', None)

        if r_child is not None and len(r_child) > 0:
            dmax_idx_child = np.argmax(r_child)
            theta_child = child.get('theta', None)
            if theta_child is not None:
                start_angle_local = theta_child[dmax_idx_child]
            else:
                start_angle_local = 0
        else:
            start_angle_local = 0

        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_center_size(), marker='*', edgecolors='black',
                   linewidth=0.5, zorder=15, label=f'C{idx + 1}')

        child_data.append({
            'color': color,
            'center': child_center,
            'center_r': child_r_parent[0],
            'center_theta': child_theta_parent[0],
            'rmax': rmax_child,
            'start_angle_local': start_angle_local
        })

    _configure_polar_ax(ax, display_rmax, show_grid=True)

    circle_lines = []
    pen_markers = []
    for idx, cd in enumerate(child_data):
        line, = ax.plot([], [], cd['color'], linewidth=CHILD_SILHOUETTE_LINE_WIDTH, linestyle='-', zorder=5)
        pen, = ax.plot([], [], 'o', color='white', markersize=6,
                       markeredgecolor=cd['color'], markeredgewidth=2, zorder=20)
        circle_lines.append(line)
        pen_markers.append(pen)

    title = ax.set_title('Step 14.5\nChild - Drawing d_max Silhouettes...', fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)

    def init():
        for line, pen in zip(circle_lines, pen_markers):
            line.set_data([], [])
            pen.set_data([], [])
        return circle_lines + pen_markers + [title]

    def animate(frame):
        progress = (frame + 1) / n_frames
        sweep_angle = 2 * np.pi * progress

        for idx, (cd, line, pen) in enumerate(zip(child_data, circle_lines, pen_markers)):
            start_angle_local = cd['start_angle_local']
            n_points = max(int(50 * progress), 5)
            angles_local = np.linspace(start_angle_local, start_angle_local + sweep_angle, n_points)

            x_local = cd['rmax'] * np.cos(angles_local)
            y_local = cd['rmax'] * np.sin(angles_local)
            points_local = np.column_stack([x_local, y_local])
            points_global = points_local + cd['center']

            r_global, theta_global = convert_to_polar(points_global, parent_centroid)
            line.set_data(theta_global, r_global)

            current_angle_local = start_angle_local + sweep_angle
            pen_x = cd['rmax'] * np.cos(current_angle_local)
            pen_y = cd['rmax'] * np.sin(current_angle_local)
            pen_global = np.array([[pen_x, pen_y]]) + cd['center']
            pen_r, pen_theta = convert_to_polar(pen_global, parent_centroid)
            pen.set_data([pen_theta[0]], [pen_r[0]])

        title.set_text(f'Step 14.5\nChild - Drawing d_max Silhouettes ({int(progress * 100)}%)')

        return circle_lines + pen_markers + [title]

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=duration_ms, blit=True
    )

    anim.save(output_path, writer='pillow', fps=1000 // duration_ms, dpi=RENDER_DPI)
    plt.close(fig)

    print(f"✅ Child circle GIF saved to: {output_path}")
    return output_path
