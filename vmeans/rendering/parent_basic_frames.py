from .base import *

def draw_show_raw_data(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """
    Step 1: Display raw data points in polar coordinates, without center point
    """
    if opts is None:
        opts = _DEFAULT_RENDER_OPTIONS

    points = P['points']
    centroid = P['centroid']
    r = P['r']
    theta = P['theta']
    rmax = P.get('rmax', np.max(r) if len(r) > 0 else 1.0)

    _compute_axes_shift(r, theta)

    _set_global_ranges(rmax, centroid)
    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Only draw data points, not center point - size controlled by opts
    point_size = opts.get_point_size(MPL_NORMAL_SIZE)
    ax.scatter(theta, r, c=COLOR_DATA_POINTS, s=point_size,
               alpha=0.7, zorder=2)

    # Basic configuration
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, display_rmax * 1.05)
    ax.set_rticks([])
    ax.grid(False)

    # Invisible placeholder - maintain layout stability
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('none')
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    ax.tick_params(axis='x', colors='none', labelcolor='none')

    ax.set_title(_get_title('Show Raw Data', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    # Show basic stats in the info panel
    info_text = f"Points: {len(points)}\nr_max: {rmax:.2f}\nDisplay: {display_rmax:.2f}"
    _fill_info_panel(ax_info, info_text=info_text, info_color='steelblue')

    return fig


def draw_show_center(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """
    Step 2: Display data points and computed center point
    """
    points = P['points']
    centroid = P['centroid']
    r = P['r']
    theta = P['theta']
    rmax = P.get('rmax', np.max(r) if len(r) > 0 else 1.0)

    _set_global_ranges(rmax, centroid)
    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Draw data points - NOW BLACK
    ax.scatter(theta, r, c=COLOR_DATA_POINTS, s=_get_point_size(),
               alpha=0.7, zorder=2)

    # Draw center point (in polar coordinates, center point is at origin r=0)
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(),
               marker='*', edgecolors='black', linewidth=0.5,
               zorder=10, label='Center')

    # Configure polar axis - same clean style as Step 1
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, display_rmax * 1.05)
    ax.set_rticks([])
    ax.grid(False)

    # Hide outer ring and angle labels
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('none')
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    ax.tick_params(axis='x', colors='none', labelcolor='none')

    ax.set_title(_get_title('Compute Center Point', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_phase_transition(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Phase transition marker frame"""
    phase_number = P.get('phase_number', 0)
    phase_name = P.get('phase_name', 'Unknown')
    phase_description = P.get('phase_description', '')

    phase_colors = {
        1: '#3498db', 2: '#2ecc71', 3: '#e74c3c',
        4: '#f39c12', 5: '#9b59b6', 6: '#1abc9c', 7: '#34495e',
    }
    color = phase_colors.get(phase_number, '#95a5a6')

    fig = Figure(figsize=figsize, dpi=RENDER_DPI, facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('white')

    ax.text(0.5, 0.7, f'PHASE {phase_number}', ha='center', va='center',
            fontsize=42, fontweight='bold', color=color)
    ax.text(0.5, 0.5, phase_name, ha='center', va='center',
            fontsize=26, fontweight='bold', color=color)
    ax.text(0.5, 0.3, phase_description, ha='center', va='center',
            fontsize=14, color='gray')

    # Phase transition is a special splash page -- no polar plot, no info panel
    return fig


def draw_init_center(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """
    Step 1: Clean without frame, but use invisible placeholder to lock layout size
    """
    points = P['points']
    centroid = P['centroid']
    r = P['r']
    theta = P['theta']
    rmax = P.get('rmax', np.max(r) if len(r) > 0 else 1.0)

    _set_global_ranges(rmax, centroid)
    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # 1. Draw data points - NOW BLACK
    ax.scatter(theta, r, c=COLOR_DATA_POINTS, s=_get_point_size(),
               alpha=0.7, label='Points', edgecolors='none', zorder=2)

    # 2. Draw center point
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(),
               zorder=10, label='Center', edgecolors='none')

    # 3. Basic configuration
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, display_rmax * 1.05)
    ax.set_rticks([])
    ax.grid(False)  # No grid display

    # 4. [Key technique] Invisible placeholder - enable frame and text but set to transparent
    #    This way tight_layout will reserve the same space as Step 2
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('none')  # Frame exists but invisible

    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    ax.tick_params(axis='x', colors='none', labelcolor='none')  # Text exists but invisible

    ax.set_title(_get_title('Compute Center Point', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_cartesian_with_polar_grid(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """
    Step 1.5: Custom grid + invisible placeholder to lock layout size
    """
    r = P['r']
    theta = P['theta']
    rmax = P['rmax']
    centroid = P['centroid']

    _set_global_ranges(rmax, centroid)
    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # 1. Draw custom radial rays (8 lines)
    n_rays = 8
    for angle in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        ax.plot([angle, angle], [0, display_rmax], color='lightgray',
                linewidth=1, linestyle=':', zorder=1)

    # 2. Draw custom concentric circles (6 circles)
    n_circles = 6
    circle_radii = np.linspace(0, display_rmax, n_circles + 1)[1:]
    for i, r_circle in enumerate(circle_radii):
        circle_theta = np.linspace(0, 2 * np.pi, 100)
        if i == n_circles - 1:
            # Outermost ring: orange dashed line
            ax.plot(circle_theta, [r_circle] * 100, color='orange',
                    linewidth=2, linestyle='--', zorder=3)
        else:
            ax.plot(circle_theta, [r_circle] * 100, color='lightgray',
                    linewidth=1, linestyle=':', zorder=1)

    # 3. Draw data points - NOW BLACK
    ax.scatter(theta, r, c=COLOR_DATA_POINTS, s=_get_point_size(),
               alpha=0.7, label='Points', zorder=5)

    # 4. Draw center point
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(),
               zorder=10, label='Center')

    # 5. Configure polar axis
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, display_rmax * 1.05)
    ax.set_rticks([])
    ax.grid(False)

    # 6. Invisible placeholder to lock size
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('none')
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    ax.tick_params(axis='x', colors='none', labelcolor='none')

    ax.set_title(_get_title('Polar Coordinate Overlay', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_highlight_dmax(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Step 3: Highlight d_max point"""
    r = P['r']
    theta = P['theta']
    rmax = P['rmax']
    dmax_idx = P['dmax_idx']
    dmax_value = P['dmax_value']

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Data points - NOW DARKER GRAY
    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.6, label='Points', zorder=2)

    # Center point
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(),
               zorder=10, label='Center')

    # d_max point
    ax.scatter([theta[dmax_idx]], [r[dmax_idx]], c=COLOR_DMAX, s=_get_highlight_size(),
               zorder=15, label=f'd={dmax_value:.1f}')

    # Configuration - keep clean
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, display_rmax * 1.05)
    ax.set_rticks([])
    ax.grid(False)

    # Hide outer ring and angle labels
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('none')
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    ax.tick_params(axis='x', colors='none', labelcolor='none')

    ax.set_title(_get_title('Highlight Furthest Point (d_max)', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    # Grab legend handles for the info panel
    handles, labels = ax.get_legend_handles_labels()

    # Build d_max detail text
    if len(r) > 0:
        dmax_idx = np.argmax(r)
        dmax_r = r[dmax_idx]
        dmax_theta = theta[dmax_idx]
        info_text = f"d_max:\n  dist: {dmax_r:.3f}\n  angle: {np.degrees(dmax_theta):.1f}°\n  idx: {dmax_idx}"
    else:
        info_text = "No data"

    # Show both legend and info in the panel
    _fill_info_panel(ax_info, handles=handles, labels=labels,
                     info_text=info_text, info_color='orange')

    return fig


def draw_dmax_circle(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Step 4: Draw edge from center to d_max (clean, no frame)"""
    r = P['r']
    theta = P['theta']
    rmax = P['rmax']
    dmax_idx = P['dmax_idx']

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Data points
    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.5, label='Points', zorder=2)
    # Center point
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(),
               zorder=10, label='Center')
    # d_max point
    ax.scatter([theta[dmax_idx]], [r[dmax_idx]], c=COLOR_DMAX, s=_get_highlight_size(),
               zorder=15, label=f'd={rmax:.1f}')
    # Ray: center -> d_max
    ax.plot([0, theta[dmax_idx]], [0, r[dmax_idx]], COLOR_DMAX, linewidth=3,
            label='→d_max', zorder=8)

    # Configuration - clean without frame
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, display_rmax * 1.05)
    ax.set_rticks([])
    ax.grid(False)

    # Hide outer ring and angle labels
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('none')
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    ax.tick_params(axis='x', colors='none', labelcolor='none')

    ax.set_title(_get_title('Draw Edge from Center to d_max', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_dmax_circle_only(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Step 5 (final): Draw complete d_max circle"""
    r = P['r']
    theta = P['theta']
    rmax = P['rmax']
    dmax_idx = P['dmax_idx']

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Data points
    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.5, label='Points', zorder=2)
    # Center point
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(),
               zorder=10, label='Center')
    # d_max point
    ax.scatter([theta[dmax_idx]], [r[dmax_idx]], c=COLOR_DMAX, s=_get_highlight_size(),
               zorder=15, label=f'd={rmax:.1f}')
    # Ray (faded)
    ax.plot([0, theta[dmax_idx]], [0, r[dmax_idx]], COLOR_DMAX, linewidth=3,
            alpha=0.8, zorder=8)
    # d_max circle - COMPLETE (Blue silhouette, thinner)
    circle_theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(circle_theta, [rmax] * 100, COLOR_SILHOUETTE, linewidth=SILHOUETTE_LINE_WIDTH, linestyle='-',
            label='Silhouette', zorder=5)

    # Configuration - clean without frame
    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Draw d_max Silhouette', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


# ============================================================================
# NEW: Animated circle drawing
# ============================================================================

def draw_dmax_circle_animating(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """
    NEW: Step 5 Animation - Draw d_max circle progressively
    Shows the circle being drawn from dmax point angle, sweeping around
    """
    r = P['r']
    theta = P['theta']
    rmax = P['rmax']
    dmax_idx = P['dmax_idx']
    progress = P.get('progress', 1.0)  # 0.0 to 1.0

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Data points
    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.5, label='Points', zorder=2)
    # Center point
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(),
               zorder=10, label='Center')
    # d_max point
    ax.scatter([theta[dmax_idx]], [r[dmax_idx]], c=COLOR_DMAX, s=_get_highlight_size(),
               zorder=15, label=f'd={rmax:.1f}')
    # Ray (faded)
    ax.plot([0, theta[dmax_idx]], [0, r[dmax_idx]], COLOR_DMAX, linewidth=3,
            alpha=0.8, zorder=8)

    # ANIMATED CIRCLE: start from dmax point angle, sweep around
    start_angle = theta[dmax_idx]
    sweep_angle = 2 * np.pi * progress

    # Create arc from start_angle to start_angle + sweep_angle
    arc_theta = np.linspace(start_angle, start_angle + sweep_angle, int(100 * progress) + 2)
    arc_r = [rmax] * len(arc_theta)

    ax.plot(arc_theta, arc_r, COLOR_SILHOUETTE, linewidth=SILHOUETTE_LINE_WIDTH, linestyle='-',
            label=f'Silhouette ({int(progress * 100)}%)', zorder=5)

    # Draw a "pen" indicator at the current drawing position
    current_angle = start_angle + sweep_angle
    ax.scatter([current_angle], [rmax], c='white', s=60, edgecolors=COLOR_SILHOUETTE,
               linewidth=2, zorder=20)

    # Configuration
    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title(f'Drawing d_max Silhouette...', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_show_segments(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Display all segment lines"""
    r = P['r']
    theta = P['theta']
    theta_edges = P['theta_edges']
    rmax = P['rmax']

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.5, label='Points', zorder=2)
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10, label='Center')

    # Blue silhouette circle
    circle_theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(circle_theta, [rmax] * 100, COLOR_SILHOUETTE, linewidth=SILHOUETTE_LINE_WIDTH, linestyle='-',
            label='Silhouette', alpha=0.8)

    for t_edge in theta_edges:
        ax.plot([t_edge, t_edge], [0, display_rmax * 1.1], color='gray',
                linewidth=1, linestyle=':', alpha=0.7)

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title(f'Divide into {len(theta_edges) - 1} Segments', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_progressive_segments(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Progressive segment line display - one by one"""
    r = P['r']
    theta = P['theta']
    theta_edges = P['theta_edges']
    rmax = P['rmax']
    visible_edges_count = P.get('visible_edges_count', len(theta_edges))
    dmax_idx = P.get('dmax_idx', 0)

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Data points
    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.5, label='Points', zorder=2)
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10, label='Center')

    # Blue silhouette circle
    circle_theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(circle_theta, [rmax] * 100, COLOR_SILHOUETTE, linewidth=SILHOUETTE_LINE_WIDTH, linestyle='-',
            label='Silhouette', alpha=0.8)

    # Progressive segment lines
    for i in range(min(visible_edges_count, len(theta_edges))):
        t_edge = theta_edges[i]
        ax.plot([t_edge, t_edge], [0, rmax], color=COLOR_SEGMENT_LINE,
                linewidth=1.5, linestyle='-', alpha=0.8)

    _configure_polar_ax(ax, display_rmax)

    total_segments = len(theta_edges) - 1
    ax.set_title(_get_title(f'Divide into Segments ({visible_edges_count}/{total_segments})', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_segment_shrinking(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Segment shrinking animation -- each segment's outer arc shrinks
    down to its actual d_max height. The silhouette updates accordingly.
    """
    r = P['r']
    theta = P['theta']
    theta_edges = P['theta_edges']
    rmax = P['rmax']
    current_heights = P.get('current_heights', [])
    segment_points = P.get('segment_points', [])
    current_segment_idx = P.get('current_segment_idx', 0)

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.4, zorder=2)
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10)

    # Draw the updated silhouette (connected + smoothed)
    _draw_visible_silhouette(ax, theta_edges, current_heights, label='Silhouette', zorder=5)

    # Highlight the segment currently being processed (orange arc)
    if current_segment_idx < len(current_heights):
        i = current_segment_idx
        theta_start = theta_edges[i]
        theta_end = theta_edges[i + 1] if i + 1 < len(theta_edges) else theta_edges[0]
        height = current_heights[i]

        if height > 0:
            theta_arc = np.linspace(theta_start, theta_end, 20)
            r_arc = [height] * 20
            ax.plot(theta_arc, r_arc, color='orange', linewidth=2.5, linestyle='-', zorder=6)

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Reduce Empty Space', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    # Fill info panel (extend as needed)
    _fill_info_panel(ax_info, )
    return fig


def draw_progressive_shrinking(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Progressive shrinking - same as segment_shrinking"""
    return draw_segment_shrinking(P, meta, figsize)


def draw_segmentation(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Final polar segmentation -- shows the completed silhouette."""
    r = P['r']
    theta = P['theta']
    theta_edges = P['theta_edges']
    rmax = P['rmax']
    dmax_list = P.get('dmax_list', [])
    segment_points = P.get('segment_points', [])

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.5, label='Points', zorder=2)
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10, label='Center')

    # Draw silhouette with consistent line width
    _draw_visible_silhouette(ax, theta_edges, dmax_list, label='Silhouette')

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Algorithm 1 Pass Complete', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_d_values(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Display the d-max value for each segment. Silhouette drawn with uniform width."""
    r = P['r']
    theta = P['theta']
    theta_edges = P['theta_edges']
    rmax = P['rmax']
    dmax_list = P.get('dmax_list', [])
    segment_points = P.get('segment_points', [])

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.4, zorder=2)
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10, label='Center')

    # Draw silhouette with consistent line width
    _draw_visible_silhouette(ax, theta_edges, dmax_list, label='Silhouette')

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Finding the Large Gradients', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_gradient_scan_progress(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Gradient boundary scan progress"""
    r = P['r']
    theta = P['theta']
    theta_edges = P['theta_edges']
    rmax = P['rmax']
    gradient_boundaries = P.get('gradient_boundaries', [])
    dmax_list = P.get('dmax_list', [])
    segment_points = P.get('segment_points', [])

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    ax.scatter(theta, r, c=COLOR_SEGMENT_LINE, s=_get_point_size(),
               alpha=0.7, label='Points')
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10, label='Center')

    for t_edge in theta_edges:
        ax.plot([t_edge, t_edge], [0, display_rmax], color='lightgray',
                linewidth=1, linestyle=':', alpha=0.5)

    # Draw segment borders (outer arc only, no spokes)
    if dmax_list and segment_points:
        for i in range(len(dmax_list)):
            theta_start = theta_edges[i]
            theta_end = theta_edges[i + 1] if i + 1 < len(theta_edges) else theta_edges[0]
            has_data = len(segment_points[i]) > 0
            linestyle = '-' if has_data else ':'
            height = dmax_list[i] if has_data else rmax

            if height > 0:
                theta_arc = np.linspace(theta_start, theta_end, 20)
                r_arc = [height] * 20
                ax.plot(theta_arc, r_arc, color='lightblue', linewidth=1, linestyle=linestyle)

    # Detected gradient boundaries
    for seg1, seg2 in gradient_boundaries:
        if abs(seg1 - seg2) == 1 or abs(seg1 - seg2) == len(theta_edges) - 2:
            boundary_theta = theta_edges[max(seg1, seg2)]
            ax.plot([boundary_theta, boundary_theta], [0, display_rmax * 1.1],
                    color='orange', linewidth=3, zorder=8)

    _configure_polar_ax(ax, display_rmax)

    scan_step = meta.get('scan_step', 1)
    total = meta.get('total_boundaries', len(gradient_boundaries))
    ax.set_title(f'Gradient Boundaries - Scanning {scan_step}/{total}',
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    # Fill info panel (extend as needed)
    _fill_info_panel(ax_info, )
    return fig
