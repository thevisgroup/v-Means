from .base import *
from .parent_basic_frames import *

def draw_gradient_comparison(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """
    Step 9: Gradient comparison between adjacent segments.
    """
    r = P['r']
    theta = P['theta']
    theta_edges = P['theta_edges']
    rmax = P['rmax']
    dmax_list = P['dmax_list']
    segment_points = P.get('segment_points', [])
    seg_i = P['seg_i']
    seg_j = P['seg_j']
    has_gradient = P['has_gradient']
    found_gradients_so_far = P.get('found_gradients_so_far', [])
    threshold = P.get('threshold', 0.25 * rmax)
    gradient_count = P.get('gradient_count', len(found_gradients_so_far))
    total_gradients = P.get('total_gradients', 0)

    display_rmax = _get_display_rmax(rmax)

    # Figure with info panel for comparison details
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(reduction=2),
               alpha=0.4, zorder=2)
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10)

    # Get values for comparison
    val_i = dmax_list[seg_i] if seg_i < len(dmax_list) else 0
    val_j = dmax_list[seg_j] if seg_j < len(dmax_list) else 0
    is_empty_i = len(segment_points[seg_i]) == 0 if seg_i < len(segment_points) else True
    is_empty_j = len(segment_points[seg_j]) == 0 if seg_j < len(segment_points) else True
    diff = abs(val_i - val_j)

    # Base silhouette layer (connected + smoothed)
    _draw_visible_silhouette(ax, theta_edges, dmax_list, label='Silhouette', zorder=3)

    # Pre-compute the smoothed silhouette path so highlight arcs match exactly
    n_segments_total = len(dmax_list)
    _pts_per_seg = 20
    _all_r_raw = []
    for _si in range(n_segments_total):
        _h = max(dmax_list[_si], 0.01)
        _all_r_raw.extend([_h] * _pts_per_seg)
    _all_r_raw.append(_all_r_raw[0])  # close
    _smoothed_r = _smooth_radii(_all_r_raw, sigma=SILHOUETTE_SMOOTH_SIGMA)

    # Highlight the two segments being compared (sample from smoothed path)
    for i in [seg_i, seg_j]:
        if i >= len(dmax_list):
            continue
        theta_start = theta_edges[i]
        theta_end = theta_edges[i + 1] if i + 1 < len(theta_edges) else theta_edges[0]
        has_data = len(segment_points[i]) > 0 if i < len(segment_points) else False

        if not has_data:
            continue

        if has_gradient:
            line_color = '#FFD700' if i == seg_i else '#32CD32'
        else:
            line_color = '#AAAAAA' if i == seg_i else '#BBBBBB'

        # Extract the smoothed radii for this segment's slice
        seg_start_idx = i * _pts_per_seg
        seg_end_idx = seg_start_idx + _pts_per_seg
        seg_r_smooth = _smoothed_r[seg_start_idx:seg_end_idx]

        if len(seg_r_smooth) > 0 and np.max(seg_r_smooth) > 0:
            theta_arc = np.linspace(theta_start, theta_end, len(seg_r_smooth))
            ax.plot(theta_arc, seg_r_smooth, color=line_color, linewidth=3, linestyle='-', zorder=4)

    # Collect all gradient angles found so far and tag them
    gradient_angles = []
    for (prev_i, prev_j) in found_gradients_so_far:
        boundary_theta = _gradient_pair_boundary_angle(
            theta_edges, prev_i, prev_j
        )
        if boundary_theta not in gradient_angles:
            gradient_angles.append(boundary_theta)

    # Include the current pair if it's a gradient too
    if has_gradient:
        current_boundary = _gradient_pair_boundary_angle(
            theta_edges, seg_i, seg_j
        )
        if current_boundary not in gradient_angles:
            gradient_angles.append(current_boundary)

    # Sort by angle for consistent labeling
    gradient_angles.sort()

    # Previously found gradient lines (uniform width)
    for idx, (prev_i, prev_j) in enumerate(found_gradients_so_far):
        if (prev_i, prev_j) == (seg_i, seg_j):
            continue
        boundary_theta = _gradient_pair_boundary_angle(
            theta_edges, prev_i, prev_j
        )
        ax.plot([boundary_theta, boundary_theta], [0, display_rmax],
                color='magenta', linewidth=2.5, linestyle='--', alpha=0.8, zorder=8)

    # Current gradient line (if detected)
    if has_gradient:
        boundary_theta = _gradient_pair_boundary_angle(
            theta_edges, seg_i, seg_j
        )
        ax.plot([boundary_theta, boundary_theta], [0, display_rmax],
                color='magenta', linewidth=2.5, linestyle='--', alpha=0.9, zorder=9)

    # Label each gradient with G1, G2, ...
    for idx, g_angle in enumerate(gradient_angles):
        ax.text(g_angle, display_rmax * 1.06, f'G{idx + 1}', fontsize=10,
                color='magenta', fontweight='bold', ha='center')

    _configure_polar_ax(ax, display_rmax)

    status = "✓ GRADIENT" if has_gradient else "○ Pass"
    ax.set_title(_get_title(f'Seg {seg_i} vs {seg_j}: {status}', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    # === Info panel -- safe from clipping ===

    # Progress block
    progress_text = f"Found: {gradient_count}/{total_gradients}"

    # Comparison details (built as a list to keep it readable)
    compare_lines = []
    compare_lines.append(f"Seg{seg_i}: d={val_i:.2f}" + ("(E)" if is_empty_i else ""))
    compare_lines.append(f"Seg{seg_j}: d={val_j:.2f}" + ("(E)" if is_empty_j else ""))
    compare_lines.append(f"Diff: {diff:.2f}")
    compare_lines.append(f"Thresh: {threshold:.2f}")
    compare_lines.append("─" * 14)

    if is_empty_i != is_empty_j:
        compare_lines.append("→ Empty boundary")
        compare_lines.append("  detected!")
        result_color = '#28a745'
    elif diff > threshold:
        compare_lines.append(f"→ {diff:.2f}>{threshold:.2f}")
        compare_lines.append("  = GRADIENT!")
        result_color = '#28a745'
    else:
        compare_lines.append(f"→ {diff:.2f}≤{threshold:.2f}")
        compare_lines.append("  = No gradient")
        result_color = '#6c757d'

    compare_text = "\n".join(compare_lines)

    _add_info_to_panel(ax_info, [
        {'text': progress_text, 'y': 0.88, 'fontsize': 6, 'color': 'steelblue', 'title': 'Progress'},
        {'text': compare_text, 'y': 0.55, 'fontsize': 6, 'color': result_color, 'title': 'Comparison'}
    ])

    return fig


def draw_gradient_lines_final(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Final gradient lines overlaid on the connected silhouette."""
    r = P['r']
    theta = P['theta']
    theta_edges = P['theta_edges']
    rmax = P['rmax']
    gradient_angles = P.get('gradient_angles', [])

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.5, label='Points', zorder=2)
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10, label='Center')

    # Connected silhouette with spokes at gradient positions
    segment_dmax = _compute_segment_dmax(np.array(r), np.array(theta), theta_edges)
    _draw_connected_silhouette_with_gradients(ax, theta_edges, segment_dmax,
                                              gradient_angles=gradient_angles, label='Silhouette')

    for t_edge in theta_edges:
        ax.plot([t_edge, t_edge], [0, display_rmax], color='lightgray',
                linewidth=0.5, linestyle=':', alpha=0.5)

    for idx, angle in enumerate(gradient_angles):
        ax.plot([angle, angle], [0, display_rmax], color='magenta',
                linewidth=2.5, linestyle='--', alpha=0.9, zorder=8)
        ax.text(angle, display_rmax * 1.06, f'G{idx + 1}', fontsize=10,
                color='magenta', fontweight='bold', ha='center')

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Gradient Lines (G-lines)', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_dynamic_center_found(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Dynamic center discovery -- new cluster centers shown on the silhouette."""
    r = P['r']
    theta = P['theta']
    theta_edges = P.get('theta_edges', [])
    rmax = P['rmax']
    gradient_angles = P.get('gradient_angles', [])
    centers_so_far = P.get('centers_so_far', [])
    center_labels = P.get('center_labels', [])
    centroid = P['centroid']

    from vmeans.data import convert_to_polar

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.5, label='Points', zorder=2)
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10, label='Center')

    # Draw connected silhouette with gradient spokes
    # If theta_edges is missing, fall back to 60 uniform segments
    n_segments = 60
    if len(theta_edges) == 0:
        theta_edges = np.linspace(0, 2 * np.pi, n_segments + 1).tolist()

    if len(theta_edges) > 0:
        segment_dmax = _compute_segment_dmax(np.array(r), np.array(theta), theta_edges)
        _draw_connected_silhouette_with_gradients(ax, theta_edges, segment_dmax,
                                                  gradient_angles=gradient_angles,
                                                  label='Silhouette', zorder=4)

    # Gradient lines with G-labels
    for idx, angle in enumerate(gradient_angles):
        ax.plot([angle, angle], [0, display_rmax], color='magenta',
                linewidth=2.5, linestyle='--', alpha=0.8, zorder=8)
        ax.text(angle, display_rmax * 1.06, f'G{idx + 1}', fontsize=10,
                color='magenta', fontweight='bold', ha='center')

    # Assign colors based on center angles
    if len(centers_so_far) > 0:
        colors = get_colors_for_centers(np.array(centers_so_far), origin=centroid)
    else:
        colors = []
    for idx, center in enumerate(centers_so_far):
        center_r, center_theta = convert_to_polar(np.array([center]), centroid)
        color = colors[idx] if idx < len(colors) else '#888888'
        label = center_labels[idx] if idx < len(center_labels) else f'C{idx + 1}'

        ax.scatter([center_theta[0]], [center_r[0]], c=color, s=_get_highlight_size() * 1.5,
                   marker='X', edgecolors='black', linewidth=1.5,
                   zorder=15, label=label)

    _configure_polar_ax(ax, display_rmax)

    n_centers = len(centers_so_far)
    ax.set_title(
        _get_title(f'Compute New Centers - C{n_centers} ({n_centers}/{meta.get("total_centers", n_centers)})', meta),
        fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


# ============================================================================
# NEW: Gradient pair search visualization
# ============================================================================

def draw_gradient_pair_search(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """
    NEW: Visualize the search for matching gradient pairs
    Shows which G-line interval is being checked for data points
    """
    r = P['r']
    theta = P['theta']
    points = P['points']
    rmax = P['rmax']
    theta_edges = P.get('theta_edges', [])
    gradient_angles = P.get('gradient_angles', [])
    centroid = P['centroid']
    current_pair_idx = P.get('current_pair_idx', 0)
    points_in_range = P.get('points_in_range_count', 0)
    has_points = P.get('has_points', False)
    centers_found_so_far = P.get('centers_found_so_far', [])

    from vmeans.data import convert_to_polar

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Calculate current interval
    n_angles = len(gradient_angles)
    if n_angles >= 2:
        g_current_idx = current_pair_idx % n_angles
        g_next_idx = (current_pair_idx + 1) % n_angles
        g_current = gradient_angles[g_current_idx]
        g_next = gradient_angles[g_next_idx]
    else:
        g_current = 0
        g_next = 2 * np.pi

    # Highlight the region being searched
    crosses_zero = g_current > g_next

    # Create filled wedge for search region
    if crosses_zero:
        # Region crosses 0: draw two arcs
        arc1_theta = np.linspace(g_current, 2 * np.pi, 30)
        arc2_theta = np.linspace(0, g_next, 30)
        arc_theta = np.concatenate([arc1_theta, arc2_theta])
    else:
        arc_theta = np.linspace(g_current, g_next, 50)

    arc_r = [display_rmax * 0.95] * len(arc_theta)

    # Fill the search region
    fill_color = '#90EE90' if has_points else '#FFB6C1'  # Green if has points, pink if empty
    theta_fill = np.concatenate([[g_current], arc_theta, [g_next], [g_current]])
    r_fill = np.concatenate([[0], arc_r, [0], [0]])
    ax.fill(theta_fill, r_fill, color=fill_color, alpha=0.3, zorder=1)
    ax.plot(theta_fill, r_fill, color=fill_color, linewidth=2, alpha=0.8, zorder=2)

    # Draw all data points
    ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.5, zorder=3)

    # Draw connected silhouette
    # Reconstruct default 60 segments if theta_edges is missing
    n_segments = 60  # default segment count
    if len(theta_edges) == 0:
        theta_edges = np.linspace(0, 2 * np.pi, n_segments + 1).tolist()

    if len(theta_edges) > 0:
        segment_dmax = _compute_segment_dmax(np.array(r), np.array(theta), theta_edges)
        # Pass gradient_angles so spokes get drawn at the right positions
        _draw_connected_silhouette_with_gradients(ax, theta_edges, segment_dmax,
                                                  gradient_angles=gradient_angles,
                                                  label='Silhouette', zorder=4)

    # Highlight points in the current search region
    if has_points:
        # Find and highlight points in this region
        for j, angle in enumerate(theta):
            if crosses_zero:
                in_range = (angle >= g_current) or (angle <= g_next)
            else:
                in_range = g_current <= angle <= g_next
            if in_range:
                ax.scatter([angle], [r[j]], c='#32CD32', s=_get_point_size() + 10,
                           alpha=0.8, zorder=5, edgecolors='black', linewidth=0.5)

    # Draw center
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10, label='Center')

    # Draw all gradient lines (uniform width)
    for idx, angle in enumerate(gradient_angles):
        is_current = (idx == g_current_idx) or (idx == g_next_idx)
        line_color = '#FF1493' if is_current else 'magenta'  # Brighter for current pair
        ax.plot([angle, angle], [0, display_rmax], color=line_color,
                linewidth=2.5, linestyle='--', alpha=0.9, zorder=8)
        ax.text(angle, display_rmax * 1.06, f'G{idx + 1}', fontsize=10,
                color='magenta', fontweight='bold', ha='center')

    # Draw previously found centers
    if len(centers_found_so_far) > 0:
        colors = get_colors_for_centers(np.array(centers_found_so_far), origin=centroid)
    else:
        colors = []
    for idx, center in enumerate(centers_found_so_far):
        center_r, center_theta = convert_to_polar(np.array([center]), centroid)
        color = colors[idx] if idx < len(colors) else '#888888'
        ax.scatter([center_theta[0]], [center_r[0]], c=color, s=_get_highlight_size() * 1.5,
                   marker='X', edgecolors='black', linewidth=1.5, zorder=15)

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title(f'Search Gradient Pair G{g_current_idx + 1}-G{g_next_idx + 1}', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    # Info text
    status = f"✓ Found {points_in_range} pts → C{len(centers_found_so_far) + 1}" if has_points else "✗ Empty → Skip"
    info_text = f"G{g_current_idx + 1}→G{g_next_idx + 1}\n"
    info_text += f"Pts: {points_in_range}\n"
    info_text += status

    result_color = 'green' if has_points else 'red'
    # Info goes into the info panel

    _fill_info_panel(ax_info, info_text=info_text, info_color='gray')

    return fig


def draw_find_matching_gradient_pairs(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Find matching gradient pairs - final result showing colored regions"""
    points = P['points']
    r = P['r']
    theta = P['theta']
    theta_edges = P['theta_edges']
    rmax = P['rmax']
    gradient_angles = P.get('gradient_angles', [])
    additional_centers = P.get('additional_centers', [])
    center_labels = P.get('center_labels', [])
    centroid = P['centroid']
    regions = P.get('regions', [])

    from vmeans.data import convert_to_polar

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Generate colors based on cluster center angles (color wheel mapping)
    if additional_centers and len(additional_centers) > 0:
        region_colors = get_colors_for_centers(np.array(additional_centers), origin=centroid)
    else:
        # Fallback palette
        region_colors = get_cluster_colors(10)  # fallback

    # Draw points colored by region
    if regions:
        # First draw all points as gray background
        ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
                   alpha=0.3, zorder=2)

        # Then color points by region
        for region_idx, region_point_indices in enumerate(regions):
            if len(region_point_indices) > 0:
                region_color = region_colors[region_idx % len(region_colors)]
                region_theta = theta[region_point_indices]
                region_r = r[region_point_indices]
                ax.scatter(region_theta, region_r, c=region_color, s=_get_point_size(),
                           alpha=0.7, zorder=3, label=f'Region {region_idx + 1}')
    else:
        ax.scatter(theta, r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
                   alpha=0.5, zorder=2)

    # Draw center
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10, label='Center')

    # No silhouette needed on this step

    # Draw gradient lines with labels
    for idx, angle in enumerate(gradient_angles):
        ax.plot([angle, angle], [0, display_rmax], color='magenta',
                linewidth=2.5, linestyle='--', alpha=0.9, zorder=8)
        ax.text(angle, display_rmax * 1.06, f'G{idx + 1}', fontsize=10,
                color='magenta', fontweight='bold', ha='center')

    # Draw additional centers
    for idx, center in enumerate(additional_centers):
        center_r, center_theta = convert_to_polar(np.array([center]), centroid)
        color = region_colors[idx % len(region_colors)]
        label = center_labels[idx] if idx < len(center_labels) else f'C{idx + 1}'

        ax.scatter([center_theta[0]], [center_r[0]], c=color, s=_get_highlight_size() * 1.5,
                   marker='X', edgecolors='black', linewidth=1.5,
                   zorder=15, label=label)

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Gradient Pairs Matched', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_no_gradient_boundaries(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """No gradient boundaries detected"""
    r = P['r']
    theta = P['theta']
    rmax = P['rmax']

    display_rmax = _get_display_rmax(rmax)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    ax.scatter(theta, r, c=COLOR_DATA_POINTS, s=_get_point_size(),
               alpha=0.7, label='Points', zorder=2)
    ax.scatter([0], [0], c=COLOR_CENTER, s=_get_center_size(), zorder=10, label='Center')

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('No Gradient Boundaries Detected', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    # Show legend and info in the panel
    _fill_info_panel(ax_info, handles=handles, labels=labels,
                     info_text="No gradient\nboundaries found", info_color='gray')

    return fig
