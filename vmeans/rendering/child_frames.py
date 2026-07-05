from .base import *

def draw_parallel_child_init(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - show centers"""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    parent_theta = P['parent_theta']

    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Draw parent data points
    ax.scatter(parent_theta, parent_r, c=COLOR_DATA_POINTS_GRAY, s=_get_point_size(),
               alpha=0.4, zorder=2)

    # Draw parent center
    ax.scatter([0], [0], c='black', s=_get_center_size(), zorder=10, label='Origin')

    # Draw child centers
    child_colors = _get_child_colors(child_analyses, parent_centroid)
    for idx, child in enumerate(child_analyses):
        child_center = child['center_global']
        child_r, child_theta = convert_to_polar(np.array([child_center]), parent_centroid)
        color = child_colors[idx % len(child_colors)]

        ax.scatter([child_theta[0]], [child_r[0]], c=color, s=_get_highlight_size(),
                   marker='X', edgecolors='black', linewidth=1,
                   zorder=15, label=f'C{idx + 1}')

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Child - Show Centers', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_parallel_child_segments(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - show segments"""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    parent_theta = P['parent_theta']

    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Draw parent center
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

        # Draw child center
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_highlight_size(), marker='X', edgecolors='black',
                   linewidth=1, zorder=15, label=f'C{idx + 1}')

        # Draw segment arcs for this child (outer arc only, no spokes)
        for i in range(len(dmax_list)):
            theta_start_local = theta_edges_local[i]
            theta_end_local = theta_edges_local[i + 1] if i + 1 < len(theta_edges_local) else theta_edges_local[0]

            has_data = len(segment_points[i]) > 0 if i < len(segment_points) else False
            if not has_data:
                continue

            height = rmax_child  # Initially show at max height

            if height > 0:
                theta_arc_local = np.linspace(theta_start_local, theta_end_local, 15)
                x_local = height * np.cos(theta_arc_local)
                y_local = height * np.sin(theta_arc_local)

                points_local = np.column_stack([x_local, y_local])
                points_global = points_local + child_center

                r_global, theta_global = convert_to_polar(points_global, parent_centroid)

                ax.plot(theta_global, r_global, color=COLOR_SILHOUETTE, linewidth=1, linestyle='-')

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Child - Show Segments', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_parallel_child_shrinking(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - shrinking animation with visible silhouette (connected + smoothed)."""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    parent_theta = P['parent_theta']

    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    child_colors = _get_child_colors(child_analyses, parent_centroid)

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        current_heights = child.get('current_heights', child['dmax_list'])
        segment_points = child['segment_points']
        theta_edges_local = child['theta_edges']
        current_segment_idx = child.get('current_segment_idx', -1)

        # Draw data points first
        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)

        # Draw child center
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_highlight_size(), marker='X', edgecolors='black',
                   linewidth=1, zorder=15, label=f'C{idx + 1}')

        # Draw visible silhouette (connected + smoothed)
        _draw_child_visible_silhouette(ax, child_center, theta_edges_local, current_heights,
                                       parent_centroid, color=COLOR_SILHOUETTE, zorder=5)

        # Highlight current segment being processed (orange)
        if current_segment_idx >= 0 and current_segment_idx < len(current_heights):
            i = current_segment_idx
            theta_start_local = theta_edges_local[i]
            theta_end_local = theta_edges_local[i + 1] if i + 1 < len(theta_edges_local) else theta_edges_local[0]
            height = current_heights[i]

            if height > 0:
                theta_arc_local = np.linspace(theta_start_local, theta_end_local, 15)
                x_local = height * np.cos(theta_arc_local)
                y_local = height * np.sin(theta_arc_local)
                points_local = np.column_stack([x_local, y_local])
                points_global = points_local + child_center
                r_global, theta_global = convert_to_polar(points_global, parent_centroid)
                ax.plot(theta_global, r_global, color='orange', linewidth=2.5, linestyle='-', zorder=6)

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Child - Reduce Empty Space', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


def draw_parallel_child_gradient_comparison(P, meta, figsize, opts=None):
    """Step 14.9: Child cluster gradient comparison - aligned with Step 9 logic."""
    from vmeans.data import convert_to_polar

    parent_centroid = P['parent_centroid']
    child_analyses   = P['child_analyses']
    parent_r         = P['parent_r']
    parent_theta     = P['parent_theta']

    max_r        = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    child_colors = _get_child_colors(child_analyses, parent_centroid)
    info_texts   = []

    for idx, child in enumerate(child_analyses):
        color       = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        dmax_list    = child['dmax_list']
        segment_points = child['segment_points']
        theta_edges_local   = child['theta_edges']
        current_comparison  = child.get('current_comparison', None)
        found_gradients_so_far = child.get('found_gradients_so_far', [])
        gradient_count      = child.get('gradient_count', 0)
        total_gradients     = child.get('total_gradients', 0)
        comparison_complete = child.get('comparison_complete', False)
        rmax_child  = child['rmax']

        # --- 1. Draw data points ---
        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color,
                       s=_get_point_size(), alpha=0.4, zorder=2)

        # Draw child center marker
        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)
        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_highlight_size(), marker='X',
                   edgecolors='black', linewidth=1, zorder=15, label=f'C{idx+1}')

        # --- 2. Draw continuous smoothed silhouette (same as Step 9) ---
        _draw_child_visible_silhouette_connected(
            ax, child_center, theta_edges_local, dmax_list,
            parent_centroid, color=COLOR_SILHOUETTE,
            linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.8, zorder=3
        )

        # --- 3. Highlight the two segments being compared ---
        # Pre-compute the smoothed silhouette path in local coords,
        # then slice out each segment's radii -- mirrors Step 9 exactly.
        if current_comparison and isinstance(current_comparison, dict):
            seg_i = current_comparison.get('seg_i', -1)
            seg_j = current_comparison.get('seg_j', -1)
            has_gradient = current_comparison.get('has_gradient', False)

            # Build the full smoothed radius array for this child
            _pts_per_seg = 20
            _all_r_raw = []
            for _si in range(len(dmax_list)):
                _all_r_raw.extend([max(dmax_list[_si], 0.01)] * _pts_per_seg)
            _all_r_raw.append(_all_r_raw[0])  # close the path
            _smoothed_r = _smooth_radii(_all_r_raw, sigma=SILHOUETTE_SMOOTH_SIGMA)

            for i in [seg_i, seg_j]:
                if i < 0 or i >= len(dmax_list):
                    continue
                has_data = (len(segment_points[i]) > 0
                            if i < len(segment_points) else False)
                if not has_data:
                    continue

                # Gold for seg_i, green for seg_j when gradient found; gray otherwise
                if has_gradient:
                    line_color = '#FFD700' if i == seg_i else '#32CD32'
                else:
                    line_color = '#AAAAAA' if i == seg_i else '#BBBBBB'

                # Slice the smoothed radii for this segment
                seg_r_smooth = _smoothed_r[i * _pts_per_seg : (i+1) * _pts_per_seg]

                theta_start_local = theta_edges_local[i]
                theta_end_local   = (theta_edges_local[i+1]
                                     if i+1 < len(theta_edges_local)
                                     else theta_edges_local[0] + 2*np.pi)
                theta_arc_local = np.linspace(theta_start_local, theta_end_local,
                                              len(seg_r_smooth))

                # Convert: local polar → global Cartesian → parent polar
                x_local = seg_r_smooth * np.cos(theta_arc_local)
                y_local = seg_r_smooth * np.sin(theta_arc_local)
                pts_global = np.column_stack([x_local, y_local]) + child_center
                r_global, theta_global = convert_to_polar(pts_global, parent_centroid)

                if np.max(r_global) > 0:
                    ax.plot(theta_global, r_global,
                            color=line_color, linewidth=3,
                            linestyle='-', zorder=4)

        # --- 4. Draw previously found gradient lines ---
        g_counter = 0
        for prev_i, prev_j in found_gradients_so_far:
            boundary_theta_local = _gradient_pair_boundary_angle(
                theta_edges_local, prev_i, prev_j
            )
            x_end = rmax_child * np.cos(boundary_theta_local)
            y_end = rmax_child * np.sin(boundary_theta_local)
            line_pts = np.array([child_center,
                                  [child_center[0]+x_end, child_center[1]+y_end]])
            r_line, theta_line = convert_to_polar(line_pts, parent_centroid)
            ax.plot(theta_line, r_line, color='magenta',
                    linewidth=2.5, linestyle='--', alpha=0.7)
            g_counter += 1
            ax.text(theta_line[1], r_line[1]*1.06, f'C{idx+1}-G{g_counter}',
                    fontsize=6, color='magenta', fontweight='bold', ha='center')

        # --- 5. Build status text for info panel ---
        if comparison_complete:
            info_texts.append(f"C{idx+1}: Done ({gradient_count}G)")
        elif current_comparison and isinstance(current_comparison, dict):
            s_i = current_comparison.get('seg_i', 0)
            s_j = current_comparison.get('seg_j', 0)
            hg  = current_comparison.get('has_gradient', False)
            info_texts.append(f"C{idx+1}: {s_i}↔{s_j} {'✓' if hg else '○'}")
        else:
            info_texts.append(f"C{idx+1}: Waiting...")

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Gradient Comparison', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    level_label = P.get('level_label', 'Child')
    info_text = f"{level_label} Status\n" + "\n".join(info_texts)
    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(
        ax_info,
        handles=handles,
        labels=labels,
        info_text=info_text,
        info_color='gray'
    )
    return fig

def draw_parallel_child_gradient_lines(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - gradient lines"""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    parent_theta = P['parent_theta']

    from vmeans.data import convert_to_polar

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    # Polar plot on the left, info panel on the right
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # Don't draw parent center when recursing

    child_colors = _get_child_colors(child_analyses, parent_centroid)

    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        child_points = child.get('points', None)
        dmax_list = child['dmax_list']
        rmax_child = child['rmax']
        gradient_angles_local = child.get('gradient_angles', [])

        # Draw data points first
        if child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            ax.scatter(pts_theta, pts_r, c=color, s=_get_point_size(),
                       alpha=0.4, zorder=2)

        child_r_parent, child_theta_parent = convert_to_polar(
            np.array([child_center]), parent_centroid)

        ax.scatter([child_theta_parent[0]], [child_r_parent[0]], c=color,
                   s=_get_highlight_size(), marker='X', edgecolors='black',
                   linewidth=1, zorder=15, label=f'C{idx + 1}')

        # Draw connected silhouette outline (smoothed)
        theta_edges_local = child['theta_edges']
        _draw_child_visible_silhouette_connected(ax, child_center, theta_edges_local, dmax_list,
                                                 parent_centroid, color=COLOR_SILHOUETTE,
                                                 linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.8,
                                                 label=None)

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
            # Add G-line label (with child prefix to avoid duplicate names)
            ax.text(theta_global[1], r_global[1] * 1.06, f'C{idx + 1}-G{g_idx + 1}',
                    fontsize=6, color='magenta', fontweight='bold', ha='center')

    _configure_polar_ax(ax, display_rmax)

    ax.set_title(_get_title('Child - Gradient Lines', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)
    # Legend goes into the info panel

    handles, labels = ax.get_legend_handles_labels()

    _fill_info_panel(ax_info, handles=handles, labels=labels, )
    return fig


# ============================================================================
# Child-level Gradient Pair Search / Center Finding / Algorithm Pass Complete
# ============================================================================

def draw_parallel_child_gradient_pair_search(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - Search gradient pairs for each child"""
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

        # Draw silhouette shape (connected + smoothed)
        _draw_child_visible_silhouette_connected(ax, child_center, theta_edges_local, dmax_list,
                                                 parent_centroid, color=COLOR_SILHOUETTE,
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

            # Highlight found points in the interval
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
    """Child clusters - Show new centers found for each child"""
    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    centers_shown = P.get('centers_shown', 1)

    from vmeans.data import convert_to_polar

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

        # Draw silhouette (connected + smoothed)
        _draw_child_visible_silhouette_connected(ax, child_center, theta_edges_local, dmax_list,
                                                 parent_centroid, color=COLOR_SILHOUETTE,
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
