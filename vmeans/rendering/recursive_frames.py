from .base import *

def _collect_all_sub_regions(child_analyses, parent_centroid):
    """Collect all sub-regions across children."""
    from vmeans.data import convert_to_polar
    from vmeans.core_analysis import segment_points_by_region

    all_sub_regions = []

    for idx, child in enumerate(child_analyses):
        child_center = child['center_global']
        child_points = child.get('points', None)
        child_theta_data = child.get('theta', None)
        gradient_angles_local = child.get('gradient_angles', [])
        additional_centers_local = child.get('additional_centers', [])
        regions_local = child.get('regions', [])

        if (child_points is not None and len(child_points) > 0 and
                child_theta_data is not None and len(gradient_angles_local) >= 2 and
                len(additional_centers_local) > 0):
            if not regions_local:
                regions_local = segment_points_by_region(
                    child_points, child_theta_data, gradient_angles_local, additional_centers_local
                )
            for sub_idx, sub_region_indices in enumerate(regions_local):
                if len(sub_region_indices) > 0:
                    sub_points = child_points[sub_region_indices]
                    sub_r, sub_theta = convert_to_polar(sub_points, parent_centroid)
                    sub_center = additional_centers_local[sub_idx] if sub_idx < len(
                        additional_centers_local) else np.mean(sub_points, axis=0)
                    label = f'Region {len(all_sub_regions) + 1}'
                    all_sub_regions.append((sub_r, sub_theta, label, sub_center))
        elif child_points is not None and len(child_points) > 0:
            pts_r, pts_theta = convert_to_polar(child_points, parent_centroid)
            label = f'Region {len(all_sub_regions) + 1}'
            all_sub_regions.append((pts_r, pts_theta, label, child_center))

    return all_sub_regions


def _get_sub_region_colors(all_sub_regions, parent_centroid):
    """Get colors for sub-regions."""
    n = len(all_sub_regions)
    if n == 0:
        return []
    region_centers = np.array([sr[3] for sr in all_sub_regions])
    region_counts = [len(sr[0]) for sr in all_sub_regions]
    return get_colors_for_centers(region_centers, origin=parent_centroid,
                                  point_counts=region_counts, ensure_distinct=True)


def _collect_recursive_context_leaves(P):
    """Return untouched child leaves with the colors used in Step 14.14."""
    from vmeans.data import convert_to_polar
    from vmeans.core_analysis import segment_points_by_region

    child_analyses = P.get('context_child_analyses', [])
    parent_centroid = P['parent_centroid']
    active_analyses = P.get('child_analyses', [])
    replaced_keys = {
        (analysis.get('parent_region_id'), analysis.get('region_id'))
        for analysis in active_analyses
        if analysis.get('parent_region_id') is not None
    }

    leaves = []
    color_inputs = []
    for child_idx, child in enumerate(child_analyses):
        child_id = child.get('region_id', child_idx + 1)
        child_points = child.get('points')
        child_theta = child.get('theta')
        gradient_angles = child.get('gradient_angles', [])
        additional_centers = child.get('additional_centers', [])
        regions = child.get('regions', [])

        if (child_points is not None and len(child_points) > 0 and
                child_theta is not None and len(gradient_angles) >= 2 and
                len(additional_centers) > 0):
            if not regions:
                regions = segment_points_by_region(
                    child_points, child_theta, gradient_angles, additional_centers
                )
            for sub_idx, indices in enumerate(regions):
                if len(indices) == 0:
                    continue
                points = child_points[indices]
                center = (
                    additional_centers[sub_idx]
                    if sub_idx < len(additional_centers)
                    else np.mean(points, axis=0)
                )
                radius, angle = convert_to_polar(points, parent_centroid)
                leaves.append({
                    'key': (child_id, sub_idx + 1),
                    'points': points,
                    'center': center,
                    'radius': radius,
                    'angle': angle,
                    'number': len(leaves) + 1,
                })
                color_inputs.append((radius, angle, '', center))
        elif child_points is not None and len(child_points) > 0:
            radius, angle = convert_to_polar(child_points, parent_centroid)
            leaves.append({
                'key': (child_id, None),
                'points': child_points,
                'center': child.get('center_global', np.mean(child_points, axis=0)),
                'radius': radius,
                'angle': angle,
                'number': len(leaves) + 1,
            })
            color_inputs.append((radius, angle, '', leaves[-1]['center']))

    colors = _get_sub_region_colors(color_inputs, parent_centroid)
    untouched = []
    for index, leaf in enumerate(leaves):
        if leaf['key'] in replaced_keys:
            continue
        leaf = dict(leaf)
        leaf['color'] = colors[index] if index < len(colors) else COLOR_DATA_POINTS_GRAY
        untouched.append(leaf)
    return untouched


def _draw_recursive_context(ax, P, include_labels=False, include_silhouettes=True):
    """Draw child leaves that are not being replaced by the grandchild pass."""
    from vmeans.core_analysis import compute_better_center
    from vmeans.data import convert_to_polar
    from vmeans.segment import segment_points_by_theta

    parent_centroid = P['parent_centroid']
    leaf_segments = max(int(P.get('context_segments', 60)) // 2, 8)

    for leaf in _collect_recursive_context_leaves(P):
        label = f'Unchanged Region {leaf["number"]}' if include_labels else None
        ax.scatter(
            leaf['angle'],
            leaf['radius'],
            c=leaf['color'],
            s=_get_point_size(),
            alpha=0.4,
            zorder=1,
            label=label
        )

        points = leaf['points']
        if not include_silhouettes or len(points) < 2:
            continue
        leaf_center = compute_better_center(points, method='centroid')
        leaf_r, leaf_theta = convert_to_polar(points, leaf_center)
        segment_data, theta_edges = segment_points_by_theta(
            leaf_r, leaf_theta, s=leaf_segments
        )
        dmax_list = [max(segment) if segment else 0.0 for segment in segment_data]
        _draw_child_visible_silhouette_connected(
            ax,
            leaf_center,
            theta_edges,
            dmax_list,
            parent_centroid,
            color=leaf['color'],
            linewidth=CHILD_SILHOUETTE_LINE_WIDTH,
            alpha=0.65,
            zorder=1,
            label=None
        )


def _collect_analysis_leaf_clusters(analyses):
    """Collect final non-empty leaf point sets from one analysis level."""
    leaf_clusters = []
    for analysis in analyses:
        points = analysis.get('points')
        regions = analysis.get('regions', [])

        if points is not None and len(regions) > 0:
            for indices in regions:
                if len(indices) >= 1:
                    leaf_clusters.append(points[indices])
        elif points is not None and len(points) >= 1:
            leaf_clusters.append(points)
    return leaf_clusters


def _collect_final_leaf_clusters(P):
    """Compose the deepest recursive leaves with all untouched child leaves."""
    active_leaves = _collect_analysis_leaf_clusters(P.get('child_analyses', []))
    if not P.get('context_child_analyses'):
        return active_leaves

    untouched_leaves = [
        leaf['points']
        for leaf in _collect_recursive_context_leaves(P)
    ]
    return active_leaves + untouched_leaves


def _collect_final_leaf_render_data(P):
    """Build one globally numbered and colored leaf list for recursive summaries."""
    from vmeans.data import convert_to_polar

    parent_centroid = P['parent_centroid']
    leaves = []
    color_inputs = []
    for points in _collect_final_leaf_clusters(P):
        if points is None or len(points) == 0:
            continue
        center = np.mean(points, axis=0)
        radius, angle = convert_to_polar(points, parent_centroid)
        leaves.append({
            'points': points,
            'center': center,
            'radius': radius,
            'angle': angle,
        })
        color_inputs.append((radius, angle, '', center))

    colors = _get_sub_region_colors(color_inputs, parent_centroid)
    for index, leaf in enumerate(leaves):
        leaf['color'] = (
            colors[index] if index < len(colors) else COLOR_DATA_POINTS_GRAY
        )
        leaf['label'] = f'Region {index + 1}'
    return leaves


def _draw_final_leaf_silhouette(ax, leaf, parent_centroid, segments,
                                color, alpha=0.9, zorder=2):
    """Draw one leaf silhouette while keeping singleton leaves as valid clusters."""
    from vmeans.core_analysis import compute_better_center
    from vmeans.data import convert_to_polar
    from vmeans.segment import segment_points_by_theta

    points = leaf['points']
    if len(points) < 2:
        return

    leaf_center = compute_better_center(points, method='centroid')
    leaf_r, leaf_theta = convert_to_polar(points, leaf_center)
    segment_data, theta_edges = segment_points_by_theta(
        leaf_r, leaf_theta, s=segments
    )
    dmax_list = [max(segment) if segment else 0.0 for segment in segment_data]
    _draw_child_visible_silhouette_connected(
        ax,
        leaf_center,
        theta_edges,
        dmax_list,
        parent_centroid,
        color=color,
        linewidth=CHILD_SILHOUETTE_LINE_WIDTH,
        alpha=alpha,
        zorder=zorder,
        label=None
    )


def _draw_grandchild_composed_summary(
        P, meta, figsize, opts, final=False,
        use_cluster_silhouette_colors=False):
    """Draw Steps 15.13/15.14 from the same global final leaf composition."""
    if opts is None:
        opts = _DEFAULT_RENDER_OPTIONS

    parent_centroid = P['parent_centroid']
    child_analyses = P.get('child_analyses', [])
    parent_r = P.get('parent_r', [])
    active_leaf_segments = max(int(P.get('segments', 60)) // 2, 8)
    context_leaf_segments = max(
        int(P.get('context_segments', P.get('segments', 60))) // 2,
        8
    )
    leaves = _collect_final_leaf_render_data(P)
    active_leaf_count = len(_collect_analysis_leaf_clusters(child_analyses))

    max_r = max(
        _get_child_max_r(parent_r, child_analyses, parent_centroid),
        float(P.get('display_rmax', 0))
    )
    display_rmax = _get_display_rmax(max_r)
    fig, ax, ax_info = _create_figure_with_panel(figsize)

    for leaf in leaves:
        ax.scatter(
            leaf['angle'],
            leaf['radius'],
            c=leaf['color'],
            s=_get_point_size(),
            alpha=0.4,
            zorder=3,
            label=leaf['label']
        )

    for leaf_index, leaf in enumerate(leaves):
        leaf_segments = (
            active_leaf_segments
            if leaf_index < active_leaf_count
            else context_leaf_segments
        )
        _draw_final_leaf_silhouette(
            ax, leaf, parent_centroid, leaf_segments, leaf['color']
        )

    base_title = (
        'Grandchild - Final Analysis Complete'
        if final else 'Grandchild - Algorithm 1 Pass Complete'
    )

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(
        _get_title(base_title, meta),
        fontsize=8,
        fontweight="bold",
        pad=0,
        y=1.05
    )
    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig


def draw_parallel_child_find_matching_pairs(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - Algorithm 1 Pass Complete."""
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

    # === Draw sub-regions ===
    all_sub_regions = _collect_all_sub_regions(child_analyses, parent_centroid)
    sub_colors = _get_sub_region_colors(all_sub_regions, parent_centroid)

    for i, (sr, st, label, _) in enumerate(all_sub_regions):
        ax.scatter(st, sr, c=sub_colors[i], s=_get_point_size(),
                   alpha=0.4, zorder=3, label=label)

    # === Draw connected blue silhouette for each child (smoothed) ===
    for idx, child in enumerate(child_analyses):
        child_center = child['center_global']
        dmax_list = child.get('dmax_list', [])
        theta_edges_local = child.get('theta_edges', [])

        if len(dmax_list) > 0 and len(theta_edges_local) > 0:
            _draw_child_visible_silhouette_connected(
                ax, child_center, theta_edges_local, dmax_list,
                parent_centroid, color=COLOR_SILHOUETTE,
                linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.9, label=None
            )

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Algorithm 1 Pass Complete', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig


def draw_parallel_child_final(P: dict, meta: dict, figsize, opts: RenderOptions = None) -> Figure:
    """Child clusters - final view with colored sub-regions and per-cluster silhouettes.

    Each final cluster (leaf sub-region) gets its own silhouette outline whose
    colour matches the points it contains. This replaces the earlier behaviour
    where one uniform blue silhouette was drawn per child, because a child may
    contain several leaf clusters that should each be outlined separately.
    """
    if opts is None:
        opts = _DEFAULT_RENDER_OPTIONS

    from vmeans.data import convert_to_polar
    from vmeans.core_analysis import compute_better_center, segment_points_by_region
    from vmeans.segment import segment_points_by_theta

    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']
    parent_segments = P.get('segments', 60)
    # Leaf-level segment count: keep the same angular resolution as the
    # preceding frames (14.10 / 14.11), otherwise the silhouette noticeably
    # loosens and fattens at the step 14.12 transition — which looked like
    # an algorithmic bug to viewers.
    leaf_segments = max(parent_segments // 2, 8)

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)
    point_size = opts.get_point_size(MPL_NORMAL_SIZE)

    # === Collect sub-regions and their colours (unchanged — points keep their colours) ===
    all_sub_regions = _collect_all_sub_regions(child_analyses, parent_centroid)
    sub_colors = _get_sub_region_colors(all_sub_regions, parent_centroid)

    # === Collect leaf point sets in the EXACT same order as _collect_all_sub_regions,
    # so leaf_point_sets[i] corresponds to all_sub_regions[i] and therefore to sub_colors[i].
    # This guarantees every silhouette colour matches the points it wraps. ===
    leaf_point_sets = []
    for child in child_analyses:
        child_points = child.get('points', None)
        child_theta_data = child.get('theta', None)
        gradient_angles_local = child.get('gradient_angles', [])
        additional_centers_local = child.get('additional_centers', [])
        regions_local = child.get('regions', [])

        if (child_points is not None and len(child_points) > 0 and
                child_theta_data is not None and len(gradient_angles_local) >= 2 and
                len(additional_centers_local) > 0):
            if not regions_local:
                regions_local = segment_points_by_region(
                    child_points, child_theta_data,
                    gradient_angles_local, additional_centers_local
                )
            for sub_region_indices in regions_local:
                if len(sub_region_indices) > 0:
                    leaf_point_sets.append(child_points[sub_region_indices])
        elif child_points is not None and len(child_points) > 0:
            leaf_point_sets.append(child_points)

    # === Draw points (same as before) ===
    for i, (sr, st, label, _) in enumerate(all_sub_regions):
        ax.scatter(st, sr, c=sub_colors[i], s=_get_point_size(),
                   alpha=0.4, zorder=3, label=label)

    # === Draw one silhouette per leaf cluster, coloured to match its points ===
    for i, pts in enumerate(leaf_point_sets):
        if len(pts) < 2:
            continue
        color = sub_colors[i] if i < len(sub_colors) else COLOR_SILHOUETTE

        leaf_center = compute_better_center(pts, method='centroid')
        leaf_r_local, leaf_theta_local = convert_to_polar(pts, leaf_center)
        seg_data, theta_edges = segment_points_by_theta(
            leaf_r_local, leaf_theta_local, s=leaf_segments
        )
        dmax_list = [max(seg) if seg else 0.0 for seg in seg_data]

        _draw_child_visible_silhouette_connected(
            ax, leaf_center, theta_edges, dmax_list,
            parent_centroid, color=color,
            linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.9, label=None
        )

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Final Analysis Complete', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig
