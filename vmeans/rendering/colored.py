from .base import *
from .colored_child_frames import *
from . import base as pyqt_matplotlib_vis

def draw_parallel_child_final(P, meta, figsize, opts: RenderOptions = None) -> Figure:
    """Step 14.14 — final frame; silhouettes identical to Step 14.13.

    Points are still coloured per sub-region (leaf).  Silhouettes are drawn
    per child using the cached child analysis, so the outline matches what
    the user already saw in the preceding child-level frames.
    """
    if opts is None:
        opts = _DEFAULT_RENDER_OPTIONS

    parent_centroid = P['parent_centroid']
    child_analyses = P['child_analyses']
    parent_r = P['parent_r']

    max_r = _get_child_max_r(parent_r, child_analyses, parent_centroid)
    display_rmax = _get_display_rmax(max_r)

    fig, ax, ax_info = _create_figure_with_panel(figsize)

    # --- Points: one colour per sub-region (unchanged from base/14.13). ---
    all_sub_regions = _collect_all_sub_regions(child_analyses, parent_centroid)
    sub_colors = _get_sub_region_colors(all_sub_regions, parent_centroid)

    for i, (sr, st, label, _sc) in enumerate(all_sub_regions):
        ax.scatter(st, sr, c=sub_colors[i], s=_get_point_size(),
                   alpha=0.4, zorder=3, label=label)

    # --- Silhouettes: one per CHILD, coloured by child (matches 14.13). ---
    child_colors = _get_child_colors(child_analyses, parent_centroid)
    for idx, child in enumerate(child_analyses):
        color = child_colors[idx % len(child_colors)]
        child_center = child['center_global']
        dmax_list = child.get('dmax_list', [])
        theta_edges_local = child.get('theta_edges', [])

        if len(dmax_list) > 0 and len(theta_edges_local) > 0:
            _draw_child_visible_silhouette_connected(
                ax, child_center, theta_edges_local, dmax_list,
                parent_centroid, color=color,
                linewidth=CHILD_SILHOUETTE_LINE_WIDTH, alpha=0.9, label=None,
            )

    _configure_polar_ax(ax, display_rmax)
    ax.set_title(_get_title('Child - Final Analysis Complete', meta),
                 fontsize=8, fontweight="bold", pad=0, y=1.05)

    handles, labels = ax.get_legend_handles_labels()
    _fill_info_panel(ax_info, handles=handles, labels=labels)
    return fig


# ===========================================================================
# Dispatcher — frame types we override go to the coloured variants above.
# Everything else falls through to the base dispatcher unchanged.
# ===========================================================================

_OVERRIDES = {
    'parallel_child_segments':             draw_parallel_child_segments,
    'parallel_child_shrinking':            draw_parallel_child_shrinking,
    'parallel_child_gradient_comparison':  draw_parallel_child_gradient_comparison,
    'parallel_child_gradient_lines':       draw_parallel_child_gradient_lines,
    'parallel_child_gradient_pair_search': draw_parallel_child_gradient_pair_search,
    'parallel_child_dynamic_center_found': draw_parallel_child_dynamic_center_found,
    'parallel_child_find_matching_pairs':  draw_parallel_child_find_matching_pairs,
    'parallel_child_dmax_circle':          draw_parallel_child_dmax_circle,
    'parallel_child_progressive_segments': draw_parallel_child_progressive_segments,
    'parallel_child_d_values':             draw_parallel_child_d_values,
    'export_silhouettes_only':             draw_export_silhouettes_only,
    # 14.14 — force same silhouette rendering as 14.13.
    'parallel_child_final':                draw_parallel_child_final,
}

_GRANDCHILD_OVERRIDES = {
    'parallel_grandchild_dmax_circle':          draw_parallel_child_dmax_circle,
    'parallel_grandchild_progressive_segments': draw_parallel_child_progressive_segments,
    'parallel_grandchild_shrinking':            draw_parallel_child_shrinking,
    'parallel_grandchild_d_values':             draw_parallel_child_d_values,
    'parallel_grandchild_gradient_comparison':  draw_parallel_child_gradient_comparison,
    'parallel_grandchild_gradient_lines':       draw_parallel_child_gradient_lines,
    'parallel_grandchild_gradient_pair_search': draw_parallel_child_gradient_pair_search,
    'parallel_grandchild_dynamic_center_found': draw_parallel_child_dynamic_center_found,
}


def _render_colored_grandchild_frame(name, P, meta, figsize, opts):
    """Reuse Child color rules for Grandchild frames without losing context."""
    if name == 'parallel_grandchild_find_matching_pairs':
        return _base._draw_grandchild_composed_summary(
            P, meta, figsize, opts, final=False,
            use_cluster_silhouette_colors=True
        )
    if name == 'parallel_grandchild_final':
        return _base._draw_grandchild_composed_summary(
            P, meta, figsize, opts, final=True
        )

    renderer = _GRANDCHILD_OVERRIDES.get(name)
    if renderer is None:
        return None

    fig = renderer(P, meta, figsize, opts)
    if fig is None or not P.get('context_child_analyses') or not fig.axes:
        return fig

    _base._draw_recursive_context(
        fig.axes[0], P, include_labels=False, include_silhouettes=True
    )
    return fig


def draw_frame_matplotlib(frame, figsize=(7, 7), global_rmax=None,
                          render_options=None):
    """Dispatch a frame to the coloured-variant renderer when one exists,
    otherwise fall through to the base implementation.

    Signature matches :func:`pyqt_matplotlib_vis.draw_frame_matplotlib` so
    callers can switch modules at import time with no other changes.
    """
    name = getattr(frame, 'name', None)

    is_colored_grandchild = (
        name in _GRANDCHILD_OVERRIDES
        or name in {
            'parallel_grandchild_find_matching_pairs',
            'parallel_grandchild_final',
        }
    )

    if name in _OVERRIDES or is_colored_grandchild:
        # Duplicate the tiny preamble the base dispatcher runs before any
        # draw call (sets global rmax / centroid / zoom / axes shift).
        # Calling into _base.draw_frame_matplotlib would run the wrong
        # drawing function, so we replicate its setup here.
        from vmeans.rendering.base import (
            _set_global_ranges, _compute_axes_shift,
            _DEFAULT_RENDER_OPTIONS as _DEFAULTS, draw_error_frame,
        )
        P = getattr(frame, 'payload', {}) or {}
        meta = getattr(frame, 'meta', {}) or {}
        opts = render_options if render_options is not None else _DEFAULTS
        _base._FRAME_DISPLAY_RMAX = P.get('display_rmax')

        if (name.startswith('parallel_child_')
                or name.startswith('parallel_grandchild_')) \
                and P.get('display_rmax') is not None:
            P = dict(P)
            parent_r = np.asarray(P.get('parent_r', []), dtype=float)
            P['parent_r'] = np.append(parent_r, float(P['display_rmax']))

        zoom_factor = opts.zoom_factor if opts else 1.0
        if global_rmax:
            centroid = P.get('centroid', None)
            _set_global_ranges(global_rmax, centroid, zoom_factor)
        elif 'rmax' in P:
            centroid = P.get('centroid', None)
            _set_global_ranges(P['rmax'], centroid, zoom_factor)
        else:
            _set_global_ranges(None, None, zoom_factor)

        if 'r' in P and 'theta' in P:
            r_vals = np.asarray(P['r'], dtype=float)
            theta_vals = np.asarray(P['theta'], dtype=float)
            _compute_axes_shift(r_vals, theta_vals)

        try:
            if is_colored_grandchild:
                return _render_colored_grandchild_frame(
                    name, P, meta, figsize, opts
                )
            return _OVERRIDES[name](P, meta, figsize, opts)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return draw_error_frame(name, str(exc), figsize)

    # Fallback: the non-coloured renderer now lives in the dispatch module.
    # Import lazily so the coloured override layer can be imported by the GUI
    # without creating an import cycle during module initialization.
    from vmeans.rendering.dispatch import draw_frame_matplotlib as _draw_base_frame
    return _draw_base_frame(
        frame, figsize=figsize, global_rmax=global_rmax,
        render_options=render_options,
    )
