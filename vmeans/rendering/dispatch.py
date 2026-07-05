from .base import *
from . import base as _base
from .parent_frames import *
from .child_frames import *
from .recursive_frames import *
from .export_frames import *
from .base import _fill_info_panel
from .recursive_frames import _draw_grandchild_composed_summary, _draw_recursive_context

def _render_grandchild_frame(name, P, meta, figsize, opts):
    """Render a grandchild step while preserving the completed child context."""
    render_payload = dict(P)
    display_rmax = P.get('display_rmax')
    if display_rmax is not None:
        parent_r = np.asarray(P.get('parent_r', []), dtype=float)
        render_payload['parent_r'] = np.append(parent_r, float(display_rmax))

    if name == 'parallel_grandchild_find_matching_pairs':
        return _draw_grandchild_composed_summary(
            render_payload, meta, figsize, opts, final=False
        )
    if name == 'parallel_grandchild_final':
        return _draw_grandchild_composed_summary(
            render_payload, meta, figsize, opts, final=True
        )

    renderers = {
        'parallel_grandchild_raw_data': draw_parallel_child_raw_data,
        'parallel_grandchild_center': draw_parallel_child_center,
        'parallel_grandchild_highlight_dmax': draw_parallel_child_highlight_dmax,
        'parallel_grandchild_dmax_edge': draw_parallel_child_dmax_edge,
        'parallel_grandchild_dmax_circle': draw_parallel_child_dmax_circle,
        'parallel_grandchild_progressive_segments': draw_parallel_child_progressive_segments,
        'parallel_grandchild_shrinking': draw_parallel_child_shrinking,
        'parallel_grandchild_d_values': draw_parallel_child_d_values,
        'parallel_grandchild_gradient_comparison': draw_parallel_child_gradient_comparison,
        'parallel_grandchild_gradient_lines': draw_parallel_child_gradient_lines,
        'parallel_grandchild_gradient_pair_search': draw_parallel_child_gradient_pair_search,
        'parallel_grandchild_dynamic_center_found': draw_parallel_child_dynamic_center_found,
    }
    renderer = renderers.get(name)
    if renderer is None:
        return None

    fig = renderer(render_payload, meta, figsize, opts)
    if fig is None or not P.get('context_child_analyses') or not fig.axes:
        return fig

    ax = fig.axes[0]
    _draw_recursive_context(
        ax,
        P,
        include_labels=name.endswith('_final'),
        include_silhouettes=True
    )

    if name.endswith('_final') and len(fig.axes) > 1:
        existing_legend = fig.axes[1].get_legend()
        if existing_legend is not None:
            existing_legend.remove()
        handles, labels = ax.get_legend_handles_labels()
        _fill_info_panel(fig.axes[1], handles=handles, labels=labels)
    return fig


# ============================================================================
# Main Render Function
# ============================================================================

def draw_frame_matplotlib(frame: StepFrame, figsize=(7, 7), global_rmax=None,
                          render_options: RenderOptions = None) -> Figure:
    """
    Main frame renderer

    Args:
        frame: StepFrame to render
        figsize: Figure size tuple
        global_rmax: Global maximum radius for consistent scaling
        render_options: RenderOptions instance for customizing display
    """
    name = frame.name
    P = frame.payload
    meta = frame.meta or {}
    _base._FRAME_DISPLAY_RMAX = P.get('display_rmax')

    if name.startswith('parallel_child_') and P.get('display_rmax') is not None:
        P = dict(P)
        parent_r = np.asarray(P.get('parent_r', []), dtype=float)
        P['parent_r'] = np.append(parent_r, float(P['display_rmax']))

    # Use default options if not provided
    opts = render_options if render_options is not None else _DEFAULT_RENDER_OPTIONS

    # Set global point size multiplier from render options
    _base._GLOBAL_POINT_MULTIPLIER = opts.point_size if opts else 1.0

    # Set global zoom factor from render options
    zoom_factor = opts.zoom_factor if opts else 1.0

    if global_rmax:
        centroid = P.get('centroid', None)
        _set_global_ranges(global_rmax, centroid, zoom_factor)
    elif 'rmax' in P:
        centroid = P.get('centroid', None)
        _set_global_ranges(P['rmax'], centroid, zoom_factor)
    else:
        # Just set zoom even if no rmax
        _set_global_ranges(None, None, zoom_factor)

    # Pre-compute the axes shift before dispatching so every step
    # sees the same _GLOBAL_AXES_SHIFT value
    if 'r' in P and 'theta' in P:
        r_vals = np.asarray(P['r'], dtype=float)
        theta_vals = np.asarray(P['theta'], dtype=float)
        _compute_axes_shift(r_vals, theta_vals)

    try:
        if name.startswith('parallel_grandchild_'):
            grandchild_figure = _render_grandchild_frame(name, P, meta, figsize, opts)
            if grandchild_figure is not None:
                return grandchild_figure
        if name == 'show_raw_data':
            return draw_show_raw_data(P, meta, figsize, opts)
        elif name == 'show_center':
            return draw_show_center(P, meta, figsize, opts)
        elif name == 'phase_transition':
            return draw_phase_transition(P, meta, figsize, opts)
        elif name == 'init_center':
            return draw_init_center(P, meta, figsize, opts)
        elif name == 'cartesian_with_polar_grid':
            return draw_cartesian_with_polar_grid(P, meta, figsize, opts)
        elif name == 'highlight_dmax':
            return draw_highlight_dmax(P, meta, figsize, opts)
        elif name == 'dmax_circle':
            return draw_dmax_circle(P, meta, figsize, opts)
        elif name == 'dmax_circle_only':
            return draw_dmax_circle_only(P, meta, figsize, opts)
        elif name == 'dmax_circle_animating':
            return draw_dmax_circle_animating(P, meta, figsize, opts)
        elif name == 'show_segments':
            return draw_show_segments(P, meta, figsize, opts)
        elif name == 'progressive_segments':
            return draw_progressive_segments(P, meta, figsize, opts)
        elif name == 'segment_shrinking':
            return draw_segment_shrinking(P, meta, figsize, opts)
        elif name == 'progressive_shrinking':
            return draw_progressive_shrinking(P, meta, figsize, opts)
        elif name == 'segmentation':
            return draw_segmentation(P, meta, figsize, opts)
        elif name == 'd_values':
            return draw_d_values(P, meta, figsize, opts)
        elif name == 'gradient_scan_progress':
            return draw_gradient_scan_progress(P, meta, figsize, opts)
        elif name == 'gradient_comparison':
            return draw_gradient_comparison(P, meta, figsize, opts)
        elif name == 'gradient_lines_final':
            return draw_gradient_lines_final(P, meta, figsize, opts)
        elif name == 'dynamic_center_found':
            return draw_dynamic_center_found(P, meta, figsize, opts)
        elif name == 'find_matching_gradient_pairs':
            return draw_find_matching_gradient_pairs(P, meta, figsize, opts)
        elif name == 'gradient_pair_search':
            return draw_gradient_pair_search(P, meta, figsize, opts)
        elif name == 'no_gradient_boundaries':
            return draw_no_gradient_boundaries(P, meta, figsize, opts)
        elif name == 'parallel_child_init':
            return draw_parallel_child_init(P, meta, figsize, opts)
        elif name == 'parallel_child_segments':
            return draw_parallel_child_segments(P, meta, figsize, opts)
        elif name in ('parallel_child_shrinking', 'parallel_grandchild_shrinking'):
            return draw_parallel_child_shrinking(P, meta, figsize, opts)
        elif name in ('parallel_child_gradient_comparison', 'parallel_grandchild_gradient_comparison'):
            return draw_parallel_child_gradient_comparison(P, meta, figsize, opts)
        elif name in ('parallel_child_gradient_lines', 'parallel_grandchild_gradient_lines'):
            return draw_parallel_child_gradient_lines(P, meta, figsize, opts)
        elif name in ('parallel_child_gradient_pair_search', 'parallel_grandchild_gradient_pair_search'):
            return draw_parallel_child_gradient_pair_search(P, meta, figsize, opts)
        elif name in ('parallel_child_dynamic_center_found', 'parallel_grandchild_dynamic_center_found'):
            return draw_parallel_child_dynamic_center_found(P, meta, figsize, opts)
        elif name in ('parallel_child_find_matching_pairs', 'parallel_grandchild_find_matching_pairs'):
            return draw_parallel_child_find_matching_pairs(P, meta, figsize, opts)
        elif name in ('parallel_child_final', 'parallel_grandchild_final'):
            return draw_parallel_child_final(P, meta, figsize, opts)
        # Additional parallel child frame types
        elif name in ('parallel_child_raw_data', 'parallel_grandchild_raw_data'):
            return draw_parallel_child_raw_data(P, meta, figsize, opts)
        elif name in ('parallel_child_center', 'parallel_grandchild_center'):
            return draw_parallel_child_center(P, meta, figsize, opts)
        elif name in ('parallel_child_highlight_dmax', 'parallel_grandchild_highlight_dmax'):
            return draw_parallel_child_highlight_dmax(P, meta, figsize, opts)
        elif name in ('parallel_child_dmax_edge', 'parallel_grandchild_dmax_edge'):
            return draw_parallel_child_dmax_edge(P, meta, figsize, opts)
        elif name in ('parallel_child_dmax_circle', 'parallel_grandchild_dmax_circle'):
            return draw_parallel_child_dmax_circle(P, meta, figsize, opts)
        elif name in ('parallel_child_progressive_segments', 'parallel_grandchild_progressive_segments'):
            return draw_parallel_child_progressive_segments(P, meta, figsize, opts)
        elif name in ('parallel_child_d_values', 'parallel_grandchild_d_values'):
            return draw_parallel_child_d_values(P, meta, figsize, opts)
        # Export-quality silhouettes-only frame (hidden feature)
        elif name == 'export_silhouettes_only':
            return draw_export_silhouettes_only(P, meta, figsize, opts)
        else:
            return draw_unknown_frame(name, meta, figsize)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return draw_error_frame(name, str(e), figsize)
