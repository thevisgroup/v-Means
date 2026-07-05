"""
Enhanced Visible Runner - Desktop Version
Generates animation frames for the polar segmentation visualization

UPDATED:
- Added animated circle drawing (Step 5)
- Enhanced gradient comparison with visible values
- Added gradient pair search visualization
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class StepFrame:
    """Represents a single animation frame"""
    name: str
    payload: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = field(default_factory=dict)


def _analyze_recursive_region(
        region_points: np.ndarray,
        segments: int,
        gradient_threshold_ratio: float,
        global_max_r: float,
        region_id: int,
        parent_region_id: Optional[int] = None,
        recursion_depth: int = 1
) -> Dict[str, Any]:
    """Run one local Visible k-Means pass using the root d_max reference."""
    from vmeans.core_analysis import (
        compute_better_center, detect_gradient_boundaries,
        detect_dynamic_centers, segment_points_by_region
    )
    from vmeans.data import convert_to_polar
    from vmeans.segment import segment_points_by_theta

    local_segments = max(int(segments), 2)
    region_centroid = compute_better_center(region_points, method='centroid')
    r_sub, theta_sub = convert_to_polar(region_points, region_centroid)
    max_r_sub = np.max(r_sub) if len(r_sub) > 0 else 1.0
    segment_points_sub, theta_edges_sub = segment_points_by_theta(
        r_sub, theta_sub, s=local_segments
    )
    dmax_sub = [max(seg) if seg else 0.0 for seg in segment_points_sub]

    gradient_boundaries_sub = detect_gradient_boundaries(
        dmax_sub,
        threshold_ratio=gradient_threshold_ratio,
        global_max=global_max_r
    )

    additional_centers_sub, gradient_angles_sub, center_labels_sub = [], [], []
    regions_sub = []
    if gradient_boundaries_sub:
        additional_centers_sub, gradient_angles_sub, center_labels_sub = detect_dynamic_centers(
            region_points, r_sub, theta_sub, gradient_boundaries_sub,
            theta_edges_sub, segment_points_sub
        )
        if additional_centers_sub:
            regions_sub = segment_points_by_region(
                region_points, theta_sub, gradient_angles_sub, additional_centers_sub
            )

    return {
        'region_id': region_id,
        'parent_region_id': parent_region_id,
        'recursion_depth': recursion_depth,
        'center_global': region_centroid,
        'r': r_sub,
        'theta': theta_sub,
        'theta_edges': theta_edges_sub,
        'rmax': max_r_sub,
        'segment_points': segment_points_sub,
        'dmax_list': dmax_sub,
        'gradient_boundaries': gradient_boundaries_sub,
        'additional_centers': additional_centers_sub,
        'gradient_angles': gradient_angles_sub,
        'center_labels': center_labels_sub,
        'regions': regions_sub,
        'points': region_points,
        'segments': local_segments,
        'grandchild_analyses': [],
    }


def _build_parallel_level_frames(
        points: np.ndarray,
        parent_centroid: np.ndarray,
        parent_r: np.ndarray,
        parent_theta: np.ndarray,
        analyses: List[Dict[str, Any]],
        level_segments: int,
        gradient_threshold_ratio: float,
        global_max_r: float,
        step_prefix: int,
        level_key: str,
        level_label: str,
        context_analyses: Optional[List[Dict[str, Any]]] = None,
        display_rmax: Optional[float] = None,
        context_segments: Optional[int] = None
) -> List[StepFrame]:
    """Build the complete synchronized animation sequence for one recursion level."""
    frames: List[StepFrame] = []
    frame_prefix = f'parallel_{level_key}'
    level_segments = max(int(level_segments), 2)

    def payload(current_analyses=None, **extra):
        data = {
            'all_points': points.copy(),
            'parent_centroid': parent_centroid.copy(),
            'child_analyses': analyses if current_analyses is None else current_analyses,
            'parent_r': parent_r.copy(),
            'parent_theta': parent_theta.copy(),
            'level_label': level_label,
            # draw_parallel_child_final halves this value to get leaf resolution.
            'segments': level_segments * 2,
            'context_child_analyses': context_analyses or [],
            'context_segments': context_segments or level_segments * 2,
        }
        if display_rmax is not None:
            data['display_rmax'] = float(display_rmax)
        data.update(extra)
        return data

    def meta(substep, description, **extra):
        data = {'step': f'{step_prefix}.{substep}', 'description': description}
        data.update(extra)
        return data

    frames.append(StepFrame(
        f'{frame_prefix}_raw_data',
        payload(show_center=False),
        meta(1, f'Recurse to {level_label} Level')
    ))
    frames.append(StepFrame(
        f'{frame_prefix}_center',
        payload(show_center=True),
        meta(2, f'{level_label} - Compute Center Points')
    ))
    frames.append(StepFrame(
        f'{frame_prefix}_highlight_dmax',
        payload(),
        meta(3, f'{level_label} - Highlight d_max Points')
    ))
    frames.append(StepFrame(
        f'{frame_prefix}_dmax_edge',
        payload(),
        meta(4, f'{level_label} - Draw Edge to d_max')
    ))
    frames.append(StepFrame(
        f'{frame_prefix}_dmax_circle',
        payload(circle_progress=1.0),
        meta(5, f'{level_label} - Draw d_max Silhouettes', frame=1, total_frames=1)
    ))

    for seg_idx in range(level_segments):
        analyses_frame = []
        for analysis in analyses:
            analysis_frame = analysis.copy()
            analysis_frame['visible_segments'] = seg_idx + 1
            analyses_frame.append(analysis_frame)
        frames.append(StepFrame(
            f'{frame_prefix}_progressive_segments',
            payload(analyses_frame),
            meta(
                6,
                f'{level_label} - Divide into Segments ({seg_idx + 1}/{level_segments})',
                frame=seg_idx + 1,
                total_frames=level_segments
            )
        ))

    for seg_idx in range(level_segments):
        analyses_frame = []
        for analysis in analyses:
            current_heights = []
            for seg_j in range(len(analysis['dmax_list'])):
                has_data = len(analysis['segment_points'][seg_j]) > 0
                if seg_j <= seg_idx:
                    current_height = analysis['dmax_list'][seg_j] if has_data else 0
                else:
                    current_height = analysis['rmax']
                current_heights.append(current_height)

            analysis_frame = analysis.copy()
            analysis_frame['current_heights'] = current_heights
            analysis_frame['current_segment_idx'] = seg_idx
            analyses_frame.append(analysis_frame)

        frames.append(StepFrame(
            f'{frame_prefix}_shrinking',
            payload(analyses_frame),
            meta(
                7,
                f'{level_label} - Reduce Empty Space ({seg_idx + 1}/{level_segments})',
                frame=seg_idx + 1,
                total_frames=level_segments
            )
        ))

    frames.append(StepFrame(
        f'{frame_prefix}_d_values',
        payload(),
        meta(8, f'{level_label} - Finding the Large Gradients')
    ))

    threshold_global = gradient_threshold_ratio * global_max_r
    all_comparisons = []
    max_comparisons = 0
    for analysis in analyses:
        comparisons = []
        dmax_values = analysis['dmax_list']
        segment_points = analysis['segment_points']
        for i in range(len(dmax_values)):
            j = (i + 1) % len(dmax_values)
            if len(segment_points[i]) == 0 and len(segment_points[j]) == 0:
                continue
            val_i = dmax_values[i] if dmax_values[i] > 0 else 0
            val_j = dmax_values[j] if dmax_values[j] > 0 else 0
            comparisons.append((
                i, j, abs(val_i - val_j) > threshold_global,
                val_i, val_j, threshold_global
            ))
        all_comparisons.append(comparisons)
        max_comparisons = max(max_comparisons, len(comparisons))

    for frame_idx in range(max_comparisons):
        analyses_frame = []
        for analysis_idx, analysis in enumerate(analyses):
            comparisons = all_comparisons[analysis_idx]
            analysis_frame = analysis.copy()
            found_so_far = [
                (comparison[0], comparison[1])
                for comparison in comparisons[:frame_idx + 1]
                if comparison[2]
            ]
            analysis_frame['found_gradients_so_far'] = found_so_far
            if frame_idx < len(comparisons):
                seg_i, seg_j, has_gradient, val_i, val_j, threshold = comparisons[frame_idx]
                analysis_frame['current_comparison'] = {
                    'seg_i': seg_i,
                    'seg_j': seg_j,
                    'has_gradient': has_gradient,
                    'val_i': val_i,
                    'val_j': val_j,
                    'threshold': threshold,
                }
                analysis_frame['comparison_complete'] = False
            else:
                analysis_frame['current_comparison'] = None
                analysis_frame['comparison_complete'] = True
            analysis_frame['gradient_count'] = len(found_so_far)
            analysis_frame['total_gradients'] = sum(
                1 for comparison in comparisons if comparison[2]
            )
            analyses_frame.append(analysis_frame)

        frames.append(StepFrame(
            f'{frame_prefix}_gradient_comparison',
            payload(analyses_frame),
            meta(
                9,
                f'{level_label} - Gradient Comparison ({frame_idx + 1}/{max_comparisons})',
                comparison=frame_idx + 1,
                total_comparisons=max_comparisons
            )
        ))

    frames.append(StepFrame(
        f'{frame_prefix}_gradient_lines',
        payload(),
        meta(10, f'{level_label} - Gradient Lines (G-lines)')
    ))

    max_pairs = max(
        (len(analysis.get('gradient_angles', [])) for analysis in analyses),
        default=0
    )
    for pair_idx in range(max_pairs):
        frames.append(StepFrame(
            f'{frame_prefix}_gradient_pair_search',
            payload(current_pair_idx=pair_idx),
            meta(
                11,
                f'{level_label} - Search Gradient Pairs ({pair_idx + 1}/{max_pairs})',
                pair=pair_idx + 1,
                total_pairs=max_pairs
            )
        ))

    max_centers = max(
        (len(analysis.get('additional_centers', [])) for analysis in analyses),
        default=0
    )
    for center_idx in range(max_centers):
        frames.append(StepFrame(
            f'{frame_prefix}_dynamic_center_found',
            payload(centers_shown=center_idx + 1),
            meta(
                12,
                f'{level_label} - Compute New Centers ({center_idx + 1}/{max_centers})',
                center_count=center_idx + 1,
                total_centers=max_centers
            )
        ))

    frames.append(StepFrame(
        f'{frame_prefix}_find_matching_pairs',
        payload(),
        meta(13, f'{level_label} - Algorithm 1 Pass Complete')
    ))
    frames.append(StepFrame(
        f'{frame_prefix}_final',
        payload(),
        meta(14, f'{level_label} - Final Analysis Complete')
    ))
    return frames
