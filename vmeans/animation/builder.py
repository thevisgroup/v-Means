from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from .recursive import StepFrame, _analyze_recursive_region, _build_parallel_level_frames

def build_enhanced_visible_frames(
        points: np.ndarray,
        segments: int = 60,
        center_method: str = 'auto',
        gradient_threshold_ratio: float = 0.25,
        enable_recursion: bool = False,
        max_recursion_depth: int = 1,
        circle_animation_frames: int = 20,  # NEW: configurable circle animation frames
        export_final_silhouettes: bool = False  # Hidden: add export-quality silhouettes frame
) -> List[StepFrame]:
    """
    Build animation frame sequence.

    Features:
    - Progressive segment line addition
    - Progressive shrinking animation
    - ANIMATED circle drawing (NEW)
    - Enhanced gradient comparison with visible values (NEW)
    - Gradient pair search visualization (NEW)
    - Parallel child cluster analysis
    - Multi-level recursion (up to max_recursion_depth levels)

    Recursion rules:
    - Only recurse if gradient_boundaries >= 2 (need at least 2 gradients to form regions)
    - Maximum depth controlled by max_recursion_depth parameter
    """
    from vmeans.core_analysis import (
        compute_better_center, detect_gradient_boundaries,
        detect_dynamic_centers, segment_points_by_region
    )
    from vmeans.data import convert_to_polar
    from vmeans.segment import segment_points_by_theta

    frames = []

    print(f"\n🎬 Building animation frames...")
    max_recursion_depth = max(1, min(int(max_recursion_depth), 2))
    print(
        f"Points: {len(points)}, Segments: {segments}, "
        f"Recursion: {enable_recursion}, Depth: {max_recursion_depth}"
    )

    # =============================
    # Execute algorithm once
    # =============================

    # Main analysis
    centroid = compute_better_center(points, method=center_method)
    r, theta = convert_to_polar(points, centroid)
    max_r = np.max(r) if len(r) > 0 else 1.0
    segment_points_data, theta_edges = segment_points_by_theta(r, theta, s=segments)
    dmax_list = [max(seg) if seg else 0.0 for seg in segment_points_data]

    # Find d_max point
    dmax_idx = np.argmax(r)

    # Gradient detection
    polar_gradient_boundaries = detect_gradient_boundaries(
        dmax_list, threshold_ratio=gradient_threshold_ratio, global_max=max_r
    )

    # Dynamic center detection
    additional_centers, gradient_angles, center_labels = [], [], []
    regions = []

    if polar_gradient_boundaries:
        additional_centers, gradient_angles, center_labels = detect_dynamic_centers(
            points, r, theta, polar_gradient_boundaries, theta_edges, segment_points_data
        )
        if additional_centers:
            regions = segment_points_by_region(points, theta, gradient_angles, additional_centers)

    # Recursive analysis (if enabled)
    child_analyses = []
    if enable_recursion and len(polar_gradient_boundaries) >= 2 and additional_centers and regions:
        print(f"🔄 Recursion: {len(regions)} children")

        for region_idx, region_point_indices in enumerate(regions):
            if len(region_point_indices) == 0:
                continue

            region_points = points[region_point_indices]
            child_analysis = _analyze_recursive_region(
                region_points,
                segments=max(segments // 2, 2),
                gradient_threshold_ratio=gradient_threshold_ratio,
                global_max_r=max_r,
                region_id=region_idx + 1,
                recursion_depth=1
            )

            if max_recursion_depth >= 2 and child_analysis['regions']:
                grandchild_segments = max(segments // 4, 2)
                for grandchild_idx, grandchild_indices in enumerate(child_analysis['regions']):
                    if len(grandchild_indices) == 0:
                        continue
                    grandchild_points = region_points[grandchild_indices]
                    grandchild_analysis = _analyze_recursive_region(
                        grandchild_points,
                        segments=grandchild_segments,
                        gradient_threshold_ratio=gradient_threshold_ratio,
                        global_max_r=max_r,
                        region_id=grandchild_idx + 1,
                        parent_region_id=region_idx + 1,
                        recursion_depth=2
                    )
                    child_analysis['grandchild_analyses'].append(grandchild_analysis)

            child_analyses.append(child_analysis)
    elif enable_recursion and len(polar_gradient_boundaries) < 2:
        print(f"   Skipping recursion: <2 parent gradients")

    recursive_display_rmax = max_r
    recursive_analyses = list(child_analyses)
    recursive_analyses.extend(
        grandchild
        for child in child_analyses
        for grandchild in child.get('grandchild_analyses', [])
    )
    for analysis in recursive_analyses:
        analysis_center = analysis.get('center_global')
        analysis_rmax = analysis.get('rmax', 0)
        if analysis_center is None or analysis_rmax <= 0:
            continue
        center_distance = np.linalg.norm(analysis_center - centroid)
        recursive_display_rmax = max(
            recursive_display_rmax,
            center_distance + analysis_rmax
        )

    # =============================
    # Build frame sequence
    # =============================



    # Step 1: Show raw data (polar view, no center point)
    frames.append(StepFrame('show_raw_data', {
        'points': points.copy(),
        'centroid': centroid.copy(),
        'r': r.copy(),
        'theta': theta.copy(),
        'rmax': max_r
    }, meta={'step': 1, 'description': 'Show Raw Data'}))

    # Step 2: Show computing the center
    frames.append(StepFrame('show_center', {
        'points': points.copy(),
        'centroid': centroid.copy(),
        'r': r.copy(),
        'theta': theta.copy(),
        'rmax': max_r
    }, meta={'step': 2, 'description': 'Compute Center Point'}))

    # Step 3: Highlight d_max point
    frames.append(StepFrame('highlight_dmax', {
        'points': points.copy(),
        'centroid': centroid.copy(),
        'r': r.copy(),
        'theta': theta.copy(),
        'rmax': max_r,
        'dmax_idx': dmax_idx,
        'dmax_value': max_r
    }, meta={'step': 3, 'description': 'Highlight Furthest Point (d_max)'}))

    # Step 4: Draw d_max circle + ray
    frames.append(StepFrame('dmax_circle', {
        'points': points.copy(),
        'centroid': centroid.copy(),
        'r': r.copy(),
        'theta': theta.copy(),
        'rmax': max_r,
        'dmax_idx': dmax_idx,
        'show_circle': False
    }, meta={'step': 4, 'description': 'Draw Edge from Center to d_max'}))

    # =============================
    # Step 5 - Animated silhouette drawing (CHANGED: Circle → Silhouette)
    # =============================

    for frame_idx in range(circle_animation_frames):
        progress = (frame_idx + 1) / circle_animation_frames
        frames.append(StepFrame('dmax_circle_animating', {
            'points': points.copy(),
            'centroid': centroid.copy(),
            'r': r.copy(),
            'theta': theta.copy(),
            'rmax': max_r,
            'dmax_idx': dmax_idx,
            'progress': progress
        }, meta={
            'step': 5,
            'frame': frame_idx + 1,
            'total_frames': circle_animation_frames,
            'description': f'Drawing d_max Silhouette ({int(progress*100)}%)'
        }))

    # Step 5 final: Complete silhouette
    frames.append(StepFrame('dmax_circle_only', {
        'points': points.copy(),
        'centroid': centroid.copy(),
        'r': r.copy(),
        'theta': theta.copy(),
        'rmax': max_r,
        'dmax_idx': dmax_idx,
        'show_circle': True
    }, meta={'step': 5, 'description': 'Draw d_max Silhouette (Complete)'}))

    # Step 6: Progressive segment boundaries

    for seg_idx in range(segments):
        frames.append(StepFrame('progressive_segments', {
            'points': points.copy(),
            'r': r.copy(),
            'theta': theta.copy(),
            'theta_edges': theta_edges.copy(),
            'visible_edges_count': seg_idx + 1,
            'rmax': max_r,
            'dmax_idx': dmax_idx,
            'show_circle': True
        }, meta={'step': 6, 'description': f'Divide region into segments ({seg_idx + 1}/{segments})'}))

    # Step 7: Reduce Empty Space

    for seg_idx in range(segments):
        current_heights = []

        for seg_j in range(segments):
            has_data = len(segment_points_data[seg_j]) > 0

            if seg_j < seg_idx:
                # Already processed: use actual dmax if has data, else 0 (removed)
                current_height = dmax_list[seg_j] if has_data else 0
            elif seg_j == seg_idx:
                # Currently processing: shrink to dmax if has data, else remove
                current_height = dmax_list[seg_j] if has_data else 0
            else:
                # Not yet processed: stay at max_r (full height)
                current_height = max_r

            current_heights.append(current_height)

        frames.append(StepFrame('progressive_shrinking', {
            'points': points.copy(),
            'r': r.copy(),
            'theta': theta.copy(),
            'theta_edges': theta_edges.copy(),
            'rmax': max_r,
            'current_heights': current_heights,
            'segment_points': segment_points_data,
            'current_segment_idx': seg_idx,
            'show_circle': True
        }, meta={'step': 7, 'frame': seg_idx + 1, 'total_frames': segments,
                 'description': f'Reduce Empty Space ({seg_idx + 1}/{segments})'}))

    # Step 8: Finding the Large Gradients (CHANGED: added "Large")
    frames.append(StepFrame('d_values', {
        'r': r.copy(),
        'theta': theta.copy(),
        'theta_edges': theta_edges.copy(),
        'rmax': max_r,
        'dmax_list': dmax_list.copy(),
        'segment_points': segment_points_data
    }, meta={'step': 8, 'description': 'Finding the Large Gradients'}))

    # =============================
    # ENHANCED: Step 9 - Gradient comparison with visible values
    # Show meaningful comparisons only (skip empty vs empty)
    # =============================
    threshold = gradient_threshold_ratio * max_r  # Calculate absolute threshold

    # Collect meaningful comparisons (skip empty vs empty)
    all_comparisons = []
    for i in range(segments):
        j = (i + 1) % segments
        val_i = dmax_list[i] if dmax_list[i] > 0 else 0
        val_j = dmax_list[j] if dmax_list[j] > 0 else 0

        is_empty_i = len(segment_points_data[i]) == 0
        is_empty_j = len(segment_points_data[j]) == 0

        # Skip if BOTH segments are empty - not interesting
        if is_empty_i and is_empty_j:
            continue

        # Use same logic as detect_gradient_boundaries: only check value difference
        diff = abs(val_i - val_j)
        has_gradient = diff > threshold

        all_comparisons.append((i, j, has_gradient))

    # Count total gradients for display
    total_gradients = sum(1 for _, _, g in all_comparisons if g)

    found_gradients_so_far = []
    gradient_count = 0

    for idx, (seg_i, seg_j, has_gradient) in enumerate(all_comparisons):
        if has_gradient:
            found_gradients_so_far.append((seg_i, seg_j))
            gradient_count += 1

        frames.append(StepFrame('gradient_comparison', {
            'r': r.copy(),
            'theta': theta.copy(),
            'theta_edges': theta_edges.copy(),
            'rmax': max_r,
            'dmax_list': dmax_list.copy(),
            'segment_points': segment_points_data,
            'seg_i': seg_i,
            'seg_j': seg_j,
            'has_gradient': has_gradient,
            'found_gradients_so_far': list(found_gradients_so_far),
            'threshold': threshold,
            'gradient_count': gradient_count,
            'total_gradients': total_gradients
        }, meta={
            'step': 9,
            'comparison': idx + 1,
            'total_comparisons': len(all_comparisons),
            'description': f'Gradient Comparison {idx + 1}/{len(all_comparisons)}'
        }))

    # Step 10: Final gradient lines (if detected)
    if polar_gradient_boundaries and gradient_angles:
        frames.append(StepFrame('gradient_lines_final', {
            'r': r.copy(),
            'theta': theta.copy(),
            'theta_edges': theta_edges.copy(),
            'rmax': max_r,
            'gradient_angles': gradient_angles.copy(),
            'centroid': centroid.copy(),
            'show_circle': True
        }, meta={'step': 10, 'description': 'Gradient Lines (G-lines)'}))
    else:
        frames.append(StepFrame('no_gradient_boundaries', {
            'r': r.copy(),
            'theta': theta.copy(),
            'theta_edges': theta_edges.copy(),
            'rmax': max_r,
            'centroid': centroid.copy(),
            'show_circle': True
        }, meta={'step': 10, 'description': 'No Gradient Boundaries Detected'}))

    # =============================
    # Step 11: Search Gradient Pairs (MUST come BEFORE center discovery)
    # This is the search process - checking which G-line intervals have data
    # =============================
    if len(gradient_angles) >= 2:

        centers_found_so_far = []
        center_idx_mapping = []  # Track which pairs have centers

        for pair_idx in range(len(gradient_angles)):
            next_idx = (pair_idx + 1) % len(gradient_angles)
            g_current = gradient_angles[pair_idx]
            g_next = gradient_angles[next_idx]

            # Check if this interval has points (same logic as detect_dynamic_centers)
            crosses_zero = g_current > g_next
            points_in_range_count = 0

            for j, angle in enumerate(theta):
                if crosses_zero:
                    in_range = (angle >= g_current) or (angle <= g_next)
                else:
                    in_range = g_current <= angle <= g_next
                if in_range:
                    points_in_range_count += 1

            has_points = points_in_range_count > 0

            # Track which centers we've found so far
            if has_points:
                center_idx_mapping.append(len(centers_found_so_far))
                if len(centers_found_so_far) < len(additional_centers):
                    centers_found_so_far.append(additional_centers[len(centers_found_so_far)])

            frames.append(StepFrame('gradient_pair_search', {
                'points': points.copy(),
                'r': r.copy(),
                'theta': theta.copy(),
                'rmax': max_r,
                'theta_edges': theta_edges.copy(),
                'gradient_angles': gradient_angles.copy(),
                'centroid': centroid.copy(),
                'current_pair_idx': pair_idx,
                'points_in_range_count': points_in_range_count,
                'has_points': has_points,
                'centers_found_so_far': []  # Don't show centers yet during search
            }, meta={
                'step': 11,
                'pair': pair_idx + 1,
                'total_pairs': len(gradient_angles),
                'description': f'Search Gradient Pair G{pair_idx + 1}-G{next_idx + 1} ({pair_idx + 1}/{len(gradient_angles)})'
            }))

    # =============================
    # Step 12: Compute New Centers (CHANGED: was "Dynamic center discovery")
    # =============================
    if additional_centers:
        for i in range(len(additional_centers)):
            frames.append(StepFrame('dynamic_center_found', {
                'r': r.copy(),
                'theta': theta.copy(),
                'theta_edges': theta_edges.copy(),
                'rmax': max_r,
                'gradient_angles': gradient_angles.copy(),
                'centers_so_far': np.array(additional_centers[:i + 1]).copy(),
                'center_labels': center_labels[:i + 1].copy(),
                'centroid': centroid.copy(),
                'show_circle': True
            }, meta={
                'step': 12,
                'center_count': i + 1,
                'total_centers': len(additional_centers),
                'description': f'Compute New Centers - {center_labels[i]} ({i + 1}/{len(additional_centers)})'
            }))

    # Step 13: Algorithm 1 Pass Complete (CHANGED: was "Final Result - Regions Identified")
    if additional_centers and gradient_angles:
        regions = segment_points_by_region(points, theta, gradient_angles, additional_centers)

        frames.append(StepFrame('find_matching_gradient_pairs', {
            'points': points.copy(),
            'r': r.copy(),
            'theta': theta.copy(),
            'theta_edges': theta_edges.copy(),
            'rmax': max_r,
            'gradient_angles': gradient_angles.copy(),
            'additional_centers': additional_centers.copy(),
            'center_labels': center_labels.copy(),
            'centroid': centroid.copy(),
            'regions': regions,
            'show_circle': True
        }, meta={
            'step': 13,
            'description': 'Algorithm 1 Pass Complete'
        }))

    # Step 14: Parallel child analysis - SAME STEPS AS PARENT
    if enable_recursion and child_analyses:

        child_segments = segments // 2  # Children use half segments

        # =============================
        # Step 14.1: Recurse (CHANGED: was "Child - Show Raw Data")
        # =============================
        frames.append(StepFrame('parallel_child_raw_data', {
            'all_points': points.copy(),
            'parent_centroid': centroid.copy(),
            'child_analyses': child_analyses,
            'parent_r': r.copy(),
            'parent_theta': theta.copy(),
            'show_center': False
        }, meta={'step': 14.1, 'description': 'Recurse'}))

        # =============================
        # Step 14.2: Show center points
        # =============================
        frames.append(StepFrame('parallel_child_center', {
            'all_points': points.copy(),
            'parent_centroid': centroid.copy(),
            'child_analyses': child_analyses,
            'parent_r': r.copy(),
            'parent_theta': theta.copy(),
            'show_center': True
        }, meta={'step': 14.2, 'description': 'Child - Compute Center Points'}))

        # =============================
        # Step 14.3: Highlight d_max points
        # =============================
        frames.append(StepFrame('parallel_child_highlight_dmax', {
            'all_points': points.copy(),
            'parent_centroid': centroid.copy(),
            'child_analyses': child_analyses,
            'parent_r': r.copy(),
            'parent_theta': theta.copy()
        }, meta={'step': 14.3, 'description': 'Child - Highlight d_max Points'}))

        # =============================
        # Step 14.4: Draw edge from center to d_max
        # =============================
        frames.append(StepFrame('parallel_child_dmax_edge', {
            'all_points': points.copy(),
            'parent_centroid': centroid.copy(),
            'child_analyses': child_analyses,
            'parent_r': r.copy(),
            'parent_theta': theta.copy()
        }, meta={'step': 14.4, 'description': 'Child - Draw Edge to d_max'}))

        # =============================
        # Step 14.5: Draw d_max silhouettes (CHANGED: Circle → Silhouette)
        # =============================
        frames.append(StepFrame('parallel_child_dmax_circle', {
            'all_points': points.copy(),
            'parent_centroid': centroid.copy(),
            'child_analyses': child_analyses,
            'parent_r': r.copy(),
            'parent_theta': theta.copy(),
            'circle_progress': 1.0  # Will be replaced with GIF
        }, meta={
            'step': 14.5,
            'frame': 1,
            'total_frames': 1,
            'description': 'Child - Draw d_max Silhouette'
        }))

        # =============================
        # Step 14.6: Progressive segment lines
        # =============================
        for seg_idx in range(child_segments):
            child_analyses_frame = []
            for child in child_analyses:
                child_frame = child.copy()
                child_frame['visible_segments'] = seg_idx + 1
                child_analyses_frame.append(child_frame)

            frames.append(StepFrame('parallel_child_progressive_segments', {
                'all_points': points.copy(),
                'parent_centroid': centroid.copy(),
                'child_analyses': child_analyses_frame,
                'parent_r': r.copy(),
                'parent_theta': theta.copy()
            }, meta={
                'step': 14.6,
                'frame': seg_idx + 1,
                'total_frames': child_segments,
                'description': f'Child - Divide into Segments ({seg_idx + 1}/{child_segments})'
            }))

        # =============================
        # Step 14.7: Progressive shrinking
        # =============================
        for seg_idx in range(child_segments):
            child_analyses_frame = []
            for child in child_analyses:
                current_heights = []
                for seg_j in range(len(child['dmax_list'])):
                    has_data = len(child['segment_points'][seg_j]) > 0
                    if seg_j < seg_idx:
                        current_height = child['dmax_list'][seg_j] if has_data else 0
                    elif seg_j == seg_idx:
                        current_height = child['dmax_list'][seg_j] if has_data else 0
                    else:
                        current_height = child['rmax']
                    current_heights.append(current_height)

                child_frame = child.copy()
                child_frame['current_heights'] = current_heights
                child_frame['current_segment_idx'] = seg_idx
                child_analyses_frame.append(child_frame)

            frames.append(StepFrame('parallel_child_shrinking', {
                'all_points': points.copy(),
                'parent_centroid': centroid.copy(),
                'child_analyses': child_analyses_frame,
                'parent_r': r.copy(),
                'parent_theta': theta.copy()
            }, meta={
                'step': 14.7,
                'frame': seg_idx + 1,
                'total_frames': child_segments,
                'description': f'Child - Reduce Empty Space ({seg_idx + 1}/{child_segments})'
            }))

        # =============================
        # Step 14.8: Finding the Large Gradients (CHANGED: added "Large")
        # =============================
        frames.append(StepFrame('parallel_child_d_values', {
            'all_points': points.copy(),
            'parent_centroid': centroid.copy(),
            'child_analyses': child_analyses,
            'parent_r': r.copy(),
            'parent_theta': theta.copy()
        }, meta={'step': 14.8, 'description': 'Child - Finding the Large Gradients'}))

        # =============================
        # Step 14.9: Gradient comparison (skip empty vs empty)
        # =============================
        # Build comparison lists for each child
        all_child_comparisons = []
        max_comparisons = 0

        # Use parent's max_r for threshold calculation (same as gradient detection)
        threshold_global = gradient_threshold_ratio * max_r

        for child in child_analyses:
            dmax_list_child = child['dmax_list']
            segment_points_child = child['segment_points']

            comparisons = []
            for i in range(len(dmax_list_child)):
                j = (i + 1) % len(dmax_list_child)

                is_empty_i = len(segment_points_child[i]) == 0
                is_empty_j = len(segment_points_child[j]) == 0

                # Skip empty vs empty
                if is_empty_i and is_empty_j:
                    continue

                val_i = dmax_list_child[i] if dmax_list_child[i] > 0 else 0
                val_j = dmax_list_child[j] if dmax_list_child[j] > 0 else 0

                # Use same logic as detect_gradient_boundaries: only check value difference
                diff = abs(val_i - val_j)
                has_gradient = diff > threshold_global

                comparisons.append((i, j, has_gradient, val_i, val_j, threshold_global))

            all_child_comparisons.append(comparisons)
            max_comparisons = max(max_comparisons, len(comparisons))

        # Generate synchronized comparison frames
        if max_comparisons > 0:
            for frame_idx in range(max_comparisons):
                child_analyses_frame = []

                for child_idx, child in enumerate(child_analyses):
                    comparisons = all_child_comparisons[child_idx]
                    child_frame = child.copy()

                    # Track found gradients up to this point
                    found_so_far = [(c[0], c[1]) for c in comparisons[:frame_idx+1] if c[2]]
                    child_frame['found_gradients_so_far'] = found_so_far

                    if frame_idx < len(comparisons):
                        seg_i, seg_j, has_gradient, val_i, val_j, threshold_child = comparisons[frame_idx]
                        child_frame['current_comparison'] = {
                            'seg_i': seg_i,
                            'seg_j': seg_j,
                            'has_gradient': has_gradient,
                            'val_i': val_i,
                            'val_j': val_j,
                            'threshold': threshold_child
                        }
                        child_frame['comparison_complete'] = False
                    else:
                        child_frame['current_comparison'] = None
                        child_frame['comparison_complete'] = True

                    child_frame['gradient_count'] = len(found_so_far)
                    child_frame['total_gradients'] = sum(1 for c in comparisons if c[2])
                    child_analyses_frame.append(child_frame)

                frames.append(StepFrame('parallel_child_gradient_comparison', {
                    'all_points': points.copy(),
                    'parent_centroid': centroid.copy(),
                    'child_analyses': child_analyses_frame,
                    'parent_r': r.copy(),
                    'parent_theta': theta.copy()
                }, meta={
                    'step': 14.9,
                    'comparison': frame_idx + 1,
                    'total_comparisons': max_comparisons,
                    'description': f'Child - Gradient Comparison ({frame_idx + 1}/{max_comparisons})'
                }))

        # =============================
        # Use a running counter so step numbers stay sequential
        # =============================
        child_step = 10  # starts at 14.10

        # Step 14.10: Final gradient lines (G-lines)
        frames.append(StepFrame('parallel_child_gradient_lines', {
            'all_points': points.copy(),
            'parent_centroid': centroid.copy(),
            'child_analyses': child_analyses,
            'parent_r': r.copy(),
            'parent_theta': theta.copy()
        }, meta={'step': f'14.{child_step}', 'description': 'Child - Gradient Lines (G-lines)'}))
        child_step += 1

        # Step: Search Gradient Pairs for each child (like parent Step 11)
        # Find the max number of gradient pairs across all children
        max_pairs = 0
        for child in child_analyses:
            ga = child.get('gradient_angles', [])
            if len(ga) >= 2:
                max_pairs = max(max_pairs, len(ga))

        if max_pairs > 0:
            for pair_idx in range(max_pairs):
                frames.append(StepFrame('parallel_child_gradient_pair_search', {
                    'all_points': points.copy(),
                    'parent_centroid': centroid.copy(),
                    'child_analyses': child_analyses,
                    'parent_r': r.copy(),
                    'parent_theta': theta.copy(),
                    'current_pair_idx': pair_idx,
                }, meta={
                    'step': f'14.{child_step}',
                    'pair': pair_idx + 1,
                    'total_pairs': max_pairs,
                    'description': f'Child - Search Gradient Pairs ({pair_idx + 1}/{max_pairs})'
                }))
            child_step += 1

        # Step: Compute New Centers for each child (like parent Step 12)
        max_centers = 0
        for child in child_analyses:
            ac = child.get('additional_centers', [])
            if ac:
                max_centers = max(max_centers, len(ac))

        if max_centers > 0:
            for center_idx in range(max_centers):
                frames.append(StepFrame('parallel_child_dynamic_center_found', {
                    'all_points': points.copy(),
                    'parent_centroid': centroid.copy(),
                    'child_analyses': child_analyses,
                    'parent_r': r.copy(),
                    'parent_theta': theta.copy(),
                    'centers_shown': center_idx + 1,
                }, meta={
                    'step': f'14.{child_step}',
                    'center_count': center_idx + 1,
                    'total_centers': max_centers,
                    'description': f'Child - Compute New Centers ({center_idx + 1}/{max_centers})'
                }))
            child_step += 1

        # Step: Algorithm Pass Complete for children (like parent Step 13)
        # Always show - displays sub-regions if found, or children as regions
        frames.append(StepFrame('parallel_child_find_matching_pairs', {
            'all_points': points.copy(),
            'parent_centroid': centroid.copy(),
            'child_analyses': child_analyses,
            'parent_r': np.append(r.copy(), recursive_display_rmax),
            'parent_theta': theta.copy()
        }, meta={'step': f'14.{child_step}', 'description': 'Child - Algorithm 1 Pass Complete'}))
        child_step += 1

        # Step: Final result - same as above but clean view
        frames.append(StepFrame('parallel_child_final', {
            'all_points': points.copy(),
            'parent_centroid': centroid.copy(),
            'child_analyses': child_analyses,
            'parent_r': np.append(r.copy(), recursive_display_rmax),
            'parent_theta': theta.copy()
        }, meta={'step': f'14.{child_step}', 'description': 'Child - Final Analysis Complete'}))

        for frame in frames:
            frame.payload['display_rmax'] = float(recursive_display_rmax)

    grandchild_analyses = [
        grandchild
        for child in child_analyses
        for grandchild in child.get('grandchild_analyses', [])
    ]
    if enable_recursion and max_recursion_depth >= 2 and grandchild_analyses:
        print(f"🔄 Second recursion: {len(grandchild_analyses)} grandchildren")
        frames.extend(_build_parallel_level_frames(
            points=points,
            parent_centroid=centroid,
            parent_r=r,
            parent_theta=theta,
            analyses=grandchild_analyses,
            level_segments=max(segments // 4, 2),
            gradient_threshold_ratio=gradient_threshold_ratio,
            global_max_r=max_r,
            step_prefix=15,
            level_key='grandchild',
            level_label='Grandchild',
            context_analyses=child_analyses,
            display_rmax=recursive_display_rmax,
            context_segments=segments
        ))

    print(f"🎬 Generated {len(frames)} frames")

    # Print frame statistics
    frame_types = {}
    for frame in frames:
        frame_types[frame.name] = frame_types.get(frame.name, 0) + 1

    print(f"Frame type breakdown:")
    for frame_type, count in sorted(frame_types.items()):
        print(f"  {frame_type}: {count} frames")

    # =============================
    # Hidden: Export-quality silhouettes frame
    # =============================
    if export_final_silhouettes and child_analyses:
        export_analyses = grandchild_analyses or child_analyses
        export_payload = {
            'all_points': points.copy(),
            'parent_centroid': centroid.copy(),
            'child_analyses': export_analyses,
            'parent_r': r.copy(),
            'parent_theta': theta.copy(),
            'rmax': max_r,
            'segments': segments,
            'display_rmax': recursive_display_rmax,
        }
        if grandchild_analyses:
            export_payload.update({
                'context_child_analyses': child_analyses,
                'context_segments': segments,
                'level_label': 'Grandchild',
            })
        frames.append(StepFrame(
            'export_silhouettes_only',
            export_payload,
            meta={'step': 'Export', 'description': 'Silhouettes Only (Export Quality)'}
        ))

    return frames
