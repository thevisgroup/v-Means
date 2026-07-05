"""
Core Analysis Module - Desktop Version (No Streamlit)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable

from vmeans.data import convert_to_polar
from vmeans.segment import segment_points_by_theta


def compute_better_center(points: np.ndarray, method: str = 'auto') -> np.ndarray:
    """Calculate center point - unified function"""
    if method == 'geometric':
        x_center = (points[:, 0].min() + points[:, 0].max()) / 2
        y_center = (points[:, 1].min() + points[:, 1].max()) / 2
        return np.array([x_center, y_center])
    return np.mean(points, axis=0)


def detect_gradient_boundaries(lengths: List[float], threshold_ratio: float = 0.25, 
                                global_max: float = None) -> List[Tuple[int, int]]:
    """Gradient boundary detection"""
    boundaries = []
    n = len(lengths)
    if global_max is None:
        global_max = max(lengths) if lengths else 1.0
    absolute_threshold = threshold_ratio * global_max

    # Gradient detection (compact output)

    for i in range(n):
        current, next_length = lengths[i], lengths[(i + 1) % n]
        if abs(current - next_length) > absolute_threshold:
            boundaries.append((i, (i + 1) % n))

    print(f"Gradients: {len(boundaries)} boundaries (threshold={absolute_threshold:.3f})")
    return list(dict.fromkeys(boundaries))


def detect_dynamic_centers(points: np.ndarray, r: np.ndarray, theta: np.ndarray,
                           gradient_boundaries: List[Tuple[int, int]],
                           theta_edges: np.ndarray,
                           segment_points: List) -> Tuple[List, List, List]:
    """
    Dynamic center point detection algorithm
    """




    # 1. Extract gradient boundary angles and sort
    gradient_angles = []
    for i, j in gradient_boundaries:
        if abs(i - j) == 1 or abs(i - j) == len(theta_edges) - 2:  # Handle wrap-around
            if abs(i - j) == 1:
                boundary_angle = theta_edges[max(i, j)]
            else:  # wrap-around case
                boundary_angle = theta_edges[len(theta_edges) - 1]  # = 2π
            gradient_angles.append(boundary_angle)

    gradient_angles = sorted(list(set(gradient_angles)))
    print(f"G-angles: {[f'{np.degrees(a):.0f}°' for a in gradient_angles]}")

    if len(gradient_angles) < 2:
        print("Insufficient gradient boundaries, cannot perform pairing")
        return [], gradient_angles, []

    # 2. Check if there are data points between each pair of adjacent gradients
    additional_centers = []
    center_labels = []

    for i in range(len(gradient_angles)):
        next_i = (i + 1) % len(gradient_angles)
        g_current = gradient_angles[i]
        g_next = gradient_angles[next_i]

        crosses_zero = g_current > g_next
        # (interval check)

        # 3. Collect points within current interval
        points_in_range_indices = []
        for j, angle in enumerate(theta):
            if crosses_zero:
                in_range = (angle >= g_current) or (angle <= g_next)
            else:
                in_range = g_current <= angle <= g_next
            if in_range:
                points_in_range_indices.append(j)


        # 4. Only create center if there are actually points in this region
        if len(points_in_range_indices) > 0:

            points_in_range = points[points_in_range_indices]
            center_x, center_y = np.mean(points_in_range, axis=0)
            additional_centers.append([center_x, center_y])
            center_labels.append(f"C{len(additional_centers)}")

            # For logging
            center_r, center_theta = convert_to_polar(np.array([[center_x, center_y]]), compute_better_center(points))

    print(f"Centers: {len(additional_centers)} found {center_labels}")
    return additional_centers, gradient_angles, center_labels


def segment_points_by_region(points: np.ndarray, theta: np.ndarray,
                              gradient_angles: List[float],
                              additional_centers: List) -> List[List[int]]:
    """
    Assign points to different regions based on gradient boundaries
    """




    if len(gradient_angles) < 2 or len(additional_centers) == 0:
        print("Cannot create regions: insufficient gradients or centers")
        return []

    n_regions = len(additional_centers)
    regions = [[] for _ in range(n_regions)]
    center_idx = 0



    for i in range(len(gradient_angles)):
        if center_idx >= n_regions:
                break

        next_i = (i + 1) % len(gradient_angles)
        g_current, g_next = gradient_angles[i], gradient_angles[next_i]
        crosses_zero = g_current > g_next


        # Check if this interval has points
        indices_in_range = []
        for idx, angle in enumerate(theta):
            if (crosses_zero and (angle >= g_current or angle <= g_next)) or \
                    (not crosses_zero and g_current <= angle <= g_next):
                indices_in_range.append(idx)


        if len(indices_in_range) > 0:
            regions[center_idx] = indices_in_range
            center_idx += 1

    region_summary = ", ".join(f"R{i}:{len(r)}" for i, r in enumerate(regions))
    print(f"Regions: {region_summary}")

    return regions


def recursive_polar_analysis(points: np.ndarray, source_label: str, segments: int,
                              plot_options: Any, center_method: str,
                              recursion_level: int = 0, max_recursion: int = 1,
                              global_max_r: float = None) -> Optional[Dict]:
    """
    Recursive polar analysis
    """
    indent = '  ' * recursion_level

    if recursion_level > max_recursion:
        return None

    if len(points) < 2:
        return None

    # Core analysis
    centroid = compute_better_center(points, method=center_method)
    r, theta = convert_to_polar(points, centroid)
    max_r = np.max(r) if len(r) > 0 else 1.0

    # First call: record top-level d_max as fixed reference
    if global_max_r is None:
        global_max_r = max_r

    segment_points_data, theta_edges = segment_points_by_theta(r, theta, s=segments)
    dmax_list = [max(seg) if seg else 0.0 for seg in segment_points_data]

    # Gradient detection - always use top-level d_max as reference
    gradient_boundaries = []
    if plot_options.show_gradient_lines:
        gradient_boundaries = detect_gradient_boundaries(
            dmax_list, plot_options.gradient_threshold_ratio, global_max=global_max_r
        )

    # Dynamic centers
    additional_centers, gradient_angles, center_labels = [], [], []
    regions = []

    if gradient_boundaries:
        additional_centers, gradient_angles, center_labels = detect_dynamic_centers(
            points, r, theta, gradient_boundaries, theta_edges, segment_points_data
        )
        if additional_centers:
            regions = segment_points_by_region(points, theta, gradient_angles, additional_centers)

    # Build result
    analysis_result = {
        'points': points,
        'centroid': centroid,
        'r': r,
        'theta': theta,
        'segment_points': segment_points_data,
        'theta_edges': theta_edges,
        'dmax_list': dmax_list,
        'd_max_val': max_r,
        'gradient_boundaries': gradient_boundaries,
        'additional_centers': additional_centers,
        'gradient_angles': gradient_angles,
        'center_labels': center_labels,
        'sub_analyses': []
    }

    # Recurse into sub-regions
    if recursion_level < max_recursion and additional_centers and regions:

        for i, center_label in enumerate(center_labels):
            if i < len(regions):
                region_points_count = len(regions[i])

                if region_points_count > 10:
                    try:
                        sub_points = points[list(set(regions[i]))]

                        sub_result = recursive_polar_analysis(
                            sub_points, f"{source_label} - {center_label}",
                            segments // 2, plot_options, 'centroid',
                            recursion_level + 1, max_recursion,
                            global_max_r=global_max_r
                        )

                        if sub_result:
                            sub_result['parent_center_label'] = center_label
                            analysis_result['sub_analyses'].append(sub_result)

                    except Exception as e:
                        print(f"{indent}  Sub-analysis {center_label} failed: {e}")

    return analysis_result