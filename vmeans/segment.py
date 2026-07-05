# segment.py - Corrected Version

import numpy as np


def segment_points_by_theta(r, theta, s=60):
    """Polar coordinate segmentation by angle"""
    theta_edges = np.linspace(0, 2 * np.pi, s + 1)
    segment_points = [[] for _ in range(s)]
    for i in range(len(theta)):
        seg_id = np.searchsorted(theta_edges, theta[i], side='right') - 1
        if seg_id >= s:
            seg_id = 0  # 2π wraps to 0
        segment_points[seg_id].append(r[i])
    return segment_points, theta_edges


def segment_points_by_x(points, s=60):
    """
    Cartesian coordinate segmentation by X coordinate
    Correction: Each segment's "height" is the Y value range within that segment (max_y - min_y)

    Args:
        points: Original data points (N, 2)
        s: Number of segments

    Returns:
        segment_ranges: List of Y value ranges (max_y - min_y) for each segment
        segment_y_mins: List of minimum Y values for each segment (for plotting positioning)
        segment_y_maxs: List of maximum Y values for each segment (for plotting positioning)
        x_edges: X coordinate segmentation boundaries
        segment_points_indices: List of point indices contained in each segment
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min = np.min(x_coords)
    x_max = np.max(x_coords)

    # Create X coordinate segmentation boundaries
    x_edges = np.linspace(x_min, x_max, s + 1)

    # Point indices and Y value ranges for each segment
    segment_points_indices = [[] for _ in range(s)]
    segment_ranges = []
    segment_y_mins = []
    segment_y_maxs = []

    for i in range(s):
        x_left = x_edges[i]
        x_right = x_edges[i + 1]

        # Find points within this X range
        if i == s - 1:  # Last segment includes right boundary
            mask = (x_coords >= x_left) & (x_coords <= x_right)
        else:
            mask = (x_coords >= x_left) & (x_coords < x_right)

        indices = np.where(mask)[0]
        segment_points_indices[i] = indices.tolist()

        if len(indices) > 0:
            # Calculate Y value range for this segment
            y_values_in_segment = y_coords[indices]
            min_y = np.min(y_values_in_segment)
            max_y = np.max(y_values_in_segment)
            y_range = max_y - min_y

            segment_ranges.append(y_range)
            segment_y_mins.append(min_y)
            segment_y_maxs.append(max_y)
        else:
            segment_ranges.append(0.0)
            segment_y_mins.append(0.0)
            segment_y_maxs.append(0.0)

    return segment_ranges, segment_y_mins, segment_y_maxs, x_edges, segment_points_indices