"""
Plot Module - Desktop Version (Matplotlib only, no Streamlit)
Static analysis plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
from typing import List, Dict, Any, Optional, Callable, Tuple

from vmeans.segment import segment_points_by_x
from vmeans.data import convert_to_polar


# Unified figure size for consistency
FIGURE_SIZE = (5, 5)
FIGURE_SIZE_WIDE = (8, 4)  # For dot plot only


def draw_recursive_overlay(analysis_result: Dict, source_label: str = "") -> Figure:
    """
    Draw recursive overlay plot
    Overlay child analysis results on parent plot
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=FIGURE_SIZE)
    ax.set_title(f"Recursive Overlay: {source_label}", pad=20, fontsize=12, fontweight='bold')

    # 1. Parent information
    parent_points = analysis_result['points']
    parent_center = analysis_result['centroid']
    r_all, theta_all = convert_to_polar(parent_points, parent_center)

    # 2. Plot all points as gray background
    ax.scatter(theta_all, r_all, c='gainsboro', s=10, alpha=0.6, zorder=1, label='_nolegend_')

    # 3. Draw parent center point
    ax.scatter(0, 0, marker='*', color='black', s=200, edgecolor='white',
               label='Parent Center (K=1)', zorder=20)

    # 4. Iterate through sub-analysis results and overlay plot
    sub_analyses = analysis_result.get('sub_analyses', [])
    if len(sub_analyses) > 0:
        colors = plt.cm.get_cmap('Set2', len(sub_analyses))

        for i, sub_result in enumerate(sub_analyses):
            color = colors(i)
            label_prefix = f"Region {sub_result.get('label', f'C{i + 1}')}"

            # Child region information
            child_points = sub_result['points']
            if len(child_points) == 0:
                continue

            child_center = sub_result['centroid']
            child_segments = sub_result['segment_points']
            child_theta_edges = sub_result['theta_edges']
            child_gradients = sub_result.get('gradient_boundaries', [])

            # Draw child region center points on main plot
            r_offset, theta_offset = convert_to_polar(np.array([child_center]), parent_center)
            ax.scatter(theta_offset, r_offset, marker='X', color=color, s=150, edgecolor='black',
                       label=f"{label_prefix} Center (K=2)", zorder=21)

            # Use polygons to draw child region segments
            for j in range(len(child_segments)):
                if not child_segments[j]:
                    continue

                r_seg = max(child_segments[j])
                theta_start_local = child_theta_edges[j]
                theta_end_local = child_theta_edges[j + 1]

                # Use multiple points to define smooth arc edges
                arc_angles = np.linspace(theta_start_local, theta_end_local, 15)
                arc_x_local = r_seg * np.cos(arc_angles)
                arc_y_local = r_seg * np.sin(arc_angles)

                # Create polygon vertices (in child region's local Cartesian coordinate system)
                poly_verts_local_cartesian = np.vstack([
                    np.array([[0, 0]]),  # Start from child region center
                    np.column_stack([arc_x_local, arc_y_local]),
                    np.array([[0, 0]])  # Return to child region center to close
                ])

                # Translate vertices to global Cartesian coordinate system
                poly_verts_global_cartesian = poly_verts_local_cartesian + child_center

                # Convert global Cartesian coordinates to parent plot's polar coordinates
                r_poly, theta_poly = convert_to_polar(poly_verts_global_cartesian, parent_center)

                # Draw filled polygon
                ax.fill(theta_poly, r_poly, alpha=0.4, color=color, zorder=10)
                # Draw polygon outline
                ax.plot(theta_poly, r_poly, color=color, linewidth=1.0, zorder=11)

            # Draw child region gradient lines
            child_max_r = np.max([max(seg) if seg else 0 for seg in child_segments])
            if child_gradients and child_max_r > 0:
                for k, l in child_gradients:
                    if abs(k - l) == 1:
                        boundary_theta_local = child_theta_edges[max(k, l)]

                        # Define gradient line start and end points
                        start_point_global = child_center
                        end_point_global = np.array([
                            child_center[0] + child_max_r * np.cos(boundary_theta_local),
                            child_center[1] + child_max_r * np.sin(boundary_theta_local)
                        ])

                        line_points_global = np.array([start_point_global, end_point_global])

                        # Convert line endpoints to parent plot's polar coordinates
                        r_line, theta_line = convert_to_polar(line_points_global, parent_center)
                        ax.plot(theta_line, r_line, color=color, linestyle='--', linewidth=2.5, zorder=15)

    # Configure polar axes
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.grid(True, alpha=0.3)

    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def draw_segment_structure_with_dynamic_centers(
        r: np.ndarray, theta: np.ndarray,
        segment_points: List, theta_edges: np.ndarray,
        max_r: float, centroid: np.ndarray,
        points: np.ndarray = None,
        point_dmax: np.ndarray = None, point_dmin: np.ndarray = None,
        theta_dmax: float = None, theta_dmin: float = None,
        r_dmax: float = None, r_dmin: float = None,
        title: str = "Polar with Dynamic Centers",
        show_segments: bool = True,
        gradient_boundaries: List = None,
        detect_centers_func: Callable = None) -> Figure:
    """
    Polar coordinate segmented structure plot with dynamic center points
    """
    s = len(segment_points)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=FIGURE_SIZE)
    ax.set_title(title, pad=20, fontsize=12, fontweight='bold')

    # Draw segmented bar chart
    if show_segments:
        for i in range(s):
            r_i = max(segment_points[i]) if segment_points[i] else 0.0
            color = 'lightblue' if r_i > 0 else 'whitesmoke'
            ax.bar(
                x=(theta_edges[i] + theta_edges[i + 1]) / 2,
                height=r_i,
                width=theta_edges[1] - theta_edges[0],
                bottom=0.0,
                color=color,
                edgecolor='navy',
                linewidth=0.3,
                alpha=0.7
            )

    # Draw original data points
    ax.scatter(theta, r, c='dimgray', s=6, alpha=0.4)

    # Draw original center point and extreme points
    if theta_dmax is not None and r_dmax is not None:
        ax.scatter(theta_dmax, r_dmax, c='darkorange', s=80, edgecolor='black',
                   zorder=10, label=f'd_max ({r_dmax:.2f})', marker='o', linewidth=2)

    if theta_dmin is not None and r_dmin is not None:
        ax.scatter(theta_dmin, r_dmin, c='darkblue', s=80, edgecolor='white',
                   zorder=10, label=f'd_min ({r_dmin:.2f})', marker='o', linewidth=2)

    # Apply dynamic center detection algorithm
    if gradient_boundaries and points is not None and detect_centers_func is not None:
        additional_centers, gradient_angles, center_labels = detect_centers_func(
            points, r, theta, gradient_boundaries, theta_edges, segment_points
        )

        # Draw gradient boundary lines and annotations
        for idx, angle in enumerate(gradient_angles):
            ax.plot([angle, angle], [0, max_r],
                    color='magenta', linewidth=2, linestyle='--', alpha=0.8,
                    zorder=8)
            # Annotate gradient number
            ax.text(angle, max_r * 1.05, f'G{idx + 1}',
                    rotation=np.degrees(angle) - 90, ha='center', va='bottom',
                    fontsize=9, color='magenta', fontweight='bold')

        # Draw additional center points
        colors = ['cyan', 'lime', 'yellow', 'orange', 'pink']
        for idx, (center_x, center_y) in enumerate(additional_centers):
            # Calculate center point's polar coordinates relative to original centroid
            center_offset = np.array([center_x, center_y]) - centroid
            center_r = np.linalg.norm(center_offset)
            center_theta = np.arctan2(center_offset[1], center_offset[0])
            if center_theta < 0:
                center_theta += 2 * np.pi

            # Draw center point
            label = center_labels[idx] if idx < len(center_labels) else f'C{idx + 1}'
            color = colors[idx % len(colors)]
            ax.scatter(center_theta, center_r, marker='X', color=color, s=150,
                       edgecolor='black', linewidth=2, zorder=15, label=f'{label}')

    # Draw original center
    ax.scatter(0, 0, marker='*', color='red', s=200, edgecolor='black',
               label='Center', zorder=20)

    # Configure polar axes
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, max_r * 1.15)
    ax.grid(True, alpha=0.3)

    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def draw_dmin_dmax_edges(theta_edges: np.ndarray, dmin_list: List[float],
                         dmax_list: List[float], r: np.ndarray = None,
                         theta: np.ndarray = None,
                         point_dmax: np.ndarray = None, point_dmin: np.ndarray = None,
                         theta_dmax: float = None, theta_dmin: float = None,
                         r_dmax: float = None, r_dmin: float = None,
                         title: str = "Segment Boundaries",
                         gradient_boundaries: List = None,
                         centroid: np.ndarray = None) -> Figure:
    """
    Polar coordinate boundary plot
    """
    s = len(dmin_list)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=FIGURE_SIZE)
    ax.set_title(title, pad=20, fontsize=12, fontweight='bold')

    for i in range(s):
        angle_mid = (theta_edges[i] + theta_edges[i + 1]) / 2
        width = theta_edges[1] - theta_edges[0]
        dmin = dmin_list[i]
        dmax = dmax_list[i]

        # d_max boundary
        if dmax > 0:
            ax.bar(x=angle_mid, height=dmax, width=width, bottom=0.0,
                   color='none', edgecolor='steelblue', linestyle='-', linewidth=2,
                   label='Max boundary' if i == 0 else "")

        # d_min boundary
        if dmin > 0:
            ax.bar(x=angle_mid, height=dmin, width=width, bottom=0.0,
                   color='none', edgecolor='crimson', linestyle=':', linewidth=2,
                   label='Min boundary' if i == 0 else "")

    # Original data points
    if r is not None and theta is not None:
        ax.scatter(theta, r, c='lightgray', s=4, alpha=0.6)

    # Special points
    if theta_dmax is not None and r_dmax is not None:
        ax.scatter(theta_dmax, r_dmax, c='darkorange', s=100, edgecolor='black',
                   zorder=10, label=f'Global Max ({r_dmax:.2f})', marker='*', linewidth=2)

    if theta_dmin is not None and r_dmin is not None:
        ax.scatter(theta_dmin, r_dmin, c='darkblue', s=100, edgecolor='white',
                   zorder=10, label=f'Global Min ({r_dmin:.2f})', marker='*', linewidth=2)

    # Gradient boundary lines
    if gradient_boundaries:
        max_r = max(dmax_list) if dmax_list else 1
        for i, j in gradient_boundaries:
            if abs(i - j) == 1:
                boundary_theta = theta_edges[max(i, j)]
            else:
                continue
            ax.plot([boundary_theta, boundary_theta], [0, max_r],
                    color='magenta', linewidth=2, linestyle='--', alpha=0.8,
                    zorder=8)

    # Centroid
    ax.scatter(0, 0, marker='*', color='gold', s=120, edgecolor='black',
               zorder=15, linewidth=1.5)

    # Configure polar axes
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.grid(True, alpha=0.3)

    if dmax_list:
        max_radius = max(dmax_list) * 1.1 if max(dmax_list) > 0 else 1
        ax.set_ylim(0, max_radius)

    ax.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center',
             ncol=2, frameon=True, fancybox=True, shadow=True, fontsize=9)
    fig.tight_layout()
    return fig


def draw_cartesian_x_based_analysis(points: np.ndarray, segments: int = 60,
                                     gradient_threshold_ratio: float = None,
                                     title: str = "Cartesian X-based Analysis") -> Figure:
    """
    Cartesian X-based analysis plot
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_title(title, pad=15, fontsize=12, fontweight='bold')

    # X-based segmentation
    segment_ranges, segment_y_mins, segment_y_maxs, x_edges, segment_points_indices = segment_points_by_x(points, s=segments)

    # Calculate centroid
    centroid_y = np.mean(points[:, 1])

    # Draw Y ranges for each X segment
    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(segments)]
    x_width = x_edges[1] - x_edges[0]

    for i in range(segments):
        y_min = segment_y_mins[i]
        y_max = segment_y_maxs[i]
        has_points = len(segment_points_indices[i]) > 0

        if has_points:
            if y_max >= centroid_y and y_min >= centroid_y:
                bottom_pos = centroid_y
                bar_height = y_max - centroid_y
                color = 'lightblue'
            elif y_max <= centroid_y and y_min <= centroid_y:
                bottom_pos = y_min
                bar_height = centroid_y - y_min
                color = 'lightcoral'
            else:
                bottom_pos = y_min
                bar_height = y_max - y_min
                color = 'lightyellow'

            ax.bar(x_centers[i], bar_height, width=x_width * 0.8,
                   bottom=bottom_pos, alpha=0.7, color=color,
                   edgecolor='black', linewidth=0.5)

    # Draw data points
    ax.scatter(points[:, 0], points[:, 1], c='dimgray', s=10, alpha=0.5, label='Data Points')

    # Draw Y centroid line
    ax.axhline(y=centroid_y, color='green', linewidth=2, alpha=0.7,
               label=f'Y Centroid ({centroid_y:.2f})', linestyle='-')

    # Gradient detection
    if gradient_threshold_ratio is not None:
        # Calculate effective lengths
        effective_lengths = []
        is_empty_segment = []

        for i in range(segments):
            y_min = segment_y_mins[i]
            y_max = segment_y_maxs[i]
            has_points = len(segment_points_indices[i]) > 0
            is_empty_segment.append(not has_points)

            if has_points:
                distance_to_center_max = abs(y_max - centroid_y)
                distance_to_center_min = abs(y_min - centroid_y)
                max_distance_to_center = max(distance_to_center_max, distance_to_center_min)
                effective_length = max(segment_ranges[i], max_distance_to_center)
                effective_lengths.append(effective_length)
            else:
                effective_lengths.append(0.0)

        # Find gradient boundaries
        global_y_range = points[:, 1].max() - points[:, 1].min()
        threshold_absolute = global_y_range * gradient_threshold_ratio

        gradient_boundaries = []
        for i in range(len(effective_lengths) - 1):
            current_empty = is_empty_segment[i]
            next_empty = is_empty_segment[i + 1]
            current_length = effective_lengths[i]
            next_length = effective_lengths[i + 1]

            if current_empty != next_empty:
                gradient_boundaries.append((i, i + 1))
            elif not current_empty and not next_empty:
                diff = abs(current_length - next_length)
                if diff > threshold_absolute:
                    gradient_boundaries.append((i, i + 1))

        # Draw gradient lines
        for i, j in gradient_boundaries:
            boundary_x = x_edges[j]
            ax.axvline(x=boundary_x, color='purple', linewidth=2, linestyle='--', alpha=0.8)

    # Mark centroid
    centroid_x = np.mean(points[:, 0])
    ax.scatter(centroid_x, centroid_y, c='green', s=100, edgecolor='black',
               label=f'Centroid', zorder=10, marker='*', linewidth=2)

    ax.set_xlabel('X coordinate', fontsize=10)
    ax.set_ylabel('Y coordinate', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=8)

    fig.tight_layout()
    return fig


def draw_cartesian_x_distance_scatter(points: np.ndarray,
                                       title: str = "Distance vs X Scatter") -> Figure:
    """
    Cartesian X-based distance scatter plot
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_title(title, pad=15, fontsize=12, fontweight='bold')

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # Find extreme points
    y_max = np.max(y_coords)
    y_min = np.min(y_coords)
    highest_idx = np.argmax(y_coords)
    lowest_idx = np.argmin(y_coords)

    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    x_range = x_max - x_min if x_max > x_min else 1

    # Normalize X to 0-360 degrees
    x_normalized = (x_coords - x_min) / x_range * 360

    # Draw scatter plot
    ax.scatter(x_normalized, y_coords, c='black', s=10, alpha=0.6, label='Data Points')

    # Annotate special points
    ax.scatter(x_normalized[highest_idx], y_coords[highest_idx], c='red', s=60, edgecolor='black',
               label=f'Highest (y={y_coords[highest_idx]:.1f})', zorder=5)
    ax.annotate(f'Highest\n({x_coords[highest_idx]:.1f}, {y_coords[highest_idx]:.1f})',
                (x_normalized[highest_idx], y_coords[highest_idx]),
                xytext=(10, 10), textcoords='offset points', fontsize=8, color='red')

    if lowest_idx != highest_idx:
        ax.scatter(x_normalized[lowest_idx], y_coords[lowest_idx], c='blue', s=60, edgecolor='black',
                   label=f'Lowest (y={y_coords[lowest_idx]:.1f})', zorder=5)
        ax.annotate(f'Lowest\n({x_coords[lowest_idx]:.1f}, {y_coords[lowest_idx]:.1f})',
                    (x_normalized[lowest_idx], y_coords[lowest_idx]),
                    xytext=(10, -15), textcoords='offset points', fontsize=8, color='blue')

    # Set horizontal axis
    ax.set_xlim(0, 360)
    ax.set_xlabel(f"X mapped to degrees", fontsize=10)
    ax.set_ylabel("Y coordinate", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=8)

    # Add reference lines
    ax.axhline(y=y_max, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=y_min, color='blue', linestyle='--', alpha=0.5, linewidth=1)

    # Add X position reference lines
    for angle_ref in [90, 180, 270]:
        ax.axvline(x=angle_ref, color='gray', linestyle=':', alpha=0.3)

    fig.tight_layout()
    return fig


def draw_segment_distance_dot_plot(segment_points: List,
                                    theta_edges: np.ndarray) -> Optional[Tuple]:
    """
    Segment-level distance scatter plot
    Returns: (fig, d_max_val, d_min_val, empty_count) or None
    """
    s = len(segment_points)
    d_values = np.array([max(seg) if seg else 0.0 for seg in segment_points])
    angle_centers = np.array([(theta_edges[i] + theta_edges[i + 1]) / 2 for i in range(s)])
    angle_centers_deg = np.mod(np.degrees(angle_centers), 360)

    if np.all(d_values == 0):
        return None

    pivot_idx = np.argmax(d_values)
    pivot_angle = angle_centers_deg[pivot_idx]
    rotated_angles = np.mod(angle_centers_deg - pivot_angle, 360)

    # Calculate global maximum and minimum
    d_max_val = np.max(d_values)
    d_min_val = np.min(d_values)

    # Color coding
    colors = []
    for d in d_values:
        if np.isclose(d, d_max_val):
            colors.append("red")
        elif np.isclose(d, d_min_val):
            colors.append("blue")
        else:
            colors.append("gray")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    ax.scatter(rotated_angles, d_values, c=colors, s=20)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel("Angle (degrees, 0° = segment with max d)", fontsize=10)
    ax.set_ylabel("d (distance)", fontsize=10)
    ax.set_title("Segment-wise Distance Scatter Plot", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate d_max points
    max_indices = np.where(np.isclose(d_values, d_max_val))[0]
    for idx in max_indices:
        ax.scatter(rotated_angles[idx], d_values[idx], c='red', s=60,
                   edgecolor='black', marker='*', zorder=5)

    # Annotate all d_min points
    min_indices = np.where(np.isclose(d_values, d_min_val))[0]
    for idx in min_indices:
        ax.scatter(rotated_angles[idx], d_values[idx], c='blue', s=60,
                   edgecolor='black', marker='*', zorder=5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='Segments'),
        Patch(facecolor='red', label=f'd_max = {d_max_val:.2f}'),
        Patch(facecolor='blue', label=f'd_min = {d_min_val:.2f}')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.tight_layout()

    empty_count = len(min_indices) if d_min_val == 0 else 0
    return fig, d_max_val, d_min_val, empty_count