
import html
import time
import numpy as np
from typing import Dict, Any, Iterable, List

from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QWidget, QGroupBox, QComboBox, QPushButton, QTextEdit,
    QPlainTextEdit, QCheckBox, QToolTip, QRubberBand
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent, QRect, QSize

import pyqtgraph as pg

from scipy.spatial import cKDTree

from vmeans.ai_client import MODEL_PRESETS, ask_ai
from vmeans.colors import get_colors_for_centers

RECURSIVE_HOVER_FRAMES = {
    'parallel_child_final',
    'parallel_child_find_matching_pairs',
    'parallel_grandchild_final',
    'parallel_grandchild_find_matching_pairs',
    'export_silhouettes_only',
}


def _extract_region_data(frame_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract points, region assignments, colors, and centroid from the last frame.

    Handles three cases:
      1. Recursion: parallel_child_final / parallel_child_find_matching_pairs
      2. No recursion but regions found: find_matching_gradient_pairs
      3. Fallback (no regions): single cluster
    """

    # --- Case 1: Recursion ---
    if frame_name in RECURSIVE_HOVER_FRAMES:
        all_points = payload.get('all_points')
        if all_points is None:
            raise ValueError(
                f"Frame '{frame_name}' does not include point coordinates for hover details."
            )
        all_points = np.asarray(all_points, dtype=float)
        if all_points.ndim != 2 or all_points.shape[1] < 2 or len(all_points) == 0:
            raise ValueError(f"Frame '{frame_name}' has invalid point coordinates.")

        centroid = payload.get('parent_centroid')
        child_analyses = payload.get('child_analyses', [])

        n_points = len(all_points)
        region_ids = np.full(n_points, -1, dtype=int)

        centers = []
        point_counts = []
        region_labels = []
        current_region_id = 0

        # Robust global index mapping
        tree = cKDTree(all_points)

        for child in child_analyses:
            child_pts = child['points']
            child_theta = child.get('theta', None)
            gradient_angles_local = child.get('gradient_angles', [])
            additional_centers_local = child.get('additional_centers', [])
            regions_local = child.get('regions', [])

            if len(child_pts) == 0:
                continue

            _, child_global_indices = tree.query(child_pts)

            has_sub = (
                child_theta is not None
                and len(gradient_angles_local) >= 2
                and len(additional_centers_local) > 0
            )

            if has_sub:
                if not regions_local:
                    from vmeans.core_analysis import segment_points_by_region
                    regions_local = segment_points_by_region(
                        child_pts, child_theta, gradient_angles_local, additional_centers_local
                    )

                for sub_idx, sub_local_indices in enumerate(regions_local):
                    if len(sub_local_indices) == 0:
                        continue

                    global_indices = child_global_indices[sub_local_indices]
                    region_ids[global_indices] = current_region_id

                    sub_center = (
                        additional_centers_local[sub_idx]
                        if sub_idx < len(additional_centers_local)
                        else np.mean(child_pts[sub_local_indices], axis=0)
                    )
                    centers.append(sub_center)
                    point_counts.append(len(sub_local_indices))
                    region_labels.append(f"Region {current_region_id + 1}")
                    current_region_id += 1
            else:
                region_ids[child_global_indices] = current_region_id
                centers.append(child['center_global'])
                point_counts.append(len(child_pts))
                region_labels.append(f"Region {current_region_id + 1}")
                current_region_id += 1

        # Grandchild/export frames focus on recursed regions; keep unchanged
        # child regions visible and hoverable instead of leaving them unassigned.
        for context_child in payload.get('context_child_analyses', []):
            context_pts = context_child.get('points')
            if context_pts is None or len(context_pts) == 0:
                continue

            _, context_global_indices = tree.query(context_pts)
            context_global_indices = np.atleast_1d(context_global_indices)
            unassigned = context_global_indices[region_ids[context_global_indices] < 0]
            if len(unassigned) == 0:
                continue

            region_ids[unassigned] = current_region_id
            center = context_child.get('center_global')
            if center is None:
                center = np.mean(all_points[unassigned], axis=0)
            centers.append(center)
            point_counts.append(len(unassigned))
            region_labels.append(f"Region {current_region_id + 1}")
            current_region_id += 1

        remaining = np.where(region_ids < 0)[0]
        if len(remaining) > 0:
            region_ids[remaining] = current_region_id
            centers.append(np.mean(all_points[remaining], axis=0))
            point_counts.append(len(remaining))
            region_labels.append(f"Region {current_region_id + 1}")
            current_region_id += 1

        if centers:
            colors_hex = get_colors_for_centers(
                np.array(centers),
                origin=centroid,
                point_counts=point_counts,
                ensure_distinct=True
            )
        else:
            colors_hex = ['#888888']

        return {
            'points': all_points,
            'centroid': centroid,
            'region_ids': region_ids,
            'colors_hex': colors_hex,
            'region_labels': region_labels,
            'centers': centers,
            'n_regions': len(centers),
        }

    # --- Case 2: No recursion, regions exist ---
    if frame_name == 'find_matching_gradient_pairs':
        points = payload['points']
        centroid = payload['centroid']
        regions = payload['regions']
        additional_centers = payload['additional_centers']
        center_labels = payload.get('center_labels', [])

        n_points = len(points)
        region_ids = np.full(n_points, -1, dtype=int)

        point_counts = []
        for idx, indices in enumerate(regions):
            for i in indices:
                region_ids[i] = idx
            point_counts.append(len(indices))

        if not center_labels:
            center_labels = [f"C{i + 1}" for i in range(len(additional_centers))]

        if additional_centers:
            colors_hex = get_colors_for_centers(
                np.array(additional_centers),
                origin=centroid,
                point_counts=point_counts,
                ensure_distinct=True
            )
        else:
            colors_hex = ['#888888']

        return {
            'points': points,
            'centroid': centroid,
            'region_ids': region_ids,
            'colors_hex': colors_hex,
            'region_labels': center_labels,
            'centers': additional_centers,
            'n_regions': len(additional_centers),
        }

    # --- Case 3: Fallback - no regions ---
    points = payload.get('all_points', payload.get('points'))
    centroid = payload.get('parent_centroid', payload.get('centroid'))
    if centroid is None and points is not None:
        centroid = np.mean(points, axis=0)

    n_points = len(points) if points is not None else 0
    region_ids = np.zeros(n_points, dtype=int)

    return {
        'points': points,
        'centroid': centroid,
        'region_ids': region_ids,
        'colors_hex': ['#4a90d9'],
        'region_labels': ['All'],
        'centers': [],
        'n_regions': 1 if n_points > 0 else 0,
    }
