import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class AnalysisResult:
    """Store analysis results"""
    points: np.ndarray
    centroid: np.ndarray
    r: np.ndarray
    theta: np.ndarray
    segment_points: List
    theta_edges: np.ndarray
    dmax_list: List[float]
    dmin_list: List[float]
    gradient_boundaries: List
    additional_centers: List
    gradient_angles: List
    center_labels: List
    sub_analyses: List = None
