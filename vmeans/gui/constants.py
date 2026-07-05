from typing import Dict



# ============================================================================
# CONSTANTS
# ============================================================================

FIGURE_DPI = 150  # Higher DPI for sharper rendering
CANVAS_MIN_SIZE = 450
ANIMATION_FIGURE_SIZE = (7, 7)
DEFAULT_GENERATED_POINTS = 1000
DEFAULT_POINTS_BY_DATASET: Dict[str, int] = {
    "blobs": 1000,
    "quadrants": 1000,
    "varied_blobs": 1000,
    "flower": 1000,
    "spiral": 1000,
    "cross": 1000,
    "ring": 1000,
    "concentric_circles": 1000,
    "anisotropic_blobs": 1000,
    "moons": 1000,
    "aggregation": 788,
    "zahn_compound": 399,
}


def _default_points_for_dataset(structure: str) -> int:
    """Return the internal point-count setting for generated datasets."""
    return DEFAULT_POINTS_BY_DATASET.get(structure, DEFAULT_GENERATED_POINTS)
