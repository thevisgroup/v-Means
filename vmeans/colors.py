"""
Colors Module - Paletton-style color system (v4 clean version)

Core concept (following Paletton exactly):
    Color wheel = 360°, each angle maps to a color

    Hue (outer ring) = polar angle of cluster center
    Saturation/Value (inner ring position) = number of points in cluster

    More points → more saturated, vibrant colors (outer ring)
    Fewer points → lighter, softer colors (inner ring)

This way even if two clusters have similar angles, different point counts make them distinguishable!
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass
import colorsys
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Saturation range (mapped from point count)
SAT_MIN = 0.35   # saturation when points are fewest (inner ring, pale)
SAT_MAX = 0.85   # saturation when points are most (outer ring, saturated)

# Value range
VAL_MIN = 0.55   # value when points are fewest
VAL_MAX = 0.90   # value when points are most

# Defaults (when point count is unknown)
DEFAULT_SATURATION = 0.70
DEFAULT_VALUE = 0.80

# Fallback mechanism: ensure_distinct mode parameters
MIN_HUE_GAP = 25          # minimum hue separation (degrees)
MIN_SV_DIFF = 0.15        # minimum S/V difference
HUE_NUDGE = 12            # nudge amount for hue adjustment (degrees)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ColorPalette:
    """Complete color scheme for a cluster"""
    primary: str      # main color
    light: str        # lighter (hover)
    lighter: str      # even lighter (background highlight)
    dark: str         # darker (border)
    darker: str       # even darker (shadow)
    background: str   # background fill
    muted: str        # disabled state
    hue: float        # hue angle

    def __repr__(self):
        return f"ColorPalette(hue={self.hue:.0f}°, primary={self.primary})"


# ============================================================================
# Color conversion
# ============================================================================

def hsv_to_hex(h: float, s: float, v: float) -> str:
    """HSV → Hex (h: 0-360, s: 0-1, v: 0-1)"""
    h = h % 360
    s = max(0, min(1, s))
    v = max(0, min(1, v))
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Hex → RGB (0-255)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# ============================================================================
# Core: angle + point count → color
# ============================================================================

def angle_to_color(angle_deg: float,
                   n_points: int = None,
                   max_points: int = None,
                   saturation: float = None,
                   value: float = None) -> str:
    """
    Get color based on angle and point count

    Paletton concept:
        Hue = angle_deg (direct mapping, no adjustments)
        Saturation/Value = based on point count (more points = more saturated)

    Args:
        angle_deg: polar angle (degrees), maps directly to hue
        n_points: point count for this cluster (optional)
        max_points: maximum point count (for normalization)
        saturation: manually specify saturation (overrides point count calculation)
        value: manually specify value (overrides point count calculation)

    Returns:
        hex color
    """
    hue = angle_deg % 360

    # Calculate saturation and value
    if saturation is not None and value is not None:
        # manually specified
        s, v = saturation, value
    elif n_points is not None and max_points is not None and max_points > 0:
        # calculate from point count
        ratio = min(n_points / max_points, 1.0)
        s = SAT_MIN + (SAT_MAX - SAT_MIN) * ratio
        v = VAL_MIN + (VAL_MAX - VAL_MIN) * ratio
    else:
        # use defaults
        s, v = DEFAULT_SATURATION, DEFAULT_VALUE

    return hsv_to_hex(hue, s, v)


def generate_palette_from_angle(angle_deg: float,
                                 n_points: int = None,
                                 max_points: int = None,
                                 saturation: float = None,
                                 value: float = None) -> ColorPalette:
    """
    Generate complete palette from angle
    """
    hue = angle_deg % 360

    # Calculate base S/V
    if saturation is not None and value is not None:
        s, v = saturation, value
    elif n_points is not None and max_points is not None and max_points > 0:
        ratio = min(n_points / max_points, 1.0)
        s = SAT_MIN + (SAT_MAX - SAT_MIN) * ratio
        v = VAL_MIN + (VAL_MAX - VAL_MIN) * ratio
    else:
        s, v = DEFAULT_SATURATION, DEFAULT_VALUE

    return ColorPalette(
        primary=hsv_to_hex(hue, s, v),
        light=hsv_to_hex(hue, s * 0.6, min(v + 0.12, 1.0)),
        lighter=hsv_to_hex(hue, s * 0.35, min(v + 0.18, 1.0)),
        dark=hsv_to_hex(hue, min(s * 1.2, 1.0), v * 0.65),
        darker=hsv_to_hex(hue, min(s * 1.3, 1.0), v * 0.45),
        background=hsv_to_hex(hue, s * 0.12, 0.97),
        muted=hsv_to_hex(hue, s * 0.4, v * 0.7),
        hue=hue
    )


# ============================================================================
# Fallback: ensure colors are distinguishable
# ============================================================================

def _ensure_colors_distinct(hues: List[float],
                            saturations: List[float],
                            values: List[float]) -> Tuple[List[float], List[float], List[float]]:
    """
    Fallback mechanism: nudge Hue when colors are too close in both Hue and S/V

    Only adjusts when necessary, tries to preserve original angle semantics
    """
    n = len(hues)
    if n <= 1:
        return hues, saturations, values

    hues = list(hues)
    saturations = list(saturations)
    values = list(values)

    # Check each pair
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate Hue distance
            hue_diff = abs(hues[i] - hues[j])
            hue_diff = min(hue_diff, 360 - hue_diff)  # account for wraparound

            # Calculate S/V difference
            sv_diff = abs(saturations[i] - saturations[j]) + abs(values[i] - values[j])

            # If Hue too close AND S/V too close → need adjustment
            if hue_diff < MIN_HUE_GAP and sv_diff < MIN_SV_DIFF:
                # Nudge one of the Hues
                # pick the one with larger index
                nudge_direction = 1 if (j % 2 == 0) else -1
                hues[j] = (hues[j] + HUE_NUDGE * nudge_direction) % 360

    return hues, saturations, values


# ============================================================================
# Main API
# ============================================================================

def get_cluster_colors(n: int,
                       point_counts: List[int] = None,
                       angles: List[float] = None,
                       ensure_distinct: bool = False) -> List[str]:
    """
    Get colors for n clusters

    Args:
        n: number of clusters
        point_counts: point count per cluster (optional, affects saturation)
        angles: angle per cluster (optional, evenly distributed if not provided)
        ensure_distinct: enable fallback mechanism (nudge when colors too close)

    Returns:
        n hex colors

    Example:
        # simple usage
        colors = get_cluster_colors(5)

        # with point counts (affects saturation)
        colors = get_cluster_colors(3, point_counts=[100, 50, 200])

        # with angles (specify hue)
        colors = get_cluster_colors(3, angles=[30, 120, 240])

        # enable fallback (ensure distinguishable)
        colors = get_cluster_colors(3, angles=[30, 35, 40], ensure_distinct=True)
    """
    if n <= 0:
        return []

    # Determine angles
    if angles is not None and len(angles) >= n:
        hues = [a % 360 for a in angles[:n]]
    else:
        # evenly distributed
        hues = [(i * 360 / n) % 360 for i in range(n)]

    # Determine point counts → S/V
    if point_counts is not None and len(point_counts) >= n:
        counts = point_counts[:n]
        max_count = max(counts) if counts else 1
        saturations = []
        values = []
        for c in counts:
            ratio = min(c / max_count, 1.0)
            saturations.append(SAT_MIN + (SAT_MAX - SAT_MIN) * ratio)
            values.append(VAL_MIN + (VAL_MAX - VAL_MIN) * ratio)
    else:
        saturations = [DEFAULT_SATURATION] * n
        values = [DEFAULT_VALUE] * n

    # Fallback mechanism
    if ensure_distinct:
        hues, saturations, values = _ensure_colors_distinct(hues, saturations, values)

    # Generate colors
    return [hsv_to_hex(h, s, v) for h, s, v in zip(hues, saturations, values)]


def get_colors_for_centers(centers: np.ndarray,
                           origin: np.ndarray = None,
                           point_counts: List[int] = None,
                           ensure_distinct: bool = False) -> List[str]:
    """
    Assign colors based on cluster center coordinates

    Core logic:
        1. Calculate polar angle of each center relative to origin → use as Hue
        2. If point counts provided → adjust saturation/value
        3. If ensure_distinct enabled → nudge when colors too close

    Args:
        centers: center coordinates array (n, 2)
        origin: origin point, defaults to (0, 0)
        point_counts: point count per cluster (optional)
        ensure_distinct: enable fallback mechanism

    Returns:
        color list
    """
    if origin is None:
        origin = np.array([0.0, 0.0])

    centers = np.atleast_2d(centers)
    n = len(centers)

    if n == 0:
        return []

    # Calculate angles
    shifted = centers - origin
    angles_rad = np.arctan2(shifted[:, 1], shifted[:, 0])
    hues = (np.degrees(angles_rad) % 360).tolist()

    # Point counts → S/V
    if point_counts is not None and len(point_counts) >= n:
        counts = point_counts[:n]
        max_count = max(counts) if counts else 1
        saturations = []
        values = []
        for c in counts:
            ratio = min(c / max_count, 1.0)
            saturations.append(SAT_MIN + (SAT_MAX - SAT_MIN) * ratio)
            values.append(VAL_MIN + (VAL_MAX - VAL_MIN) * ratio)
    else:
        saturations = [DEFAULT_SATURATION] * n
        values = [DEFAULT_VALUE] * n

    # Fallback mechanism
    if ensure_distinct:
        hues, saturations, values = _ensure_colors_distinct(hues, saturations, values)

    # Generate colors
    return [hsv_to_hex(h, s, v) for h, s, v in zip(hues, saturations, values)]


def get_cluster_palettes(n: int,
                         point_counts: List[int] = None,
                         angles: List[float] = None) -> List[ColorPalette]:
    """Get n complete color schemes"""
    if n <= 0:
        return []

    # Determine angles
    if angles is not None and len(angles) >= n:
        hues = [a % 360 for a in angles[:n]]
    else:
        hues = [(i * 360 / n) % 360 for i in range(n)]

    # Determine point counts
    if point_counts is not None and len(point_counts) >= n:
        counts = point_counts[:n]
        max_count = max(counts) if counts else 1
    else:
        counts = None
        max_count = None

    palettes = []
    for i, hue in enumerate(hues):
        if counts is not None:
            palette = generate_palette_from_angle(hue, n_points=counts[i], max_points=max_count)
        else:
            palette = generate_palette_from_angle(hue)
        palettes.append(palette)

    return palettes


def get_palettes_for_centers(centers: np.ndarray,
                              origin: np.ndarray = None,
                              point_counts: List[int] = None) -> List[ColorPalette]:
    """Get complete color schemes based on cluster center coordinates"""
    if origin is None:
        origin = np.array([0.0, 0.0])

    centers = np.atleast_2d(centers)
    n = len(centers)

    if n == 0:
        return []

    shifted = centers - origin
    angles_rad = np.arctan2(shifted[:, 1], shifted[:, 0])
    angles_deg = np.degrees(angles_rad) % 360

    if point_counts is not None and len(point_counts) >= n:
        counts = point_counts[:n]
        max_count = max(counts) if counts else 1
    else:
        counts = None
        max_count = None

    palettes = []
    for i, angle in enumerate(angles_deg):
        if counts is not None:
            palette = generate_palette_from_angle(angle, n_points=counts[i], max_points=max_count)
        else:
            palette = generate_palette_from_angle(angle)
        palettes.append(palette)

    return palettes


# ============================================================================
# Compatibility with old interface
# ============================================================================

def get_color_by_angle(theta: float,
                       saturation: float = DEFAULT_SATURATION,
                       value: float = DEFAULT_VALUE,
                       color_format: str = 'hex') -> Union[str, Tuple]:
    """
    [Legacy interface compatibility] Get color based on polar angle

    Args:
        theta: polar angle (radians)
        saturation: saturation
        value: value
        color_format: 'hex', 'rgb', 'rgb255'
    """
    hue = np.degrees(theta) % 360

    if color_format == 'hex':
        return hsv_to_hex(hue, saturation, value)
    elif color_format == 'rgb':
        r, g, b = colorsys.hsv_to_rgb(hue/360, saturation, value)
        return (r, g, b)
    elif color_format == 'rgb255':
        r, g, b = colorsys.hsv_to_rgb(hue/360, saturation, value)
        return (int(r*255), int(g*255), int(b*255))
    else:
        raise ValueError(f"Unknown color_format: {color_format}")


def get_cluster_colors_from_centers(centers: np.ndarray,
                                     origin: np.ndarray = None,
                                     **kwargs) -> List[str]:
    """[Legacy interface compatibility]"""
    return get_colors_for_centers(centers, origin, **kwargs)


# ============================================================================
# Preset palettes
# ============================================================================

PALETTE_VIBRANT = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                   '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4']

PALETTE_PASTEL = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7',
                  '#dda0dd', '#98d8c8', '#f7dc6f', '#bb8fce', '#85c1e9']


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Paletton-style color system v4 - clean version")
    print("=" * 70)
    print()
    print("Core concept:")
    print("  • Hue = polar angle of cluster center (direct mapping, no adjustment)")
    print("  • Saturation/Value = point count in cluster (more points = more saturated)")
    print()

    # Test 1: Basic usage
    print("【Test 1】Basic usage - 5 clusters")
    print("-" * 50)
    colors = get_cluster_colors(5)
    for i, c in enumerate(colors):
        hue = i * 360 / 5
        print(f"  Cluster {i+1}: {c} (Hue={hue:.0f}°)")

    # Test 2: Specify angles
    print("\n【Test 2】Specify angles - angles map directly")
    print("-" * 50)
    angles = [30, 35, 40]  # very close angles
    colors = get_cluster_colors(3, angles=angles)
    for i, (a, c) in enumerate(zip(angles, colors)):
        print(f"  Cluster {i+1}: angle={a}° → color={c}")
    print("  ↑ Close angles, close colors (this is correct!)")

    # Test 3: Point count affects saturation
    print("\n【Test 3】Point count affects saturation/value")
    print("-" * 50)
    angles = [30, 35, 40]
    point_counts = [200, 50, 100]  # different point counts
    colors = get_cluster_colors(3, angles=angles, point_counts=point_counts)
    for i, (a, c, n) in enumerate(zip(angles, colors, point_counts)):
        print(f"  Cluster {i+1}: angle={a}°, points={n:3d} → color={c}")
    print("  ↑ Similar angles, but different point counts, so different color intensity!")

    # Test 4: Get from coordinates
    print("\n【Test 4】Get colors from center coordinates")
    print("-" * 50)
    centers = np.array([
        [1.0, 0.0],    # 0°
        [0.0, 1.0],    # 90°
        [-1.0, 0.0],   # 180°
    ])
    point_counts = [150, 80, 200]
    colors = get_colors_for_centers(centers, point_counts=point_counts)
    angles = [0, 90, 180]
    for i, (a, c, n) in enumerate(zip(angles, colors, point_counts)):
        print(f"  Cluster {i+1}: angle={a:3d}°, points={n:3d} → {c}")

    # Test 5: Complete color scheme
    print("\n【Test 5】Complete color scheme")
    print("-" * 50)
    palette = generate_palette_from_angle(305, n_points=100, max_points=200)
    print(f"  Hue=305° (purple), points=100/200:")
    print(f"    Primary:    {palette.primary}")
    print(f"    Light:      {palette.light}")
    print(f"    Dark:       {palette.dark}")
    print(f"    Background: {palette.background}")

    print("\n" + "=" * 70)
    print("v4 clean version features:")
    print("  ✓ Angles map directly to hue (no repulsion adjustment)")
    print("  ✓ Point count determines saturation/value (Paletton inner→outer ring)")
    print("  ✓ Even with similar angles, different point counts make them distinguishable")
    print("  ✓ Cleaner code, clearer logic")
    print("=" * 70)