from dataclasses import dataclass

@dataclass
class PlotOptions:
    show_dmin_dmax_comparison: bool = True
    draw_dmin_dmax_edges: bool = True
    draw_dmax_dot_plot: bool = True
    draw_segment_bars: bool = True
    show_gradient_lines: bool = True  #
    gradient_threshold_ratio: float = 0.25
    show_distance_angle_scatter: bool = True
    show_cartesian_analysis: bool = True  #