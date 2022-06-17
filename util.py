import numpy as np


def calculate_intersection_over_union(bounding_box_lhs: np.ndarray, bounding_box_rhs: np.ndarray) -> float:

    x_start = max(bounding_box_lhs[0], bounding_box_rhs[0])
    y_start = max(bounding_box_lhs[1], bounding_box_rhs[1])

    x_end = min(bounding_box_lhs[2], bounding_box_rhs[2])
    y_end = min(bounding_box_lhs[3], bounding_box_rhs[3])

    intersection = (x_end - x_start) * (y_end - y_start)
