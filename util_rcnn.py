import numpy as np


def calculate_intersection_over_union(bounding_box_lhs: np.ndarray, bounding_box_rhs: np.ndarray) -> float:
    """
    Implementation of an Intersection over union function.
    (https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef)

    Arguments:
    ----------
    bounding_box_lhs (np.ndarray): Bounding box of the first element
    bounding_box_rhs (np.ndarray): Bounding box of the second element
    Returns:
    --------
    (np.ndarray) The ratio of intersection over the total area of the two bounding boxes

    """
    x_start = max(bounding_box_lhs[0], bounding_box_rhs[0])
    y_start = max(bounding_box_lhs[1], bounding_box_rhs[1])

    x_end = min(bounding_box_lhs[2], bounding_box_rhs[2])
    y_end = min(bounding_box_lhs[3], bounding_box_rhs[3])
    if x_end < x_start or y_end < y_start:
        return 0.

    intersection = (x_end - x_start) * (y_end - y_start)

    lhs_area = (bounding_box_lhs[2] - bounding_box_lhs[0]) * \
        (bounding_box_lhs[3] - bounding_box_lhs[1])
    rhs_area = (bounding_box_rhs[2] - bounding_box_rhs[0]) * \
        (bounding_box_rhs[3] - bounding_box_rhs[1])
    return intersection / ((lhs_area + rhs_area) - intersection)
