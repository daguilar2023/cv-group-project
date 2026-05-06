import numpy as np


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_mar(mouth_points):
    """
    mouth_points should contain 8 points:
    p1, p2, p3, p4, p5, p6, p7, p8
    in the correct order for MAR calculation.

    MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (2 * ||p1 - p5||)
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = mouth_points

    vertical_1 = euclidean_distance(p2, p8)
    vertical_2 = euclidean_distance(p3, p7)
    vertical_3 = euclidean_distance(p4, p6)
    horizontal = euclidean_distance(p1, p5)

    if horizontal == 0:
        return 0.0

    mar = (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)
    return mar
