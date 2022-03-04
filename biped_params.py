import numpy as np


def rectangular_foot_CoP():
    """
    OUTPUTS
    d ([4, 2] matrix): normal vectors of the four edges;
    b ([4, 1] matrix): center to edge perpendicular distance
    along the normal vectors of the corresponding edges;
    """
    d = np.array([[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.125], [0.08], [0.125], [0.08]])

    return d, b


def init_double_support_CoP():
    """
    OUTPUTS
    d ([4, 2] matrix): normal vectors of the four edges;
    b ([4, 1] matrix): center to edge perpendicular distance
    along the normal vectors of the corresponding edges;
    """
    d = np.array([[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.125], [0.23], [0.125], [0.23]])

    return d, b


def support_foot_limit(foot="left"):
    norm_vectors = np.array(
        [
            [-3 / np.sqrt(13), 2 / np.sqrt(13)],
            [-1 / np.sqrt(5), 2 / np.sqrt(5)],
            [0, -1],
            [1 / np.sqrt(5), 2 / np.sqrt(5)],
            [3 / np.sqrt(13), 2 / np.sqrt(13)],
        ]
    )

    # distances = np.array(
    #     [
    #         [6 / (5 * np.sqrt(13))],
    #         [4 / (5 * np.sqrt(5))],
    #         [-0.15],
    #         [4 / (5 * np.sqrt(5))],
    #         [6 / (5 * np.sqrt(13))],
    #     ]
    # )

    distances = np.array(
        [
            [0.4160251471689218],
            [0.49193495504995377],
            [-0.3],
            [0.49193495504995377],
            [0.4160251471689218],
        ]
    )

    if foot == "left":
        support_norm_vecs = norm_vectors
        support_dis = distances
    elif foot == "right":
        support_norm_vecs = np.hstack((norm_vectors[:, :1], -norm_vectors[:, 1:]))
        support_dis = distances
    else:
        raise ValueError("Unsupported foot type, only left and right are allowed")

    return support_norm_vecs, support_dis


def support_foot_limit2(foot="left"):
    norm_vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    distances = np.array([[0.3], [-1], [0.3], [1]])
    if foot == "left":
        support_norm_vecs = norm_vectors
        support_dis = distances
    elif foot == "right":
        support_norm_vecs = np.hstack((norm_vectors[:, :1], -norm_vectors[:, 1:]))
        support_dis = distances
    else:
        raise ValueError("Unsupported foot type, only left and right are allowed")

    return support_norm_vecs, support_dis


def walking_straight():
    N = np.array([[1, 0], [0, 1]])
    vmax = np.array([[10.0], [10.0]])

    return N, vmax


def get_line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    a = -(y2 - y1) / (x2 - x1)
    b = 1
    c = a * x1 + y1

    return a, b, c


def distance2line(line, p):
    a, b, c = line
    x, y = p
    d = np.absolute(a * x + b * y - c) / np.sqrt(a ** 2 + b ** 2)

    return d


def point_from_line_edge(e1, e2, p):
    a, b, c = get_line_from_points(e1, e2)
    d = distance2line((a, b, c), p)

    return d


def main():
    # edges = [(-0.3, 0.3), (-0.2, 0.45), (0.0, 0.55), (0.2, 0.45), (0.3, 0.3)]
    edges = [(-0.3, 0.15), (-0.2, 0.3), (0.0, 0.4), (0.2, 0.3), (0.3, 0.15)]
    p = (0, 0)
    d1 = point_from_line_edge(edges[0], edges[1], p)
    d2 = point_from_line_edge(edges[1], edges[2], p)
    d3 = point_from_line_edge(edges[2], edges[3], p)
    d4 = point_from_line_edge(edges[3], edges[4], p)
    d5 = point_from_line_edge(edges[0], edges[4], p)

    print(d1, d2, d3, d4, d5)

    _, support_dis = support_foot_limit()
    print(support_dis)


if __name__ == "__main__":
    main()