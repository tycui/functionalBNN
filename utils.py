import numpy as np

def median_distance_global(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.sqrt(np.sum((x_col - x_row) ** 2, -1)) # [n, n]
    return np.median(dis_a)


def median_distance_local(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.abs(x_col - x_row) # [n, n, d]
    dis_a = np.reshape(dis_a, [-1, dis_a.shape[-1]])
    return np.median(dis_a, 0) * (x.shape[1] ** 0.5)