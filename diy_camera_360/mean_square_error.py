import numpy as np


def mean_square_error(image1, image2):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    return np.mean((image1 - image2) ** 2)
