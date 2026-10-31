import numpy as np


def dice_coefficient(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.sum(y_true & y_pred)
    sum_areas = np.sum(y_true) + np.sum(y_pred)
    if sum_areas == 0:
        return 1.0
    return 2.0 * intersection / sum_areas
