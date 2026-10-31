import numpy as np
from scipy import ndimage


def sample_points_from_mask(mask, num_points=1, strategy='center', max_shift=20):
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D")

    y_indices, x_indices = np.where(mask > 0)

    if len(y_indices) == 0:
        h, w = mask.shape
        return np.array([[w//2, h//2]]), np.array([1])

    points = []

    if strategy == 'center':
        center_x = np.mean(x_indices)
        center_y = np.mean(y_indices)
        points.append([center_x, center_y])

        if num_points > 1:
            additional_points = np.random.choice(len(x_indices), 
                                               min(num_points - 1, len(x_indices)), 
                                               replace=False)
            for idx in additional_points:
                points.append([x_indices[idx], y_indices[idx]])

    elif strategy == 'random':
        sampled_indices = np.random.choice(len(x_indices), 
                                         min(num_points, len(x_indices)), 
                                         replace=False)
        for idx in sampled_indices:
            points.append([x_indices[idx], y_indices[idx]])

    elif strategy == 'boundary':
        boundary = mask - ndimage.binary_erosion(mask)
        by_indices, bx_indices = np.where(boundary > 0)

        if len(by_indices) > 0:
            boundary_samples = min(num_points, len(bx_indices))
            sampled_boundary = np.random.choice(len(bx_indices), boundary_samples, replace=False)
            for idx in sampled_boundary:
                points.append([bx_indices[idx], by_indices[idx]])

            if len(points) < num_points:
                inner_mask = mask - boundary
                iy_indices, ix_indices = np.where(inner_mask > 0)
                if len(iy_indices) > 0:
                    remaining = num_points - len(points)
                    sampled_inner = np.random.choice(len(ix_indices), 
                                                   min(remaining, len(ix_indices)), 
                                                   replace=False)
                    for idx in sampled_inner:
                        points.append([ix_indices[idx], iy_indices[idx]])
        else:
            sampled_indices = np.random.choice(len(x_indices), 
                                             min(num_points, len(x_indices)), 
                                             replace=False)
            for idx in sampled_indices:
                points.append([x_indices[idx], y_indices[idx]])

    while len(points) < num_points and len(x_indices) > 0:
        idx = np.random.randint(len(x_indices))
        points.append([x_indices[idx], y_indices[idx]])

    points = np.array(points[:num_points])
    labels = np.ones(len(points), dtype=int)

    if max_shift > 0:
        h, w = mask.shape
        for i in range(len(points)):
            shift_x = np.random.randint(-max_shift, max_shift + 1)
            shift_y = np.random.randint(-max_shift, max_shift + 1)

            points[i, 0] = np.clip(points[i, 0] + shift_x, 0, w - 1)
            points[i, 1] = np.clip(points[i, 1] + shift_y, 0, h - 1)

    return points, labels
