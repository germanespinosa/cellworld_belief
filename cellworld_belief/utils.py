import math
import typing

import numpy as np
import torch


def get_index(value, lower_bound, step) -> typing.Tuple[int, int, float]:
    index_value = (value - lower_bound) / step
    low_bound = math.floor(index_value)
    distance = index_value - low_bound
    if distance <= .5:
        index = low_bound
    else:
        index = low_bound + 1
    return index, low_bound, distance


def gaussian_tensor(dimensions,
                    sigma,
                    center=None,
                    device=torch.device('cpu')) -> torch.Tensor:
    width, height = dimensions

    if center is None:
        center_x, center_y = width/2 - .5, height/2 - .5
    else:
        center_x, center_y = center

    # Create x and y coordinates (0 to size-1)
    x = torch.arange(width, device=device)
    y = torch.arange(height, device=device)

    # Create meshgrid of coordinates
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Calculate Gaussian
    gaussian = torch.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))

    return gaussian


def fuse_tensors(tensors, weights)-> torch.Tensor:
    if len(tensors) != len(weights):
        raise ValueError("Number of tensors and weights must match")

    if not all(w >= 0 for w in weights):
        raise ValueError("Weights must be non-negative")

    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must sum to 1")

    device = tensors[0].device
    tensors = [t.to(device) for t in tensors]
    weights = torch.tensor(weights, device=device)
    weights = weights.view(-1, * [1 for _ in range(tensors[0].ndim)])
    fused_tensor = torch.sum(weights * torch.stack(tensors, dim=0), dim=0)
    return fused_tensor


def shift_tensor(tensor, displacement) -> torch.Tensor:
    def shift_tensor_int(displacement_int):
        d_x_int, d_y_int = displacement_int
        displaced = torch.zeros_like(tensor, device=tensor.device)
        if d_x_int > 0:
            if d_y_int > 0:
                displaced[d_x_int:, d_y_int:] = tensor[:-d_x_int, :-d_y_int]
            elif d_y_int < 0:
                displaced[d_x_int:, :d_y_int] = tensor[:-d_x_int, -d_y_int:]
            else:
                displaced[d_x_int:, :] = tensor[:-d_x_int, :]
        elif d_x_int < 0:
            if d_y_int > 0:
                displaced[:d_x_int, d_y_int:] = tensor[-d_x_int:, :-d_y_int]
            elif d_y_int < 0:
                displaced[:d_x_int, :d_y_int] = tensor[-d_x_int:, -d_y_int:]
            else:
                displaced[:d_x_int, :] = tensor[-d_x_int:, :]
        else:
            if d_y_int > 0:
                displaced[:, d_y_int:] = tensor[:, :-d_y_int]
            elif d_y_int < 0:
                displaced[:, :d_y_int] = tensor[:, -d_y_int:]
            else:
                displaced[:, :] = tensor[:, :]
        return displaced

    d_x, d_y = displacement
    if type(d_x) is int and type(d_y) is int:
        return shift_tensor_int(displacement_int=displacement)

    d_x_l, d_y_l = math.floor(d_x),  math.floor(d_y)
    d_x_h, d_y_h = d_x_l + 1, d_y_l + 1
    w_x = d_x - d_x_l
    w_y = d_y - d_y_l
    weights = [(1-w_x) * (1-w_y), (1-w_x) * w_y, w_x * (1-w_y), w_x * w_y]
    displacements = [(d_x_l, d_y_l), (d_x_l, d_y_h), (d_x_h, d_y_l), (d_x_h, d_y_h)]
    tensors = [shift_tensor_int(displacement_int=d) for w, d in zip(weights, displacements) if w > 0]
    weights = [w for w in weights if w > 0]
    return fuse_tensors(tensors=tensors, weights=weights)
