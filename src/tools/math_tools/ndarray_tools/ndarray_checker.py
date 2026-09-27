# coding=utf-8
from typing import List, Sequence, Union

import numpy as np
import torch


def check_ndarray_list_composante_shapes_match(
    data: List[Union[np.ndarray, torch.Tensor]],
) -> None:
    """
    Checks that all ndarrays or tensors in a list have matching shapes.

    Given a list of numpy ndarrays or torch tensors, this function validates
    that all elements in the list have identical total number of elements.
    It raises an assertion error if any of the elements differ in size.

    :param data: A list of numpy ndarrays or torch tensors where the size of
        each element should match that of the others in the list.
    :return: This function does not return any value.
    :raises AssertionError: If the sizes of any elements shape in the list do not match.
    """
    assert isinstance(data, Sequence)
    assert isinstance(data[0], (np.ndarray, torch.Tensor))
    for idx in range(len(data) - 1):
        size_a = data[idx].numel() if isinstance(data[idx], torch.Tensor) else data[idx].size
        size_b = data[idx + 1].numel() if isinstance(data[idx + 1], torch.Tensor) else data[idx + 1].size
        assert size_a == size_b, f"Input shape mismatch {data[idx].shape} != {data[idx + 1].shape}"
    return None
