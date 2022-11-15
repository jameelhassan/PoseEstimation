import torch
from torch.testing import assert_allclose

from stacked_hourglass.utils.transforms import fliplr


def test_fliplr(device):
    tensor = torch.as_tensor([[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]], dtype=torch.float32)
    expected = torch.as_tensor([[
        [3, 2, 1],
        [6, 5, 4],
        [9, 8, 7],
    ]], dtype=torch.float32)
    actual = fliplr(tensor)
    assert_allclose(actual, expected)
