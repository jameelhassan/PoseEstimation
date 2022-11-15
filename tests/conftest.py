import os
from pathlib import Path

import pytest
import torch

from stacked_hourglass.datasets.common import DataInfo
from stacked_hourglass.utils.imfit import fit
from stacked_hourglass.utils.imutils import load_image
from stacked_hourglass.utils.transforms import color_normalize

ALL_DEVICES = ['cpu']
# Add available GPU devices.
ALL_DEVICES.extend(f'cuda:{i}' for i in range(torch.cuda.device_count()))


@pytest.fixture(params=ALL_DEVICES)
def device(request):
    return torch.device(request.param)


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA device')
    return torch.device('cuda', torch.cuda.current_device())


@pytest.fixture
def data_dir():
    return Path(__file__).parent.joinpath('data')


@pytest.fixture
def mpii_image_dir():
    image_dir = os.environ.get('MPII_IMAGE_DIR') or '/data/datasets/MPII_Human_Pose/images'
    if not os.path.isdir(image_dir):
        pytest.skip('cannot find MPII image dir')
    return image_dir


@pytest.fixture
def man_running_image(data_dir):
    return load_image(str(data_dir.joinpath('man_running.jpg')))


@pytest.fixture
def man_running_pose():
    return torch.as_tensor([
        [215, 449],  # right_ankle
        [214, 345],  # right_knee
        [211, 244],  # right_hip
        [266, 244],  # left_hip
        [258, 371],  # left_knee
        [239, 438],  # left_ankle
        [237, 244],  # pelvis
        [244, 113],  # spine
        [244,  94],  # neck
        [244,  24],  # head_top
        [179, 198],  # right_wrist
        [182, 142],  # right_elbow
        [199, 103],  # right_shoulder
        [296, 105],  # left_shoulder
        [330, 171],  # left_elbow
        [299, 165],  # left_wrist
    ], dtype=torch.float32)


@pytest.fixture
def example_input(man_running_image):
    mean = torch.as_tensor([0.4404, 0.4440, 0.4327])
    std = torch.as_tensor([0.2458, 0.2410, 0.2468])
    image = fit(man_running_image, (256, 256), fit_mode='contain')
    image = color_normalize(image, mean, std)
    return image.unsqueeze(0)


@pytest.fixture
def h36m_image(data_dir):
    return load_image(str(data_dir.joinpath('h36m.png')))


@pytest.fixture
def h36m_pose():
    return torch.as_tensor([
        [145, 194],  # right_ankle
        [134, 156],  # right_knee
        [142, 120],  # right_hip
        [121, 117],  # left_hip
        [124, 158],  # left_knee
        [129, 199],  # left_ankle
        [132, 118],  # pelvis
        [135,  81],  # spine
        [134,  76],  # neck
        [136,  56],  # head_top
        [149, 125],  # right_wrist
        [151, 106],  # right_elbow
        [146,  83],  # right_shoulder
        [123,  84],  # left_shoulder
        [113, 107],  # left_elbow
        [106, 123],  # left_wrist
    ], dtype=torch.float32)


@pytest.fixture
def dummy_data_info():
    return DataInfo(
        rgb_mean=[0, 0, 0],
        rgb_stddev=[1, 1, 1],
        joint_names=[],
        hflip_indices=[],
    )
