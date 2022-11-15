from dataclasses import dataclass
from typing import List


@dataclass
class DataInfo:
    rgb_mean: List[float]
    rgb_stddev: List[float]
    joint_names: List[str]
    hflip_indices: List[int]
