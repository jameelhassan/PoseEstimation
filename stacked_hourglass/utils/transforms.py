import numpy as np
import torch

from .imutils import im_to_numpy, im_to_torch
from .misc import to_torch
from .pilutil import imresize, imrotate


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x


def flip_back(flip_output, hflip_indices):
    """flip and rearrange output maps"""
    return fliplr(flip_output)[:, hflip_indices]


def shufflelr(x, width, hflip_indices):
    """flip and rearrange coords"""
    # Flip horizontal
    x[:, 0] = width - x[:, 0]
    # Change left-right parts
    x = x[hflip_indices]
    return x


def fliplr(x):
    """Flip images horizontally."""
    if torch.is_tensor(x):
        return torch.flip(x, [-1])
    else:
        return np.ascontiguousarray(np.flip(x, -1))


def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    scale = np.asarray(scale).flatten()
    if len(scale) == 1:
        scale = scale.repeat(2)

    # Generate transformation matrix
    h = 200 * scale[0]
    w = 200 * scale[1]
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def transform_preds(coords, center, scale, res):
    for p in range(coords.size(0)):
        coords[p, 0:2] = to_torch(transform(coords[p, 0:2], center, scale, res, 1, 0))
    return coords


def crop(img, center, scale, res, rot=0):
    """
    img: torch.Tensor: shape: C,H,W
    center: torch.Tensor: shape: x, y
    scale: [H/200, W/200]
    res: H,W
    """
    scale = np.asarray(scale).flatten()
    if len(scale) == 1:
        scale = scale.repeat(2)

    img = im_to_numpy(img)

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    # If we set the scale properly at input, then this should be the same per-dimension
    sf = [scale[0] * 200.0 / res[0], scale[1] * 200.0 / res[1]]

    # NOTE: This seems to be handled, but is untested
    if sf[0] < 2 or sf[1] < 2:
        sf = [1, 1]
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf[0]))
        new_ht = int(np.math.floor(ht / sf[0]))
        new_wd = int(np.math.floor(wd / sf[1]))
        if new_size < 2:    # This returns a fully black image...
            return torch.zeros(res[0], res[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            img = imresize(img, [new_ht, new_wd])
            center = center * 1.0 / torch.tensor(sf[::-1])  # Center is x, y. Sf is H,W
            scale = [scale[0] / sf[0], scale[1] / sf[1]]

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    # NOTE: res stored as H, W. We want coordinates in x, y
    br = np.array(transform(res[::-1], center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(img.shape[1], br[0])
    old_y = max(0, ul[1]), min(img.shape[0], br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = im_to_torch(imresize(new_img, res))
    return new_img
