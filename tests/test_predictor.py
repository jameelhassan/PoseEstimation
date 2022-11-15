import torch
from torch.testing import assert_allclose

from stacked_hourglass import HumanPosePredictor, hg2
from stacked_hourglass.utils.imfit import fit


def test_do_forward(device, example_input):
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    output = predictor.do_forward(example_input.to(device))
    assert len(output) == 2  # Expect one set of heatmaps per stack.
    heatmaps = output[-1]
    assert heatmaps.shape == (1, 16, 64, 64)


def test_prepare_image(device, man_running_image):
    man_running_image = man_running_image.to(device)
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    orig_image = man_running_image.clone()
    image = predictor.prepare_image(orig_image)
    assert_allclose(orig_image, man_running_image)  # Input image should be unchanged.
    assert image.shape == (3, 256, 256)
    assert image.device == device


def test_prepare_image_mostly_ready(device):
    # This test is for preparing an image which already has the correct dtype and size.
    image_float32 = torch.empty((3, 256, 256), device=device, dtype=torch.float32).uniform_()
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    orig_image = image_float32.clone()
    image = predictor.prepare_image(orig_image)
    assert_allclose(image_float32, orig_image)  # Input image should be unchanged.
    assert image.shape == (3, 256, 256)
    assert image.device == device


def test_prepare_image_aspect_ratio(device, dummy_data_info):
    orig_image = torch.ones((3, 256, 512), dtype=torch.float32, device=device)
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device, data_info=dummy_data_info)
    image = predictor.prepare_image(orig_image)
    expected = torch.zeros((3, 256, 256), dtype=torch.float32, device=device)
    expected[:, 64:192] = 1.0
    assert_allclose(image, expected)


def test_estimate_heatmaps(device, man_running_image):
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    heatmaps = predictor.estimate_heatmaps(man_running_image)
    assert heatmaps.shape == (16, 64, 64)


def test_asymmetric_input(device, man_running_image):
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device, input_shape=(512, 64))
    orig_image = man_running_image.clone()
    image = predictor.prepare_image(orig_image)
    assert image.shape == (3, 512, 64)
    heatmaps = predictor.estimate_heatmaps(image)
    assert heatmaps.shape == (16, 128, 16)
    joints = predictor.estimate_joints(image)
    assert all(joints[:,  0] < 64)
    assert all(joints[:,  1] < 512)


def test_estimate_joints(device, man_running_image, man_running_pose):
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    joints = predictor.estimate_joints(man_running_image)
    assert joints.shape == (16, 2)
    assert_allclose(joints, man_running_pose, rtol=0, atol=20)


def test_estimate_joints_fit_contain(device, man_running_image, man_running_pose):
    # Crop the example so that it is no longer square.
    narrow_width = 256
    image = fit(man_running_image, (512, narrow_width), fit_mode='cover')
    gt_joints = man_running_pose.clone()
    gt_joints[..., 0] -= (512 - narrow_width) / 2
    # Run inference, enforcing square input.
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device, input_shape=(256, 256))
    joints = predictor.estimate_joints(image)
    # Check that the results are as expected.
    assert joints.shape == (16, 2)
    assert_allclose(joints, gt_joints, rtol=0, atol=20)


def test_estimate_joints_with_flip(device, man_running_image, man_running_pose):
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    joints = predictor.estimate_joints(man_running_image, flip=True)
    assert joints.shape == (16, 2)
    assert_allclose(joints, man_running_pose, rtol=0, atol=20)


def test_estimate_joints_h36m(device, h36m_image, h36m_pose):
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    joints = predictor.estimate_joints(h36m_image)
    assert joints.shape == (16, 2)
    assert_allclose(joints, h36m_pose, rtol=0, atol=15)


def test_estimate_joints_tensor_batch(device, h36m_image, h36m_pose):
    model = hg2(pretrained=True)
    predictor = HumanPosePredictor(model, device=device)
    batch_size = 4
    joints = predictor.estimate_joints(h36m_image.repeat(batch_size, 1, 1, 1))
    assert joints.shape == (batch_size, 16, 2)
    assert_allclose(joints, h36m_pose.repeat(batch_size, 1, 1), rtol=0, atol=15)
