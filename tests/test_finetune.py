import torch
from stacked_hourglass import hg2
from stacked_hourglass.utils.finetune import change_hg_outputs
from torch.testing import assert_allclose


def test_reorder_hg_outputs(device):
    in_data = torch.randn((4, 3, 256, 256), dtype=torch.float32, device=device)

    model = hg2(pretrained=True).to(device)
    orig_output = model(in_data)

    output_indices = list(reversed(range(16)))
    change_hg_outputs(model, output_indices)
    new_output = model(in_data)

    for orig_stage_output, new_stage_output in zip(orig_output, new_output):
        assert_allclose(new_stage_output, orig_stage_output.flip(1))
        assert_allclose(new_stage_output, orig_stage_output.flip(1))


def test_change_hg_outputs(device):
    in_data = torch.randn((4, 3, 256, 256), dtype=torch.float32, device=device)

    model = hg2(pretrained=True).to(device)
    orig_output = model(in_data)

    # New output, left ankle, right ankle, new output, new output.
    output_indices = [None, 5, 0, None, None]
    change_hg_outputs(model, output_indices)
    new_output = model(in_data)

    orig_first_stage_output = orig_output[0]
    new_first_stage_output = new_output[0]

    assert_allclose(new_first_stage_output[:, 1], orig_first_stage_output[:, 5])
    assert_allclose(new_first_stage_output[:, 2], orig_first_stage_output[:, 0])
