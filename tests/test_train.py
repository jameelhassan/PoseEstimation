import pytest
import torch
from stacked_hourglass import hg1, hg2
from stacked_hourglass.datasets.mpii import Mpii
from stacked_hourglass.train import do_training_step, do_validation_step, do_validation_epoch, \
    do_training_epoch
from torch.optim import Adam
from torch.utils.data import DataLoader


def test_do_training_step(device):
    model = hg2(pretrained=False)
    model = model.to(device)
    model.train()
    optimiser = Adam(model.parameters())
    inp = torch.randn((1, 3, 256, 256), device=device)
    target = torch.randn((1, 16, 64, 64), device=device)
    output, loss = do_training_step(model, optimiser, inp, target, Mpii.DATA_INFO)
    assert output.shape == (1, 16, 64, 64)
    assert loss > 0


def test_do_training_epoch(cuda_device, mpii_image_dir):
    model = hg1(pretrained=True)
    model = model.to(cuda_device)
    train_dataset = Mpii(mpii_image_dir, is_train=True)
    train_dataset.train_list = train_dataset.train_list[:32]
    optimiser = Adam(model.parameters())
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=2,
                              pin_memory=True)
    do_training_epoch(train_loader, model, cuda_device, Mpii.DATA_INFO, optimiser, quiet=True,
                      acc_joints=Mpii.ACC_JOINTS)


def test_do_validation_step(device):
    model = hg2(pretrained=False)
    model = model.to(device)
    model.eval()
    inp = torch.randn((1, 3, 256, 256), device=device)
    target = torch.randn((1, 16, 64, 64), device=device)
    output, loss = do_validation_step(model, inp, target, Mpii.DATA_INFO)
    assert output.shape == (1, 16, 64, 64)
    assert loss > 0


def test_do_validation_step_flip(device):
    model = hg2(pretrained=False)
    model = model.to(device)
    model.eval()
    inp = torch.randn((1, 3, 256, 256), device=device)
    target = torch.randn((1, 16, 64, 64), device=device)
    output, loss = do_validation_step(model, inp, target, Mpii.DATA_INFO, flip=True)
    assert output.shape == (1, 16, 64, 64)
    assert loss > 0


def test_do_validation_epoch(cuda_device, mpii_image_dir):
    model = hg1(pretrained=True)
    model = model.to(cuda_device)
    val_dataset = Mpii(mpii_image_dir, is_train=False)
    val_dataset.valid_list = val_dataset.valid_list[:32]
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2,
                            pin_memory=True)
    avg_loss, avg_acc, predictions = do_validation_epoch(val_loader, model, cuda_device,
                                                         Mpii.DATA_INFO,
                                                         flip=False, quiet=True,
                                                         acc_joints=Mpii.ACC_JOINTS)
    assert avg_loss == pytest.approx(0.00014652813479187898, abs=1e-6)
    assert avg_acc == pytest.approx(0.8879464417695999, abs=1e-6)
    assert predictions.shape == (32, 16, 2)
