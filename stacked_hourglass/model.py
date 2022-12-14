'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from config import INPLANES, NUM_FEATS, EXPANSION, SEPARABLE_ALL, SEPARABLE_3x3, CONCAT, PERCEPTUAL_RES, PERCEPTUAL_LOSS


__all__ = ['HourglassNet', 'hg']

model_urls = {
    'hg1': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg1-ce125879.pth',
    'hg2': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg2-15e342d9.pth',
    'hg8': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg8-90e5d470.pth',
}


class Bottleneck(nn.Module):
    # EDIT With Separable Convolution
    expansion = EXPANSION

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        separable = SEPARABLE_ALL
        separable_3x3 = SEPARABLE_3x3
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = SeparableConv2D(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True) \
            if separable else nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = SeparableConv2D(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True) \
            if (separable or separable_3x3) else nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = SeparableConv2D(planes, int(planes * Bottleneck.expansion), kernel_size=3, stride=stride, padding=1, bias=True) \
            if separable else nn.Conv2d(planes, int(planes * Bottleneck.expansion), kernel_size=1, bias=True) 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x 
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)
        self._perceptuals = None
        self._percept_loss = None

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(int(planes*block.expansion), planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))  # For narrowest layer
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x, perceptual=None):
        up1 = self.hg[n-1][0](x)    # The RES connection
        low1 = F.max_pool2d(x, 2, stride=2) # Downsampling
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1, perceptual=perceptual)
        else:
            low2 = self.hg[n-1][3](low1)    # For narrowest layer
            self._percept_loss = low2   # perceptual at narrowest point for perceptual loss, before res connection addition

            if perceptual is not None:
                low2 += perceptual  # Skip connection from previous hourglass perceptual
            self._perceptuals = low2    # perceptual at narrowest point for next
            
        low3 = self.hg[n-1][2](low2)    # RES block in upsample path
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x, perceptual=None):
        return self._hour_glass_forward(self.depth, x, perceptual=perceptual)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    concat = CONCAT
    perceptualRes = PERCEPTUAL_RES

    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()
        
        self.inplanes = INPLANES  # Edit from 64
        self.num_feats = NUM_FEATS # Edit from 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        if HourglassNet.concat:
            self.concat1 = nn.Conv2d(2 * self.num_feats * block.expansion, int(self.num_feats * block.expansion), kernel_size=1)

        # build hourglass modules
        ch = int(self.num_feats*block.expansion)
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != int(planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, int(planes * block.expansion),
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = int(planes * block.expansion)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = []
        loss_perceptuals = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            # Skip connection from perceptuals
            if i > 0 and HourglassNet.perceptualRes:
                y = self.hg[i](x, perceptual)
            else:
                y = self.hg[i](x)
            perceptual = self.hg[i]._perceptuals    # Get perceptual
            loss_perceptuals.append(self.hg[i]._percept_loss)    # Get perceptual for loss
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)    #Heatmap prediction
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                if HourglassNet.concat:
                    x = x + score_
                    x = torch.cat([x, fc_], dim=1)
                    x = self.concat1(x)
                else:
                    x = x + fc_ + score_
        
        if PERCEPTUAL_LOSS:
            return out, loss_perceptuals
        else:
            return out


def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model


def _hg(arch, pretrained, progress, **kwargs):
    model = hg(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress,
                                              map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    return model


def hg1(pretrained=False, progress=True, num_blocks=1, num_classes=16):
    return _hg('hg1', pretrained, progress, num_stacks=1, num_blocks=num_blocks,
               num_classes=num_classes)


def hg2(pretrained=False, progress=True, num_blocks=1, num_classes=16, separable=False):
    return _hg('hg2', pretrained, progress, num_stacks=2, num_blocks=num_blocks,
               num_classes=num_classes, separable=separable)


def hg3(pretrained=False, progress=True, num_blocks=1, num_classes=16, separable=False):
    return _hg('hg3', pretrained, progress, num_stacks=3, num_blocks=num_blocks,
               num_classes=num_classes)

def hg8(pretrained=False, progress=True, num_blocks=1, num_classes=16):
    return _hg('hg8', pretrained, progress, num_stacks=8, num_blocks=num_blocks,
               num_classes=num_classes)



#### Additional Classes

def count_parameters(module: nn.Module, trainable: bool = True) -> int:

    if trainable:
        num_parameters = sum(p.numel() for p in module.parameters()
                             if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in module.parameters())

    return num_parameters


def conv_parameters(in_channels, out_channels, kernel_size, bias) -> int:

    num_parameters = in_channels * out_channels * kernel_size[0] * kernel_size[
        1]

    if bias:
        num_parameters += out_channels

    return num_parameters


def separable_conv_parameters(in_channels, out_channels, kernel_size,
                              bias) -> int:

    num_parameters = in_channels * kernel_size[0] * kernel_size[
        1] + in_channels * out_channels

    if bias:
        num_parameters += (in_channels + out_channels)

    return num_parameters


class DepthwiseConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.depthwise_conv(x)

        return x


class PointwiseConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(1, 1),
                                        stride=1,
                                        padding=0,
                                        dilation=1,
                                        groups=1,
                                        bias=bias,
                                        padding_mode='zeros')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pointwise_conv(x)

        return x


class SeparableConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 dtype=None) -> None:
        super().__init__()

        self.depthwise_conv = DepthwiseConv2D(in_channels=in_channels,
                                              out_channels=in_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=in_channels,
                                              bias=bias,
                                              padding_mode=padding_mode)

        self.pointwise_conv = PointwiseConv2D(in_channels=in_channels,
                                              out_channels=out_channels,
                                              bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x