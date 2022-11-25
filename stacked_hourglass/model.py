'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


__all__ = ['HourglassNet', 'hg']


model_urls = {
    'hg1': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg1-ce125879.pth',
    'hg2': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg2-15e342d9.pth',
    'hg8': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg8-90e5d470.pth',
}


BatchNorm = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


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
                 dtype=None,**kwargs) -> None:
        super(DepthwiseConv2D, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding_mode=padding_mode)
                                        # device=device,
                                        # dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.depthwise_conv(x)

        return x

class PointwiseConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 device=None,
                 dtype=None,**kwargs) -> None:
        super(PointwiseConv2D, self).__init__()

        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(1, 1),
                                        stride=1,
                                        padding=0,
                                        dilation=1,
                                        groups=1,
                                        bias=bias,
                                        padding_mode='zeros')
                                        # device=device,
                                        # dtype=dtype)

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
                 device=None,
                 dtype=None,**kwargs) -> None:
        super(SeparableConv2D, self).__init__()

        self.depthwise_conv = DepthwiseConv2D(in_channels=in_channels,
                                              out_channels=in_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=in_channels,
                                              bias=bias,
                                              padding_mode=padding_mode,
                                              device=device,
                                              dtype=dtype)

        self.pointwise_conv = PointwiseConv2D(in_channels=in_channels,
                                              out_channels=out_channels,
                                              bias=bias,
                                              device=device,
                                              dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x

# Add a 3x3 conv with dilation = 4
class MultiDilatedLarge(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1,2,3,4), residual=True):
        super(MultiDilatedLarge, self).__init__()

        # 1x1 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = BatchNorm(planes)
        # 3x3 dep sep conv dilation = 1
        self.conv2 = SeparableConv2D(planes, planes, kernel_size = 3, stride = stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn2 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 dep sep conv dilation = 2
        self.conv3 = SeparableConv2D(planes, planes, kernel_size = 3, stride = stride,
                             padding=dilation[1], dilation=dilation[1])
        self.bn3 = BatchNorm(planes)

        # 3x3 dep sep conv dilation = 3
        self.conv4 = SeparableConv2D(planes, planes, kernel_size = 3, stride = stride,
                             padding=dilation[2], dilation=dilation[2])
        self.bn4 = BatchNorm(planes)
    
        # 3x3 dep sep conv dilation = 4
        self.conv5 = SeparableConv2D(planes, planes, kernel_size = 3, stride = stride,
                             padding=dilation[3], dilation=dilation[3])
        self.bn5 = BatchNorm(planes)

        # 1x1 conv
        self.conv6 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=True)
        self.bn6 = BatchNorm(planes*2)

        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        d1 = out

        out1 = self.conv2(d1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out2 = self.conv3(d1)
        out2 = self.bn3(out2)
        out2 = self.relu(out2)

        out3 = self.conv4(d1)
        out3 = self.bn4(out3)
        out3 = self.relu(out3)

        out4 = self.conv5(d1)
        out4 = self.bn5(out4)
        out4 = self.relu(out4)

        # out = torch.cat([out1, out2, out3], dim=1)
        out = out1 + out2 + out3 + out4

        out = self.conv6(out)
        out = self.bn6(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out

class MultiDilated(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1,2,3), residual=True):
        super(MultiDilated, self).__init__()

        # 1x1 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = BatchNorm(planes)
        # 3x3 dep sep conv dilation = 1
        self.conv2 = SeparableConv2D(planes, planes, kernel_size = 3, stride = stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn2 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 dep sep conv dilation = 2
        self.conv3 = SeparableConv2D(planes, planes, kernel_size = 3, stride = stride,
                             padding=dilation[1], dilation=dilation[1])
        self.bn3 = BatchNorm(planes)

        # 3x3 dep sep conv dilation = 3
        self.conv4 = SeparableConv2D(planes, planes, kernel_size = 3, stride = stride,
                             padding=dilation[2], dilation=dilation[2])
        self.bn4 = BatchNorm(planes)

        # 1x1 conv
        self.conv5 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=True)
        self.bn5 = BatchNorm(planes*2)

        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        d1 = out

        out1 = self.conv2(d1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out2 = self.conv3(d1)
        out2 = self.bn3(out2)
        out2 = self.relu(out2)

        out3 = self.conv4(d1)
        out3 = self.bn4(out3)
        out3 = self.relu(out3)

        # out = torch.cat([out1, out2, out3], dim=1)
        out = out1 + out2 + out3

        out = self.conv5(out)
        out = self.bn5(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


# Residual block
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        #  downsample to 128 features
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        # 3x3 conv on the lower resolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        # upsample to 256 features
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
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
        # skip connection
        out += residual

        return out

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth # how many downsampling steps we have
        self.block = block # bottleneck
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth) # one hourglass

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        # print(f"[INFO] num of blocks is {num_blocks}")
        for i in range(0, num_blocks): # number of blocks is 1 here (?)
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers) 

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth): #for each downsample
            res = []
            for j in range(3):
                #make the same residual block three times: original, upsampled, and for res connection
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                # for first depth, why do we need an extra residual block?
                res.append(self._make_residual(block, num_blocks, planes)) #narrowest middle part
            hg.append(nn.ModuleList(res))
        # list of list of residual blocks per depth
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        # n is the depth
        # starts from the outer residual block
        up1 = self.hg[n-1][0](x) # res block in res connection
        low1 = F.max_pool2d(x, 2, stride=2) # downsample
        low1 = self.hg[n-1][1](low1) # res block in encoder part

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1) #narrowest layer
        low3 = self.hg[n-1][2](low2) # res block in decoder part
        up2 = F.interpolate(low3, scale_factor=2) # upsample
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, dilatedblock, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        # before hourglasses
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(dilatedblock, num_blocks, self.num_feats, 4))
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
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y) # 1st block after hourglass
            score = self.score[i](y) # heatmap
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y) #2nd block after hourglass
                score_ = self.score_[i](score) # projected heatmap
                x = x + fc_ + score_ # input + hourglass feature map + hourglass prediction projected to 256

        return out


def hg(**kwargs):
    model = HourglassNet(Bottleneck, MultiDilated, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model


def _hg(arch, pretrained, progress, **kwargs):
    model = hg(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress,
                                              map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
    return model


def hg1(pretrained=False, progress=True, num_blocks=1, num_classes=16):
    return _hg('hg1', pretrained, progress, num_stacks=1, num_blocks=num_blocks,
               num_classes=num_classes)


def hg2(pretrained=False, progress=True, num_blocks=1, num_classes=16):
    return _hg('hg2', pretrained, progress, num_stacks=2, num_blocks=num_blocks,
               num_classes=num_classes)


def hg8(pretrained=False, progress=True, num_blocks=1, num_classes=16):
    return _hg('hg8', pretrained, progress, num_stacks=8, num_blocks=num_blocks,
               num_classes=num_classes)
