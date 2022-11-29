import torch
from torch import nn
import torch.nn.functional as F
# from nn_layers.cnn_utils import activation_fn, CBR, Shuffle, BR
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

__all__ = ['HourglassNet', 'hg']


# helper function for activations
def activation_fn(features, name='prelu', inplace=True):
    '''
    :param features: # of features (only for PReLU)
    :param name: activation name (prelu, relu, selu)
    :param inplace: Inplace operation or not
    :return:
    '''
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=inplace)
    elif name == 'prelu':
        return nn.PReLU(features)
    else:
        NotImplementedError('Not implemented yet')
        exit()

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and activation function
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, groups=1, act_name='prelu'):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        :param groups: # of groups for group-wise convolution
        :param act_name: Name of the activation function
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)*dilation
        self.cbr = nn.Sequential(
            nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups, dilation=dilation),
            nn.BatchNorm2d(nOut),
            activation_fn(features=nOut, name=act_name)
        )

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.cbr(x)

class CB(nn.Module):
    '''
    This class implements convolution layer followed by batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        :param groups: # of groups for group-wise convolution
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)*dilation
        self.cb = nn.Sequential(
            nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups, dilation=1),
            nn.BatchNorm2d(nOut),
        )

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.cb(x)


class BR(nn.Module):
    '''
    This class implements batch normalization and  activation function
    '''
    def __init__(self, nOut, act_name='prelu'):
        '''
        :param nIn: number of input channels
        :param act_name: Name of the activation function
        '''
        super().__init__()
        self.br = nn.Sequential(
            nn.BatchNorm2d(nOut),
            activation_fn(nOut, name=act_name)
        )

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.br(x)


class Shuffle(nn.Module):
    '''
    This class implements Channel Shuffling
    '''
    def __init__(self, groups):
        '''
        :param groups: # of groups for shuffling
        '''
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class DICE(nn.Module):
    '''
    This class implements the volume-wise seperable convolutions
    '''
    def __init__(self, channel_in, channel_out, height, width, kernel_size=3, dilation=[1, 1, 1], shuffle=True):
        '''
        :param channel_in: # of input channels
        :param channel_out: # of output channels
        :param height: Height of the input volume
        :param width: Width of the input volume
        :param kernel_size: Kernel size. We use the same kernel size of 3 for each dimension. Larger kernel size would increase the FLOPs and Parameters
        :param dilation: It's a list with 3 elements, each element corresponding to a dilation rate for each dimension.
        :param shuffle: Shuffle the feature maps in the volume-wise separable convolutions
        '''
        super().__init__()
        assert len(dilation) == 3
        padding_1 = int((kernel_size - 1) / 2) *dilation[0]
        padding_2 = int((kernel_size - 1) / 2) *dilation[1]
        padding_3 = int((kernel_size - 1) / 2) *dilation[2]
        self.conv_channel = nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, groups=channel_in,
                                      padding=padding_1, bias=False, dilation=dilation[0])
        self.conv_width = nn.Conv2d(width, width, kernel_size=kernel_size, stride=1, groups=width,
                               padding=padding_2, bias=False, dilation=dilation[1])
        self.conv_height = nn.Conv2d(height, height, kernel_size=kernel_size, stride=1, groups=height,
                               padding=padding_3, bias=False, dilation=dilation[2])

        self.br_act = BR(3*channel_in)
        self.weight_avg_layer = CBR(3*channel_in, channel_in, kSize=1, stride=1, groups=channel_in)

        # project from channel_in to Channel_out
        groups_proj = math.gcd(channel_in, channel_out)
        self.proj_layer = CBR(channel_in, channel_out, kSize=3, stride=1, groups=groups_proj)
        self.linear_comb_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(channel_in, channel_in // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_in //4, channel_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.vol_shuffle = Shuffle(3)

        self.width = width
        self.height = height
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.shuffle = shuffle
        self.ksize=kernel_size
        self.dilation = dilation

    def forward(self, x):
        '''
        :param x: input of dimension C x H x W
        :return: output of dimension C1 x H x W
        '''
        bsz, channels, height, width = x.size()
        # process across channel. Input: C x H x W, Output: C x H x W
        out_ch_wise = self.conv_channel(x)

        # process across height. Input: H x C x W, Output: C x H x W
        x_h_wise = x.clone()
        if height != self.height:
            if height < self.height:
                x_h_wise = F.interpolate(x_h_wise, mode='bilinear', size=(self.height, width), align_corners=True)
            else:
                x_h_wise = F.adaptive_avg_pool2d(x_h_wise, output_size=(self.height, width))

        x_h_wise = x_h_wise.transpose(1, 2).contiguous()
        out_h_wise = self.conv_height(x_h_wise).transpose(1, 2).contiguous()

        h_wise_height = out_h_wise.size(2)
        if height != h_wise_height:
            if h_wise_height < height:
                out_h_wise = F.interpolate(out_h_wise, mode='bilinear', size=(height, width), align_corners=True)
            else:
                out_h_wise = F.adaptive_avg_pool2d(out_h_wise, output_size=(height, width))

        # process across width: Input: W x H x C, Output: C x H x W
        x_w_wise = x.clone()
        if width != self.width:
            if width < self.width:
                x_w_wise = F.interpolate(x_w_wise, mode='bilinear', size=(height, self.width), align_corners=True)
            else:
                x_w_wise = F.adaptive_avg_pool2d(x_w_wise, output_size=(height, self.width))

        x_w_wise = x_w_wise.transpose(1, 3).contiguous()
        out_w_wise = self.conv_width(x_w_wise).transpose(1, 3).contiguous()
        w_wise_width = out_w_wise.size(3)
        if width != w_wise_width:
            if w_wise_width < width:
                out_w_wise = F.interpolate(out_w_wise, mode='bilinear', size=(height, width), align_corners=True)
            else:
                out_w_wise = F.adaptive_avg_pool2d(out_w_wise, output_size=(height, width))

        # Merge. Output will be 3C x H X W
        outputs = torch.cat((out_ch_wise, out_h_wise, out_w_wise), 1)
        outputs = self.br_act(outputs)
        if self.shuffle:
            outputs = self.vol_shuffle(outputs)
        outputs = self.weight_avg_layer(outputs)
        linear_wts = self.linear_comb_layer(outputs)
        proj_out = self.proj_layer(outputs)
        return proj_out * linear_wts

    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize}, vol_shuffle={shuffle}, ' \
            'width={width}, height={height}, dilation={dilation})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


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

# Residual block
class BottleneckDice(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, dim, stride=1, downsample=None):
        super(BottleneckDice, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        #  downsample to 128 features
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        # 3x3 conv on the lower resolution
        self.conv2 = DICE(planes, planes, height = dim, width = dim, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(planes)
        # upsample to 256 features
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print(residual.shape)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        # print(out.shape)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        # print(out.shape)

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

    def _make_residual(self, block, num_blocks, planes, dim):
        layers = []
        print(f"[INFO] num of blocks is {num_blocks}")
        for i in range(0, num_blocks): # number of blocks is 1 here (?)
            layers.append(block(planes*block.expansion, planes, dim))
        return nn.Sequential(*layers) 

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        dims = [64, 32, 16, 8]
        hg = []
        for i in range(depth): #for each downsample
            res = []
            for j in range(3):
                #make the same residual block three times: original, upsampled, and for res connection
                res.append(self._make_residual(block, num_blocks, planes, dims[i]))
            if i == 0:
                # for first depth, why do we need an extra residual block?
                res.append(self._make_residual(block, num_blocks, planes, dims[-1])) #narrowest middle part
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
    model = HourglassNet(Bottleneck, BottleneckDice, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
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
