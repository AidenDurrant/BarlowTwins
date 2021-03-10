# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from math import pi

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'projection_MLP', 'Sup_Head', 'alexnet', 'projection_pred']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

'''Resnet Class

    Taken from:

    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

    def forward(self, x):
        print("WS")
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        print("WS")
        weight = self.weight

        weight_mean = weight.mean(dim=1, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.linear(x, weight, self.bias)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, norm_layer=None):
    """3x3 convolution with padding"""

    if norm_layer == nn.GroupNorm:
        return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=dilation, groups=groups, bias=False, dilation=dilation)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, norm_layer=None):
    """1x1 convolution"""

    if norm_layer == nn.GroupNorm:
        return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def fc_layer(in_features, out_features, bias=True, norm_layer=None):
    """1x1 convolution"""

    if norm_layer == nn.GroupNorm:
        return nn.Linear(in_features, out_features, bias=bias)
    else:
        return nn.Linear(in_features, out_features, bias=bias)


def norm_l(norm_layer, planes):

    if norm_layer == nn.GroupNorm:
        return nn.GroupNorm(16, planes)

    elif norm_layer == nn.BatchNorm2d or norm_layer == None:
        return nn.BatchNorm2d(planes)

    else:
        raise NotImplementedError('Norm not supported: {}'.format(norm_layer))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, norm_layer=norm_layer)
        self.bn1 = norm_l(norm_layer, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, norm_layer=norm_layer)
        self.bn2 = norm_l(norm_layer, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, norm_layer=norm_layer)
        self.bn1 = norm_l(norm_layer, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, norm_layer=norm_layer)
        self.bn2 = norm_l(norm_layer, width)
        self.conv3 = conv1x1(width, planes * self.expansion, norm_layer=norm_layer)
        self.bn3 = norm_l(norm_layer, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dataset=None):
        super(ResNet, self).__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Different model for smaller image size
        if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'tinyimagenet':

            # CIFAR Stem

            self.stem = nn.Sequential()

            self.stem.add_module('conv0', nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                                    bias=False))
            self.stem.add_module('BN1', norm_l(norm_layer, self.inplanes))
            self.stem.add_module('ReLU1', nn.ReLU(inplace=True))

        # e.g. ImageNet
        else:

            self.stem = nn.Sequential()

            self.stem.add_module('conv0', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                                    bias=False))
            self.stem.add_module('BN1', norm_l(norm_layer, self.inplanes))
            self.stem.add_module('ReLU1', nn.ReLU(inplace=True))
            self.stem.add_module('MaxPool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        #
        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, norm_layer=norm_layer),
                norm_l(norm_layer, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


class SmallAlexNet(nn.Module):
    def __init__(self,):
        super(SmallAlexNet, self).__init__()

        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ))

        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ))

        # conv_block_3
        blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_4
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_5
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ))

        # fc6
        blocks.append(nn.Sequential(
            nn.Flatten(),  # 192 * 7 * 7 or 256 * 6 * 6 if 224 * 224
            # nn.Linear(192 * 4 * 4, 2)
        ))

        self.blocks = nn.ModuleList(blocks)

        self.fc = nn.Linear(192 * 4 * 4, 192 * 4 * 4)

        # self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

        self.apply(init)

    def forward(self, x, *, layer_index=-1):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer in self.blocks[:(layer_index + 1)]:
            # print(x.size())
            x = layer(x)
        x = self.fc(x)
        return x


def alexnet(**kwargs):
    return SmallAlexNet()


def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        args: arguments
    """
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        args: arguments
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        args: arguments
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        args: arguments
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        args: arguments
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


''' SimCLR Projection and Classification Heads '''


class Sup_Head(nn.Module):
    '''Supervised Classification head for the finetuning of the resnet encoder.

        - Uses the dataset and model size to determine encoder output
            representation dimension.
    '''

    def __init__(self, model, n_classes):
        super(Sup_Head, self).__init__()

        if '18' in model or '34' in model:
            n_channels = 512
        elif '50' in model or '101' in model or '152' in model:
            n_channels = 2048
        elif model == 'alexnet'or model == 'mhe_alexnet':
            n_channels = 192 * 4 * 4  # For CIFAR10 (3,32,32)
        else:
            raise NotImplementedError('model not supported: {}'.format(model))

        self.classifier = nn.Sequential()

        self.classifier.add_module('W1', nn.Linear(
            n_channels, n_classes))

        # self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.01)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x):
        return self.classifier(x)


class projection_MLP(nn.Module):
    def __init__(self, model='resnet18', output_dim=256, hidden_dim=4096, norm_layer=None):
        '''Projection head for the pretraining of the resnet encoder.

            - Uses the dataset and model size to determine encoder output
                representation dimension.
            - Outputs to a dimension of 128, and uses non-linear activation
                as described in SimCLR paper: https://arxiv.org/pdf/2002.05709.pdf
        '''
        super(projection_MLP, self).__init__()

        if '18' in model or '34' in model:
            n_channels = 512
        elif '50' in model or '101' in model or '152' in model:
            n_channels = 2048
        elif model == 'alexnet'or model == 'mhe_alexnet':
            n_channels = 192 * 4 * 4  # For CIFAR10 (3,32,32)
        else:
            raise NotImplementedError('model not supported: {}'.format(model))

        self.projection_head = nn.Sequential()

        self.projection_head.add_module('w1', fc_layer(
            n_channels, hidden_dim, norm_layer=norm_layer))
        self.projection_head.add_module('bn1', norm_l(norm_layer, hidden_dim))
        self.projection_head.add_module('ReLU', nn.ReLU(inplace=True))
        self.projection_head.add_module('w2', fc_layer(
            hidden_dim, output_dim, bias=False, norm_layer=norm_layer))
        # self.projection_head.add_module('bn2', nn.BatchNorm1d(output_dim))

        # self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.01)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x):
        return self.projection_head(x)


class projection_pred(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, hidden_dim=4096, norm_layer=None):
        '''Projection head for the pretraining of the resnet encoder.

            - Uses the dataset and model size to determine encoder output
                representation dimension.
            - Outputs to a dimension of 128, and uses non-linear activation
                as described in SimCLR paper: https://arxiv.org/pdf/2002.05709.pdf
        '''
        super(projection_pred, self).__init__()

        self.pred_head = nn.Sequential()

        self.pred_head.add_module('w1', fc_layer(
            input_dim, hidden_dim, norm_layer=norm_layer))
        self.pred_head.add_module('bn1', norm_l(norm_layer, hidden_dim))
        self.pred_head.add_module('ReLU', nn.ReLU(inplace=True))
        self.pred_head.add_module('w2', fc_layer(
            hidden_dim, output_dim, bias=False, norm_layer=norm_layer))
        # self.projection_head.add_module('bn2', nn.BatchNorm1d(output_dim))

        # self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.01)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x):
        return self.pred_head(x)


class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)
