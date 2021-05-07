#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.utils import (
    ResizeCenterCropper,
)


class WSLinear(nn.Linear):

    def __init__(self, in_channels, out_channels):
        super(WSLinear, self).__init__(in_channels, out_channels)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1, keepdim=True) + 1e-3
        weight = weight / std.expand_as(weight)
        return F.linear(x, weight, self.bias)

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        groups=groups,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1
    resneXt = False

    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        downsample=None,
        cardinality=1,
    ):
        super(BasicBlock, self).__init__()
        self.convs = nn.Sequential(
            conv3x3(inplanes, planes, stride, groups=cardinality),
            nn.GroupNorm(ngroups, planes),
            nn.ReLU(True),
            conv3x3(planes, planes, groups=cardinality),
            nn.GroupNorm(ngroups, planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = x

        out = self.convs(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


def _build_bottleneck_branch(
    inplanes, planes, ngroups, stride, expansion, groups=1
):
    return nn.Sequential(
        conv1x1(inplanes, planes),
        nn.GroupNorm(ngroups, planes),
        nn.ReLU(True),
        conv3x3(planes, planes, stride, groups=groups),
        nn.GroupNorm(ngroups, planes),
        nn.ReLU(True),
        conv1x1(planes, planes * expansion),
        nn.GroupNorm(ngroups, planes * expansion),
    )


class SE(nn.Module):
    def __init__(self, planes, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(planes, int(planes / r)),
            nn.ReLU(True),
            nn.Linear(int(planes / r), planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.squeeze(x)
        x = x.view(b, c)
        x = self.excite(x)

        return x.view(b, c, 1, 1)


def _build_se_branch(planes, r=16):
    return SE(planes, r)


class Bottleneck(nn.Module):
    expansion = 4
    resneXt = False

    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        downsample=None,
        cardinality=1,
    ):
        super().__init__()
        self.convs = _build_bottleneck_branch(
            inplanes,
            planes,
            ngroups,
            stride,
            self.expansion,
            groups=cardinality,
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def _impl(self, x):
        identity = x

        out = self.convs(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)

    def forward(self, x):
        return self._impl(x)


class SEBottleneck(Bottleneck):
    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        downsample=None,
        cardinality=1,
    ):
        super().__init__(
            inplanes, planes, ngroups, stride, downsample, cardinality
        )

        self.se = _build_se_branch(planes * self.expansion)

    def _impl(self, x):
        identity = x

        out = self.convs(x)
        out = self.se(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class SEResNeXtBottleneck(SEBottleneck):
    expansion = 2
    resneXt = True


class ResNeXtBottleneck(Bottleneck):
    expansion = 2
    resneXt = True


class TwoBranchShakeBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ngroups, stride=1, downsample=None):
        super().__init__()
        self.b1 = _build_bottleneck_branch(
            inplanes, planes, ngroups, stride, self.expansion
        )
        self.b2 = _build_bottleneck_branch(
            inplanes, planes, ngroups, stride, self.expansion
        )
        self.shake_shake = ShakeShake()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        b1 = self.b1(x)
        b2 = self.b2(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(self.shake_shake(identity, b1, b2))


class ResNet(nn.Module):
    def __init__(
        self, in_channels, base_planes, ngroups, block, layers, cardinality=1
    ):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(ngroups, base_planes),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cardinality = cardinality

        self.inplanes = base_planes
        if block.resneXt:
            base_planes *= 2

        self.layer1 = self._make_layer(block, ngroups, base_planes, layers[0])
        self.layer2 = self._make_layer(
            block, ngroups, base_planes * 2, layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, ngroups, base_planes * 2 * 2, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, ngroups, base_planes * 2 * 2 * 2, layers[3], stride=2
        )

        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2 ** 5)

    def _make_layer(self, block, ngroups, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(ngroups, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                ngroups,
                stride,
                downsample,
                cardinality=self.cardinality,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ngroups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(in_channels, base_planes, ngroups):
    model = ResNet(in_channels, base_planes, ngroups, BasicBlock, [2, 2, 2, 2])

    return model


def resnet50(in_channels, base_planes, ngroups):
    model = ResNet(in_channels, base_planes, ngroups, Bottleneck, [3, 4, 6, 3])

    return model


def resneXt50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        ResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_resnet50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels, base_planes, ngroups, SEBottleneck, [3, 4, 6, 3]
    )

    return model


def se_resneXt50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_resneXt101(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 23, 3],
        cardinality=int(base_planes / 2),
    )

    return model

SEMANTIC_EMBEDDING_SIZE = 4
class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        make_backbone=None,
        use_if_available=["rgb", "depth"],
        normalize_visual_inputs=True,
        # obs_transform=ResizeCenterCropper(size=(256, 256)),
        obs_transform=None,
        backbone_only=False,
        dense=False,
    ):
        super().__init__()

        self.obs_transform = obs_transform
        if self.obs_transform is not None:
            observation_space = self.obs_transform.transform_observation_space(
                observation_space
            )

        self.backbone_only = backbone_only
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._inputs = list(filter(lambda x: x in observation_space.spaces, use_if_available))
        self._input_sizes = [0] * len(self._inputs)
        for i, mode in enumerate(self._inputs):
            self._input_sizes[i] = observation_space.spaces[mode].shape[2]

        self.running_mean_and_var = nn.Sequential()
        self.relu = nn.ReLU(True)
        self.dense = dense

        # self.pred_fc1 = WSLinear(256, 256)
        # self.pred_fc1_bn = nn.GroupNorm(32, 256, affine=False)
        # self.pred_fc2 = WSLinear(256, 256)
        # self.pred_fc2_bn = nn.GroupNorm(32, 256, affine=False)
        # self.pred_fc3 = WSLinear(256, 256)
        # self.pred_fc3_bn = nn.GroupNorm(32, 256, affine=False)

        # self.map_fc1 = WSLinear(256, 256)
        # self.map_fc1_bn = nn.GroupNorm(32, 256, affine=False)
        # self.map_fc2 = WSLinear(256, 256)

        if not self.is_blind:
            if self.backbone_only:
                spatial_size = observation_space.spaces[self._inputs[0]].shape[0] // 2
            else:
                spatial_size = observation_space.spaces[self._inputs[0]].shape[0]

            input_channels = sum(self._input_sizes) # self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial = int(
                spatial_size * self.backbone.final_spatial_compress
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial ** 2))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )

        if not backbone_only:
            self.fc1 = nn.Linear(2048, 256)
            self.fc2 = nn.Linear(256, 128)

            self.pred_fc1 = nn.Linear(256, 256)
            self.pred_fc1_bn = nn.BatchNorm1d(256, affine=False)
            self.pred_fc2 = nn.Linear(256, 256)
            self.pred_fc2_bn = nn.BatchNorm1d(256, affine=False)
            self.pred_fc3 = nn.Linear(256, 256)
            self.pred_fc3_bn = nn.BatchNorm1d(256, affine=False)

            self.map_fc1 = nn.Linear(256, 256)
            self.map_fc1_bn = nn.BatchNorm1d(256, affine=False)
            self.map_fc2 = nn.Linear(256, 256)


    @property
    def is_blind(self):
        return sum(self._input_sizes) == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):
        if self.is_blind:
            return None

        cnn_input = []
        for mode in self._inputs:
            mode_obs = observations[mode]
            if mode == "rgb":
                mode_obs = mode_obs / 255.0  # normalize RGB
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            mode_obs = mode_obs.permute(0, 3, 1, 2)
            cnn_input.append(mode_obs)

        if self.obs_transform:
            cnn_input = [self.obs_transform(inp) for inp in cnn_input]

        x = torch.cat(cnn_input, dim=1)

        # This means we are using the habitat baselines setting
        if self.backbone_only:
            x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)

        if self.dense:
            return x

        x = self.compression(x)
        return x


    def create_rep(self, x, return_raw=False):
        # Use this branch for BYOL
        if return_raw:
            raw_logits = x

            if self.backbone_only:
                x = F.avg_pool2d(x, 2)

            x = self.running_mean_and_var(x)
            x = self.backbone(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)

            hidden = x

            h2 = self.relu(self.map_fc1_bn(self.map_fc1(hidden)))
            h2 = self.map_fc2(h2)
            proj_output = h2

            h1 = self.relu(self.pred_fc1_bn(self.pred_fc1(proj_output)))
            h1 = self.relu(self.pred_fc2_bn(self.pred_fc2(h1)))
            map_output = self.pred_fc3_bn(self.pred_fc3(h1))
            # map_output = self.pred_fc3_bn(map_output)

            proj_output = F.normalize(proj_output, p=2, dim=-1)
            map_output = F.normalize(map_output, p=2, dim=-1)

            # dot_sim = (proj_output * map_output).sum(dim=-1).mean().item()
            # if dot_sim > 0.95:

            return map_output, proj_output


        else:
            if self.backbone_only:
                x = F.avg_pool2d(x, 2)

            x = self.running_mean_and_var(x)
            x = self.backbone(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = F.normalize(x, p=2, dim=-1)
            return x
