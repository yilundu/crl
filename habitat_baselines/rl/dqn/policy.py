#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
import habitat_baselines.rl.models.resnet as resnet
from habitat_baselines.rl.models.resnet import ResNetEncoder
import torch.nn.functional as F

from habitat_baselines.common.utils import (
    ResizeCenterCropper,
)

GOAL_EMBEDDING_SIZE = 32

class Policy(nn.Module):
    def __init__(self, net, dim_actions, num_heads, **kwargs):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions
        self.num_heads = num_heads

        self.linear = nn.Linear(self.net.output_size, self.dim_actions * num_heads)

    def forward(self, observations, rnn_hidden_states, masks):
        features = self.net(
            observations, rnn_hidden_states, masks
        )
        return features
        q_vals = self.linear(features)
        s = q_vals.size()

        random_weights = torch.rand(*s[:-1], self.num_heads, 1).to(q_vals.device)
        random_weights = F.normalize(random_weights, p=1, dim=-2)
        q_vals = q_vals.view(*s[:-1], self.num_heads, self.dim_actions)
        q_vals = (q_vals * random_weights).sum(dim=-2)

        return q_vals, rnn_hidden_states


class QNetwork(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid=None,
        hidden_size=512,
        num_heads=1,
        detach=False,
        **kwargs,
    ):
        super().__init__(
            BaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
            ),
            action_space.n,
            num_heads
        )

class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass

# class BaselineNet(Net):
#     r"""Network which passes the input image through CNN and passes through RNN.
#     """
# 
#     def __init__(
#         self,
#         observation_space,
#         hidden_size,
#         goal_sensor_uuid=None,
#         detach=False,
#         additional_sensors=[] # low dim sensors corresponding to registered name
#     ):
#         super().__init__()
#         self.goal_sensor_uuid = goal_sensor_uuid
#         self.additional_sensors = additional_sensors
#         self._n_input_goal = 0
#         self._n_input_goal = 0
#         if goal_sensor_uuid is not None and goal_sensor_uuid != "no_sensor":
#             self.goal_sensor_uuid = goal_sensor_uuid
#             self._initialize_goal_encoder(observation_space)
#         self._hidden_size = hidden_size
# 
#         resnet_baseplanes = 32
#         backbone="resnet18"
#         visual_resnet = ResNetEncoder(
#             observation_space,
#             baseplanes=resnet_baseplanes,
#             ngroups=resnet_baseplanes // 2,
#             make_backbone=getattr(resnet, backbone),
#             normalize_visual_inputs=False,
#         )
# 
#         self.detach = detach
#         self.visual_resnet = visual_resnet
#         self.visual_encoder = nn.Sequential(
#             Flatten(),
#             nn.Linear(
#                 np.prod(visual_resnet.output_shape), hidden_size
#             ),
#             nn.ReLU(True),
#         )
# 
#         final_embedding_size = (0 if self.is_blind else self._hidden_size) + self._n_input_goal
#         for sensor in additional_sensors:
#             final_embedding_size += observation_space.spaces[sensor].shape[0]
# 
#         self.state_encoder = RNNStateEncoder(final_embedding_size, self._hidden_size)
#         self.train()
# 
#     @property
#     def output_size(self):
#         return self._hidden_size
# 
#     @property
#     def is_blind(self):
#         return False
# 
#     @property
#     def num_recurrent_layers(self):
#         return self.state_encoder.num_recurrent_layers
# 
#     def _initialize_goal_encoder(self, observation_space):
#         self._n_input_goal = observation_space.spaces[
#             self.goal_sensor_uuid
#         ].shape[0]
# 
#     def get_target_encoding(self, observations):
#         return observations[self.goal_sensor_uuid]
# 
#     def _append_additional_sensors(self, x, observations):
#         for sensor in self.additional_sensors:
#             x.append(observations[sensor])
#         return x
# 
#     def forward(self, observations, rnn_hidden_states, masks):
#         x = []
#         if not self.is_blind:
#             embed = self.visual_resnet(observations)
# 
#             if self.detach:
#                 embed = embed.detach()
# 
#             perception_embed = self.visual_encoder(embed)
#             x.append(perception_embed)
#         if self.goal_sensor_uuid is not None:
#             x.append(self.get_target_encoding(observations))
# 
#         x = self._append_additional_sensors(x, observations)
# 
#         x = torch.cat(x, dim=-1) # t x n x -1
# 
#         x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
#         return x, rnn_hidden_states


class BaselineNet(Net):
    r"""Network which passes the input image through CNN and passes through RNN.
    """

    def __init__(
        self,
        observation_space,
        hidden_size,
        goal_sensor_uuid=None,
        detach=False,
        additional_sensors=[] # low dim sensors corresponding to registered name
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self.additional_sensors = additional_sensors
        self._n_input_goal = 0
        self._n_input_goal = 0
        # if goal_sensor_uuid is not None and goal_sensor_uuid != "no_sensor":
        #     self.goal_sensor_uuid = goal_sensor_uuid
        #     self._initialize_goal_encoder(observation_space)
        self._hidden_size = hidden_size

        resnet_baseplanes = 32
        backbone="resnet18"
        visual_resnet = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=False,
            obs_transform=ResizeCenterCropper(size=(256, 256)),
            backbone_only=True,
        )

        self.detach = detach
        self.visual_resnet = visual_resnet
        self.visual_encoder = nn.Sequential(
            Flatten(),
            nn.Linear(
                np.prod(visual_resnet.output_shape), hidden_size
            ),
            nn.Sigmoid()
        )

        self.visual_decoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

        return x

    def forward(self, observations, rnn_hidden_states, masks):
        x = []
        embed = self.visual_resnet(observations)
        embed = self.visual_encoder(embed)
        embed_noise = embed + (torch.rand_like(embed) - 0.5) * 0.25
        embed_noise = embed_noise.view(-1, 32, 4, 4)
        rgb = self.visual_decoder(embed_noise).permute(0, 2, 3, 1).contiguous()

        return rgb


    def forward_latent(self, observations, rnn_hidden_states, masks):
        x = []
        embed = self.visual_resnet(observations)
        embed = self.visual_encoder(embed)

        return embed
