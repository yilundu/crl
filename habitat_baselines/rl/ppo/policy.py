#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
import habitat_baselines.rl.models.resnet as resnet
from habitat_baselines.rl.models.resnet import ResNetEncoder

GOAL_EMBEDDING_SIZE = 32


class TorchVisionResNet50(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self, pretrained=True, spatial_output: bool = False
    ):
        super().__init__()
        self.device = torch.device('cuda')
        self.resnet_layer_size = 2048
        linear_layer_input_size = 0

        self.cnn = models.resnet50(pretrained=pretrained)
        self.layer_extract = self.cnn._modules.get("avgpool")


    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        def resnet_forward(observation):
            resnet_output = torch.zeros(1, dtype=torch.float32, device=self.device)

            def hook(m, i, o):
                resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            h = self.layer_extract.register_forward_hook(hook)
            self.cnn(observation)
            h.remove()
            return resnet_output

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
        rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
        rgb_observations = rgb_observations / 255.0  # normalize RGB
        resnet_output = resnet_forward(rgb_observations.contiguous())

        return resnet_output


class Policy(nn.Module):
    def __init__(self, net, dim_actions, **kwargs):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs
    ):
        features, policy_features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(policy_features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, policy_features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, policy_features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(policy_features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, features, None, None, None


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class BaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid=None,
        hidden_size=512,
        detach=False,
        imagenet=False,
        **kwargs,
    ):
        super().__init__(
            BaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                detach=detach,
                imagenet=imagenet,
            ),
            action_space.n,
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

class BaselineNet(Net):
    r"""Network which passes the input image through CNN and passes through RNN.
    """

    def __init__(
        self,
        observation_space,
        hidden_size,
        goal_sensor_uuid=None,
        detach=False,
        imagenet=False,
        additional_sensors=[] # low dim sensors corresponding to registered name
    ):
        self.detach = detach
        self.imagenet = imagenet
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self.additional_sensors = additional_sensors
        self._n_input_goal = 0
        self._n_input_goal = 0
        if goal_sensor_uuid is not None and goal_sensor_uuid != "no_sensor":
            self.goal_sensor_uuid = goal_sensor_uuid
            self._initialize_goal_encoder(observation_space)
        self._hidden_size = hidden_size

        resnet_baseplanes = 64
        backbone="resnet50"
        # backbone="resnet18"

        if imagenet:
            visual_resnet = TorchVisionResNet50()
            visual_resnet.eval()
        else:
            visual_resnet = ResNetEncoder(
                observation_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=False,
            )

        self.detach = detach

        self.model_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=False,
            dense=True,
        )

        self.target_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=False,
            dense=True,
        )

        self.visual_resnet = visual_resnet

        if imagenet:
            self.visual_encoder = nn.Sequential(
                Flatten(),
                nn.Linear(
                    2048, hidden_size
                ),
                nn.ReLU(True),
            )

            self.target_image_encoder = nn.Sequential(
                Flatten(),
                nn.Linear(
                    2048, hidden_size
                ),
                nn.ReLU(True),
            )
        else:
            self.visual_encoder = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(visual_resnet.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

            self.target_image_encoder = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(visual_resnet.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

        final_embedding_size = (0 if self.is_blind else self._hidden_size) + self._n_input_goal
        for sensor in additional_sensors:
            final_embedding_size += observation_space.spaces[sensor].shape[0]

        if self.goal_sensor_uuid == 'imagegoal':
            final_embedding_size = 1024

        self.state_encoder = nn.Sequential(
            nn.Linear(
                final_embedding_size, hidden_size
            ),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size))
        self.state_policy_encoder = RNNStateEncoder(final_embedding_size, self._hidden_size)
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

    def _initialize_goal_encoder(self, observation_space):
        try:
            self._n_input_goal = observation_space.spaces[
                self.goal_sensor_uuid
            ].shape[0]
        except:
            self.goal_sensor_uuid = 'imagegoal'
            self._n_input_goal = observation_space.spaces[
                self.goal_sensor_uuid
            ].shape[0]

    def get_target_encoding(self, observations):
        if self.goal_sensor_uuid == 'imagegoal':
            embed = self.visual_resnet(observations)

            if self.detach:
                embed = embed.detach()

            perception_embed = self.target_image_encoder(embed)

            return perception_embed
        else:
            return observations[self.goal_sensor_uuid]

    def _append_additional_sensors(self, x, observations):
        for sensor in self.additional_sensors:
            x.append(observations[sensor])
        return x

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        policy_x = []
        if not self.is_blind:
            embed = self.visual_resnet(observations)
            # policy_embed = self.visual_policy_resnet(observations)
            if self.detach:
                # policy_embed = policy_embed.detach()
                embed = embed.detach()

            # if self.detach:
            #     embed = embed.detach()

            perception_embed = self.visual_encoder(embed)
            # perception_policy_embed = self.visual_policy_encoder(policy_embed)
            x.append(perception_embed)
            # policy_x.append(perception_policy_embed)
            x.append(self.get_target_encoding(observations))
            # policy_x.append(self.get_target_encoding(observations))

        x = self._append_additional_sensors(x, observations)
        # policy_x = self._append_additional_sensors(policy_x, observations)

        x = torch.cat(x, dim=-1) # t x n x -1
        # policy_x = torch.cat(policy_x, dim=-1) # t x n x -1

        # x = self.state_encoder(x)
        # policy_x = x
        x, rnn_hidden_states = self.state_policy_encoder(x, rnn_hidden_states, masks)
        policy_x = x
        return x, policy_x, rnn_hidden_states
