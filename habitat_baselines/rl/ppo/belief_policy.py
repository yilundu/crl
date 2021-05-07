#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from habitat_baselines.common.utils import Flatten
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
import habitat_baselines.rl.models.resnet as resnet
from habitat_baselines.rl.ppo.policy import Policy, Net, GOAL_EMBEDDING_SIZE

import visualpriors

class SingleBelief(Net):
    r"""
        Stripped down single recurrent belief.
        Compared to the baseline, the visual encoder has been removed.
    """

    def __init__(
        self,
        observation_space,
        hidden_size,
        goal_sensor_uuid=None,
        additional_sensors=[], # low dim sensors to merge in input
        embed_goal=False,
        device=None,
        **kwargs,
    ):
        super().__init__()

        self.goal_sensor_uuid = goal_sensor_uuid
        self.additional_sensors = additional_sensors
        self.embed_goal = embed_goal
        self.device = device
        self._n_input_goal = 0
        if goal_sensor_uuid is not None and goal_sensor_uuid != "no_sensor":
            self.goal_sensor_uuid = goal_sensor_uuid
            self._initialize_goal_encoder(observation_space)

        self._hidden_size = hidden_size
        embedding_size = (0 if self.is_blind else self._hidden_size) + self._n_input_goal
        for sensor in self.additional_sensors:
            embedding_size += observation_space.spaces[sensor].shape[0]
        self._embedding_size = embedding_size

        self._initialize_state_encoder()

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

        if not self.embed_goal:
            self._n_input_goal = observation_space.spaces[
                self.goal_sensor_uuid
            ].shape[0]
            return

        self._n_input_goal = GOAL_EMBEDDING_SIZE
        goal_space = observation_space.spaces[
            self.goal_sensor_uuid
        ]
        self.goal_embedder = nn.Embedding(goal_space.high[0] - goal_space.low[0] + 1, self._n_input_goal)

    def _initialize_state_encoder(self):
        self.state_encoder = RNNStateEncoder(self._embedding_size, self._hidden_size)

    def get_target_encoding(self, observations):
        goal = observations[self.goal_sensor_uuid]
        if self.embed_goal:
            return self.goal_embedder(goal.long()).squeeze(-2)
        return goal

    def _get_observation_embedding(self, visual_embedding, observations):
        embedding = [visual_embedding]
        if self.goal_sensor_uuid is not None:
            embedding.append(self.get_target_encoding(observations))
        for sensor in self.additional_sensors:
            embedding.append(observations[sensor])
        return torch.cat(embedding, dim=-1)

    def forward(self, visual_embedding, observations, rnn_hidden_states, prev_actions, masks):
        x = self._get_observation_embedding(visual_embedding, observations)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states

class BeliefPolicy(Policy):
    r"""
        Base class for policy that will interact with auxiliary tasks.
        Provides a visual encoder, requires a recurrent net.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        net=SingleBelief,
        aux_tasks=[], # bruh are we even forwarding these things...
        config=None,
        **kwargs, # Note, we forward kwargs to the net
    ):
        assert issubclass(net, SingleBelief), "Belief policy must use belief net"
        super().__init__(net(
            observation_space=observation_space,
            hidden_size=hidden_size,
            config=config, # Forward
            **kwargs,
        ), action_space.n)
        self.aux_tasks = aux_tasks

        resnet_baseplanes = 32
        backbone="resnet18"

        visual_resnet = resnet.ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=config.use_mean_and_var
        )

        self.visual_encoder = nn.Sequential(
            visual_resnet,
            Flatten(),
            nn.Linear(
                np.prod(visual_resnet.output_shape), hidden_size
            ),
            nn.ReLU(True),
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs
    ):
        visual_embedding = self.visual_encoder(observations)
        features, rnn_hidden_states = self.net(
            visual_embedding, observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        visual_embedding = self.visual_encoder(observations)
        features, *_ = self.net(
            visual_embedding, observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        visual_embedding = self.visual_encoder(observations)
        features, rnn_hidden_states = self.net(
            visual_embedding, observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()
        # Nones: individual_features, aux entropy, aux weights
        return value, action_log_probs, distribution_entropy, rnn_hidden_states, features, None, None, None

    def shape_aux_inputs(self, sample, final_rnn_state):
        observations = sample[0]
        n = final_rnn_state.size(1)
        masks = sample[6].view(-1, n)
        env_zeros = [] # Episode crossings per env, lots of tasks use this
        for env in range(n):
            env_zeros.append(
                (masks[:, env] == 0.0).nonzero().squeeze(-1).cpu().unbind(0)
            )
        t = masks.size(0)
        actions = sample[2].view(t, n)
        vision_embedding = self.visual_encoder(observations).view(t, n, -1)

        return observations, actions, vision_embedding, n, t, env_zeros

    def evaluate_aux_losses(self, sample, final_rnn_state, rnn_features, *args):
        if len(self.aux_tasks) == 0:
            pass
        observations, actions, vision_embedding, n, t, env_zeros = self.shape_aux_inputs(sample, final_rnn_state)
        belief_features = rnn_features.view(t, n, -1)
        final_belief_state = final_rnn_state[-1] # only use final layer
        return [task.get_loss(observations, actions, vision_embedding, final_belief_state, belief_features, n, t, env_zeros) for task in self.aux_tasks]

class MidLevelPolicy(BeliefPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        net=SingleBelief,
        aux_tasks=[],
        config=None,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net,
            aux_tasks=aux_tasks,
            config=config,
            **kwargs
        )
        self.medium = config.midlevel_medium
        self.visual_encoder = None
        self.visual_resize = nn.Sequential(
            Flatten(),
            nn.Linear(2048, hidden_size),
            nn.ReLU(True)
        )

    def visual_encoder(self, observations):
        # Use detached mid-level transform
        rgb_obs = observations["rgb"]
        rgb_obs = rgb_obs / 255.0 # normalize
        # Rescale to -1, 1.
        o_t = rgb_obs * 2 - 1 # No unsqueezing needed, we're already batched
        o_t = o_t.permute(0, 3, 1, 2) # channels first
        representation = visualpriors.representation_transform(o_t, self.medium)
        return self.visual_resize(representation)

class MultipleBeliefNet(SingleBelief):
    r"""
        Uses multiple belief RNNs. Requires num_tasks, and fusion workings.
    """
    def __init__(
        self,
        observation_space,
        hidden_size,
        num_tasks,
        **kwargs,
    ):
        self.num_tasks = num_tasks # We declare this first so state encoders can be initialized

        super().__init__(observation_space, hidden_size, **kwargs)
        self._initialize_fusion_net()

    @property
    def num_recurrent_layers(self):
        return self.state_encoders[0].num_recurrent_layers

    def _initialize_state_encoder(self):
        self.state_encoders = nn.ModuleList([
            RNNStateEncoder(self._embedding_size, self._hidden_size) for _ in range(self.num_tasks)
        ])

    def _initialize_fusion_net(self):
        pass # Do nothing as a default

    @abc.abstractmethod
    def _fuse_beliefs(self, beliefs, x, *args):
        pass

    def forward(self, visual_embedding, observations, rnn_hidden_states, prev_actions, masks):
        x = self._get_observation_embedding(visual_embedding, observations)
        # rnn_hidden_states.size(): num_layers, num_envs, num_tasks, hidden, (only first timestep)
        outputs = [encoder(x, rnn_hidden_states[:, :, i], masks) for i, encoder in enumerate(self.state_encoders)]
        embeddings, rnn_hidden_states = zip(*outputs) # (txn)xh, (layers)xnxh
        rnn_hidden_states = torch.stack(rnn_hidden_states, dim=-2) # (layers) x n x k x h
        beliefs = torch.stack(embeddings, dim=-2) # (t x n) x k x h

        contextual_embedding, weights = self._fuse_beliefs(beliefs, x)
        return contextual_embedding, rnn_hidden_states, beliefs, weights

class MultipleBeliefPolicy(BeliefPolicy):
    r""" Base policy for multiple beliefs adding basic checks and weight diagnostics
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        net=None,
        aux_tasks=[],
        **kwargs,
    ):
        # 0 tasks allowed for eval
        assert len(aux_tasks) != 1, "Multiple beliefs requires more than one auxiliary task"
        assert issubclass(net, MultipleBeliefNet), "Multiple belief policy requires compatible multiple belief net"
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net,
            aux_tasks=aux_tasks,
            **kwargs,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        weights_output=None,
    ):
        visual_embedding = self.visual_encoder(observations)
        features, rnn_hidden_states, _, weights = self.net(
            visual_embedding, observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)

        if weights_output is not None:
            weights_output.copy_(weights)
        return value, action, action_log_probs, rnn_hidden_states


    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        visual_embedding = self.visual_encoder(observations)
        # sequence forwarding
        features, rnn_hidden_states, individual_features, weights = self.net(
            visual_embedding, observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        aux_dist_entropy = Categorical(weights).entropy().mean()
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        weights = weights.mean(dim=0)

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, features, individual_features, aux_dist_entropy, weights

    def evaluate_aux_losses(self, sample, final_rnn_state, rnn_features, individual_rnn_features):
        observations, actions, vision_embedding, n, t, env_zeros = self.shape_aux_inputs(sample, final_rnn_state)
        return [task.get_loss(observations, actions, vision_embedding, final_rnn_state[-1, :, i].contiguous(), individual_rnn_features[:, i].contiguous().view(t,n,-1), n, t, env_zeros) \
            for i, task in enumerate(self.aux_tasks)]

class AttentiveBelief(MultipleBeliefNet):
    def _initialize_fusion_net(self):
        # self.visual_key_net = nn.Linear(
        self.key_net = nn.Linear(
            self._embedding_size, self._hidden_size
        )

    def _fuse_beliefs(self, beliefs, x, *args):
        key = self.key_net(x.unsqueeze(-2))
        # key = self.visual_key_net(x.unsqueeze(-2)) # (t x n) x 1 x h
        scores = torch.bmm(beliefs, key.transpose(1, 2)) / math.sqrt(self.num_tasks) # scaled dot product
        weights = F.softmax(scores, dim=1).squeeze(-1) # n x k (logits) x 1 -> (txn) x k

        # # ! OVERRIDE WEIGHTS FOR MASKING
        # dict_to_index = {
        #     "cpca1": 0,
        #     "cpca2": 1,
        #     "cpca4": 2,
        #     "cpca8": 3,
        #     "cpca16":4,
        #     "id": 5,
        #     "td": 6,
        # }
        # mode = "BLACKLIST"
        # mode = "WHITELIST"
        # marked = []
        # marked = ["cpca1"]
        # # marked = ["cpca2"]
        # # marked = ["cpca4"]
        # # marked = ["cpca8"]
        # # marked = ["cpca16"]
        # # marked = ["id"]
        # # marked = ["td"]
        # # marked = ["cpca1", "cpca8"]

        # zeroed_inds = [dict_to_index[mark] for mark in marked]
        # if mode != "BLACKLIST":
        #     zeroed_inds = list(set(list(range(5))) - set(zeroed_inds))
        # weights[:, zeroed_inds] = 0.0
        # # ! Renormalize
        # weights = weights / weights.norm(dim=1).unsqueeze(1) # n x k / n x 1

        # n x 1 x k x n x k x h
        contextual_embedding = torch.bmm(weights.unsqueeze(1), beliefs).squeeze(1) # txn x h
        return contextual_embedding, weights

class FixedAttentionBelief(MultipleBeliefNet):
    r""" Fixed Attn Baseline for comparison w/ naturally arising peaky distribution
    """
    def _fuse_beliefs(self, beliefs, x, *args):
        txn = x.size()[0]
        weights = torch.zeros(txn, self.num_tasks, dtype=torch.float32, device=beliefs.device)
        weights[:, 0] = 1.0 # all attn on first task
        contextual_embedding = torch.bmm(weights.unsqueeze(1), beliefs).squeeze(1) # txn x h
        return contextual_embedding, weights

class SoftmaxBelief(MultipleBeliefNet):
    r""" Softmax Gating Baseline for comparison w/ regular attention
    """
    def _initialize_fusion_net(self):
        self.softmax_net = nn.Linear(
            self._embedding_size, self.num_tasks
        )

    def _fuse_beliefs(self, beliefs, x, *args):
        scores = self.softmax_net(x) # (t x n) x k
        weights = F.softmax(scores, dim=-1).squeeze(-1) # (txn) x k
        contextual_embedding = torch.bmm(weights.unsqueeze(1), beliefs).squeeze(1) # txn x h
        return contextual_embedding, weights

class AverageBelief(MultipleBeliefNet):
    def _fuse_beliefs(self, beliefs, x, *args):
        txn = x.size(0)
        weights = torch.ones(txn, self.num_tasks, dtype=torch.float32, device=beliefs.device)
        weights /= self.num_tasks
        contextual_embedding = torch.bmm(weights.unsqueeze(1), beliefs).squeeze(1) # txn x h
        return contextual_embedding, weights

# A bunch of wrapper classes attaching a belief net to the multiple belief policy
class AttentiveBeliefPolicy(MultipleBeliefPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        net=AttentiveBelief,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net=net,
            **kwargs
        )

class FixedAttentionBeliefPolicy(MultipleBeliefPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net=FixedAttentionBelief,
            **kwargs
        )

class SoftmaxBeliefPolicy(MultipleBeliefPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net=SoftmaxBelief,
            **kwargs
        )

class AverageBeliefPolicy(MultipleBeliefPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net=AverageBelief,
            **kwargs
        )
