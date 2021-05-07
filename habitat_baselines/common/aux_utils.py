#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import torch
import torch.nn as nn

ACTION_EMBEDDING_DIM = 4

def subsampled_mean(x, p=0.1):
    return torch.masked_select(x, torch.rand_like(x) < p).mean()

class RolloutAuxTask(nn.Module):
    r""" Rollout-based self-supervised auxiliary task base class.
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.aux_cfg = aux_cfg
        self.task_cfg = task_cfg # Mainly tracked for actions
        self.device = device

    def forward(self, *x):
        raise NotImplementedError

    @abc.abstractmethod
    def get_loss(self, observations, actions, vision, final_belief_state, belief_features, n, t, env_zeros):
        pass

