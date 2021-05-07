#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Categorical
from typing import Type

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.aux_utils import ACTION_EMBEDDING_DIM, subsampled_mean, RolloutAuxTask

def get_aux_task_class(aux_task_name: str) -> Type[nn.Module]:
    r"""Return auxiliary task class based on name.

    Args:
        aux_task_name: name of the environment.

    Returns:
        Type[nn.Module]: aux task class.
    """
    return baseline_registry.get_aux_task(aux_task_name)

@baseline_registry.register_aux_task(name="InverseDynamicsTask")
class InverseDynamicsTask(RolloutAuxTask):
    r""" Predict action used between two consecutive frames
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.decoder = nn.Linear(2 * cfg.hidden_size + cfg.hidden_size, num_actions)

        self.classifier = nn.CrossEntropyLoss(reduction='none')

    def get_loss(self, observations, actions, vision, final_belief_state, belief_features, n, t, env_zeros):
        actions = actions[:-1] # t-1 x n
        final_belief_expanded = final_belief_state.expand(t-1, -1, -1) # 1 x n x -1 -> t-1 x n x -1
        decoder_in = torch.cat((vision[:-1], vision[1:], final_belief_expanded), dim=2)
        preds = self.decoder(decoder_in).permute(0, 2, 1) # t-1 x n x 4 -> t-1 x 4 x n
        loss = self.classifier(preds, actions)

        last_zero = torch.zeros(n, dtype=torch.long)
        for env in range(n):
            has_zeros_batch = env_zeros[env]
            if len(has_zeros_batch) == 0:
                has_zeros_batch = [-1]

            last_zero[env] = has_zeros_batch[-1]

        # select the losses coming from valid (same episode) frames
        valid_losses = []
        for env in range(n):
            if last_zero[env] >= t - 3:
                continue
            valid_losses.append(loss[last_zero[env]+1:, env]) # variable length (m,) tensors
        if len(valid_losses) == 0:
            valid_losses = torch.zeros(1, device=self.device, dtype=torch.float)
        else:
            valid_losses = torch.cat(valid_losses) # (sum m, )
        if len(valid_losses) < 10: # don't subsample
            avg_loss = valid_losses.mean()
        else:
            avg_loss = subsampled_mean(valid_losses, self.aux_cfg.subsample_rate)
        return avg_loss * self.aux_cfg.loss_factor

@baseline_registry.register_aux_task(name="TemporalDistanceTask")
class TemporalDistanceAuxTask(RolloutAuxTask):
    r""" Class for calculating timesteps between two randomly selected observations
        Specifically, randomly select `num_pairs` frames per env and predict the frames elapsed
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        self.decoder = nn.Linear(2 * cfg.hidden_size + cfg.hidden_size, 1) # 2 * embedding + belief

    def get_loss(self, observations, actions, vision, final_belief_state, belief_features, n, t, env_zeros):
        k = self.aux_cfg.num_pairs
        final_belief_expanded = final_belief_state.expand(k, -1, -1) # 1 x n x -1 -> t-1 x n x -1

        indices = torch.zeros((2, k, n), device=self.device, dtype=torch.long)
        trial_normalizer = torch.ones(n, device=self.device, dtype=torch.float)
        # find last zero index to find start of current rollout
        for env in range(n):
            has_zeros_batch = env_zeros[env]
            if len(has_zeros_batch) == 0:
                has_zeros_batch = [-0]

            last_index = has_zeros_batch[-1]
            indices[..., env] = torch.randint(last_index, t, (2, self.aux_cfg.num_pairs))
            if last_index >= t - 5: # too close, drop the trial
                trial_normalizer[env] = 0
            else:
                trial_normalizer[env] = 1.0 / float(t - last_index) # doesn't cast without for some reason
        frames = [vision[indices.view(2 * k, n)[:, i], i] for i in range(n)]
        frames = torch.stack(frames, dim=1).view(2, k, n, -1)
        decoder_in = torch.cat((frames[0], frames[1], final_belief_expanded), dim=-1)
        pred_frame_diff = self.decoder(decoder_in).squeeze(-1) # output k x n
        true_frame_diff = (indices[1] - indices[0]).float() # k x n
        pred_error = (pred_frame_diff - true_frame_diff) * trial_normalizer.view(1, n)
        loss = 0.5 * (pred_error).pow(2)
        avg_loss = loss.mean()
        return avg_loss * self.aux_cfg.loss_factor

@baseline_registry.register_aux_task(name="CPCA")
class CPCA(RolloutAuxTask):
    """ Action-conditional CPC - up to k timestep prediction
        From: https://arxiv.org/abs/1811.06407
    """
    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ) # query and perception
        self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)
        self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, cfg.hidden_size)

    def get_loss(self, observations, actions, vision, final_belief_state, belief_features, n, t, env_zeros):
        k = self.aux_cfg.num_steps # up to t
        assert 0 < k <= t, "CPC requires prediction range to be in (0, t]"

        belief_features = belief_features.view(t*n, -1).unsqueeze(0)
        positives = vision
        negative_inds = torch.randperm(t * n, device=self.device)
        negatives = torch.gather(
            positives.view(t * n, -1),
            dim=0,
            index=negative_inds.view(t * n, 1).expand(t * n, positives.size(-1)),
        ).view(t, n, -1)
        action_embedding = self.action_embedder(actions) # t n -1
        action_padding = torch.zeros(k - 1, n, ACTION_EMBEDDING_DIM, device=self.device)
        action_padded = torch.cat((action_embedding, action_padding), dim=0) # (t+k-1) x n x -1
        # t x n x -1 x k
        action_seq = action_padded.unfold(dimension=0, size=k, step=1).permute(3, 0, 1, 2).view(k, t*n, ACTION_EMBEDDING_DIM)

        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_seq, belief_features)
        query_all = out_all.view(k, t, n, -1).permute(1, 0, 2, 3)

        # Targets: predict k steps for each starting timestep
        positives_padded = torch.cat((positives[1:], torch.zeros(k, n, self.cfg.hidden_size, device=self.device)), dim=0) # (t+k) x n
        positives_expanded = positives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        positives_logits = self.classifier(torch.cat([positives_expanded, query_all], -1))
        negatives_padded = torch.cat((negatives[1:], torch.zeros(k, n, self.cfg.hidden_size, device=self.device)), dim=0) # (t+k) x n x -1
        negatives_expanded = negatives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        negatives_logits = self.classifier(torch.cat([negatives_expanded, query_all], -1))

        # Masking
        # Note which timesteps [1, t+k+1] could have valid queries, at distance (k) (note offset by 1)
        valid_modeling_queries = torch.ones(
            t + k, k, n, 1, device=self.device, dtype=torch.bool
        ) # (padded) timestep predicted x prediction distance x env
        valid_modeling_queries[t - 1:] = False # >= t is past rollout, and t is index t - 1 here
        for j in range(1, k + 1): # for j-step predictions
            valid_modeling_queries[:j - 1, j - 1] = False # first j frames cannot be valid for all envs (rollout doesn't go that early)
            for env in range(n):
                has_zeros_batch = env_zeros[env]
                # in j-step prediction, timesteps z -> z + j are disallowed as those are the first j timesteps of a new episode
                # z-> z-1 because of modeling_queries being offset by 1
                for z in has_zeros_batch:
                    valid_modeling_queries[z-1: z-1 + j, j - 1, env] = False

        # instead of the whole range, we actually are only comparing a window i:i+k for each query/target i - for each, select the appropriate k
        # we essentially gather diagonals from this full mask, t of them, k long
        valid_diagonals = [torch.diagonal(valid_modeling_queries, offset=-i) for i in range(t)] # pull the appropriate k per timestep
        valid_mask = torch.stack(valid_diagonals, dim=0).permute(0, 3, 1, 2) # t x n x 1 x k -> t x k x n x 1

        positives_masked_logits = torch.masked_select(positives_logits, valid_mask)
        negatives_masked_logits = torch.masked_select(negatives_logits, valid_mask)
        positive_loss = F.binary_cross_entropy_with_logits(
            positives_masked_logits, torch.ones_like(positives_masked_logits), reduction='none'
        )

        subsampled_positive = subsampled_mean(positive_loss, p=self.aux_cfg.subsample_rate)
        negative_loss = F.binary_cross_entropy_with_logits(
            negatives_masked_logits, torch.zeros_like(negatives_masked_logits), reduction='none'
        )
        subsampled_negative = subsampled_mean(negative_loss, p=self.aux_cfg.subsample_rate)

        aux_losses = subsampled_positive + subsampled_negative
        return aux_losses.mean() * self.aux_cfg.loss_factor

@baseline_registry.register_aux_task(name="CPCA_Weighted")
class CPCA_Weighted(RolloutAuxTask):
    """ To compare with combined aux losses. 5 * k<=1, 4 * k<=2, 3 * k<=4, 2 * k <= 8, 1 * k <= 16 (hardcoded)
        Note - this aux loss is an order of magnitude higher than others (intentionally)
    """
    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ) # query and perception
        self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)
        self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, cfg.hidden_size)

    def get_loss(self, observations, actions, vision, final_belief_state, belief_features, n, t, env_zeros):
        k = 16

        belief_features = belief_features.view(t*n, -1).unsqueeze(0)
        positives = vision
        negative_inds = torch.randperm(t * n, device=self.device)
        negatives = torch.gather(
            positives.view(t * n, -1),
            dim=0,
            index=negative_inds.view(t * n, 1).expand(t * n, positives.size(-1)),
        ).view(t, n, -1)
        action_embedding = self.action_embedder(actions) # t n -1
        action_padding = torch.zeros(k - 1, n, ACTION_EMBEDDING_DIM, device=self.device)
        action_padded = torch.cat((action_embedding, action_padding), dim=0) # (t+k-1) x n x -1
        # t x n x -1 x k
        action_seq = action_padded.unfold(dimension=0, size=k, step=1).permute(3, 0, 1, 2).view(k, t*n, ACTION_EMBEDDING_DIM)

        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_seq, belief_features)
        query_all = out_all.view(k, t, n, -1).permute(1, 0, 2, 3)

        # Targets: predict k steps for each starting timestep
        positives_padded = torch.cat((positives[1:], torch.zeros(k, n, self.cfg.hidden_size, device=self.device)), dim=0) # (t+k) x n
        positives_expanded = positives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        positives_logits = self.classifier(torch.cat([positives_expanded, query_all], -1))
        negatives_padded = torch.cat((negatives[1:], torch.zeros(k, n, self.cfg.hidden_size, device=self.device)), dim=0) # (t+k) x n x -1
        negatives_expanded = negatives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        negatives_logits = self.classifier(torch.cat([negatives_expanded, query_all], -1))

        # Masking
        # Note which timesteps [1, t+k+1] could have valid queries, at distance (k) (note offset by 1)
        valid_modeling_queries = torch.ones(
            t + k, k, n, 1, device=self.device, dtype=torch.bool # not uint so we can mask with this
        ) # (padded) timestep predicted x prediction distance x env
        valid_modeling_queries[t - 1:] = False # >= t is past rollout, and t is index t - 1 here
        for j in range(1, k + 1): # for j-step predictions
            valid_modeling_queries[:j - 1, j - 1] = False # first j frames cannot be valid for all envs (rollout doesn't go that early)
            for env in range(n):
                has_zeros_batch = env_zeros[env]
                # in j-step prediction, timesteps z -> z + j are disallowed as those are the first j timesteps of a new episode
                # z-> z-1 because of modeling_queries being offset by 1
                for z in has_zeros_batch:
                    valid_modeling_queries[z-1: z-1 + j, j - 1, env] = False

        # instead of the whole range, we actually are only comparing a window i:i+k for each query/target i - for each, select the appropriate k
        # we essentially gather diagonals from this full mask, t of them, k long
        valid_diagonals = [torch.diagonal(valid_modeling_queries, offset=-i) for i in range(t)] # pull the appropriate k per timestep
        valid_mask = torch.stack(valid_diagonals, dim=0).permute(0, 3, 1, 2) # t x n x 1 x k -> t x k x n x 1

        weight_mask = torch.tensor([5, 4, 3, 3, 2, 2, 2, 2,
                                    1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32,
                                    device=self.device) # this should be multiplied on the loss
        # mask over the losses, not the logits
        positive_loss = F.binary_cross_entropy_with_logits(
            positives_logits, torch.ones_like(positives_logits), reduction='none'
        ) # t k n 1 still

        positive_loss = positive_loss.permute(0, 2, 3, 1) * weight_mask # now t n 1 k
        positive_loss = torch.masked_select(positive_loss.permute(0, 3, 1, 2), valid_mask) # tkn1 again
        subsampled_positive = subsampled_mean(positive_loss, p=self.aux_cfg.subsample_rate)
        negative_loss = F.binary_cross_entropy_with_logits(
            negatives_logits, torch.zeros_like(negatives_logits), reduction='none'
        )
        negative_loss = negative_loss.permute(0, 2, 3, 1) * weight_mask
        negative_loss = torch.masked_select(negative_loss.permute(0, 3, 1, 2), valid_mask)

        subsampled_negative = subsampled_mean(negative_loss, p=self.aux_cfg.subsample_rate)

        aux_losses = subsampled_positive + subsampled_negative
        return aux_losses.mean() * self.aux_cfg.loss_factor

# Clones of the CPC|A task so that we can allow different task parameters under yacs
# Used to run k=1, 2, 4, 8, 16, in the same exp
@baseline_registry.register_aux_task(name="CPCA_A")
class CPCA_A(CPCA):
    pass

@baseline_registry.register_aux_task(name="CPCA_B")
class CPCA_B(CPCA):
    pass

@baseline_registry.register_aux_task(name="CPCA_C")
class CPCA_C(CPCA):
    pass

@baseline_registry.register_aux_task(name="CPCA_D")
class CPCA_D(CPCA):
    pass