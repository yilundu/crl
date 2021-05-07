#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.policy import Policy

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        aux_loss_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
        aux_tasks=[],
        aux_cfg=None,
        resume_detach=False,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.aux_loss_coef = aux_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage
        self.aux_tasks = aux_tasks
        if aux_cfg:
            self.aux_cfg = aux_cfg
        params = list(actor_critic.parameters())

        if resume_detach:
            params = list(actor_critic.action_distribution.parameters()) + list(actor_critic.critic.parameters())
            params = params + list(actor_critic.net.visual_encoder.parameters()) + list(actor_critic.net.state_encoder.parameters()) + list(actor_critic.net.state_policy_encoder.parameters())

        for task in aux_tasks:
            params += list(task.parameters())
        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, params)),
            lr=lr,
            eps=eps,
        )

        self.unsup_optimizer = optim.Adam(actor_critic.net.model_encoder.parameters(), lr=1e-5, eps=eps)
        self.model_optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, params)),
            lr=lr,
            eps=eps,
        )

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        aux_losses_epoch = [0] * len(self.aux_tasks)
        aux_entropy_epoch = 0
        aux_weights_epoch = [0] * len(self.aux_tasks)
        for e in range(self.ppo_epoch):
            # This data generator steps through the rollout (gathering n=batch_size processes rollouts)
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch,
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    final_rnn_state, # Used to encourage trajectory memory (it's the same as final rnn feature due to GRU)
                    rnn_features,
                    individual_rnn_features,
                    aux_dist_entropy,
                    aux_weights
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                # Value loss is MSE with actual(TM) value/rewards
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                total_aux_loss = 0
                aux_losses = []
                if len(self.aux_tasks) > 0: # Only nonempty in training
                    raw_losses = self.actor_critic.evaluate_aux_losses(sample, final_rnn_state, rnn_features, individual_rnn_features)
                    aux_losses = torch.stack(raw_losses)
                    total_aux_loss = torch.sum(aux_losses, dim=0)
                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    + total_aux_loss * self.aux_loss_coef
                    - dist_entropy * self.entropy_coef
                )
                if aux_dist_entropy is not None:
                    total_loss -= aux_dist_entropy * self.aux_cfg.entropy_coef

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                if aux_dist_entropy is not None:
                    aux_entropy_epoch += aux_dist_entropy.item()
                for i, aux_loss in enumerate(aux_losses):
                    aux_losses_epoch[i] += aux_loss.item()
                if aux_weights is not None:
                    for i, aux_weight in enumerate(aux_weights):
                        aux_weights_epoch[i] += aux_weight.item()

            profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        for i, aux_loss in enumerate(aux_losses):
            aux_losses_epoch[i] /= num_updates
        if aux_weights is not None:
            for i, aux_weight in enumerate(aux_weights):
                aux_weights_epoch[i] /= num_updates
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, aux_losses_epoch, aux_entropy_epoch, aux_weights_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        params = list(self.actor_critic.parameters())
        for task in self.aux_tasks:
            params += list(task.parameters())
        nn.utils.clip_grad_norm_(
            params, self.max_grad_norm
        )

    def after_step(self):
        pass
