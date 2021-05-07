#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import numpy as np
from imageio import imwrite
from torchvision.utils import make_grid

import torch
import lpips

loss_fn = lpips.LPIPS(net='vgg')

class RolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        num_recurrent_memories=1,
        log_env=False
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        # * Note: these modules are hidden in the rollout, i.e. if an architecture uses one module the rollout API does not expect the module dimension
        # * This is to be compatible with baseline architectures
        self.num_modules = num_recurrent_memories
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            self.num_modules,
            recurrent_hidden_state_size,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)
        self.log_env = log_env

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0
        self.log_step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )

        if self.num_modules == 1:
            recurrent_hidden_states = recurrent_hidden_states.unsqueeze(-2)
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = self.step + 1

    def get_recurrent_states(self):
        if self.num_modules == 1:
            return self.recurrent_hidden_states.squeeze(-2)
        return self.recurrent_hidden_states

    def after_update(self):
        if self.log_env:
            im = self.observations['rgb'].detach().cpu().numpy().astype(np.uint8)
            # select_idx = torch.arange(128)[:100]
            im = im[:128, :, :, :, :]
            # im = im[select_idx, :].transpose((1, 0, 2, 3, 4))

            im = torch.Tensor(im)
            s = im.size()
            im_flat = im.reshape(s[0]*s[1], s[2], s[3], 3)
            rix = torch.randperm(im_flat.size(0))
            im_flat = im_flat[rix]

            dists = []

            for i in range(im_flat.size(0) - 1):
                im_1 = im_flat[i]
                im_2 = im_flat[i+1]
                im_1 = (im_1 / 255 - 0.5) * 2
                im_2 = (im_2 / 255 - 0.5) * 2
                dist = loss_fn(im_1[None].permute(0, 3, 1, 2), im_2[None].permute(0, 3, 1, 2))
                dists.append(dist.item())

            print(self.log_step, np.mean(dists), np.std(dists) / len(dists) ** 0.5)
            # im_flat = im_flat[rix]
            # im_flat = im_flat.permute(0, 3, 1, 2)
            # im_panel = make_grid(im_flat, nrow=25)
            # im_panel = im_panel.permute(1, 2, 0)
            # im_panel = im_panel.cpu().detach().numpy()
            # imwrite("panel_im.png", im_panel)
            # pointgoal = self.observations['pointgoal_with_gps_compass'].detach().cpu().numpy()
            # actions = self.actions.detach().cpu().numpy()
            # masks = self.masks.detach().cpu().numpy()
            # rewards = self.rewards.detach().cpu().numpy()

            # save_dict = {}
            # save_dict['im'] = im
            # save_dict['pointgoal'] = pointgoal.astype(np.float16)
            # save_dict['actions'] = actions.astype(np.uint8)
            # save_dict['masks'] = masks.astype(np.bool)
            # save_dict['rewards'] = rewards.astype(np.float16)
            # np.savez("rollouts/{}.npz".format(self.log_step), **save_dict)

            self.log_step = self.log_step + 1


        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[self.step] = next_value
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset] # the env

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind] # add the first hidden state
                )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )

                adv_targ.append(advantages[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )
            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)

            # States is a (num_recurrent_layers, N, k, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1)
            if self.num_modules == 1:
                recurrent_hidden_states_batch = recurrent_hidden_states_batch.squeeze(-2)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])
