#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
import random
import json
import attr

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.auxiliary_tasks import get_aux_task_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.dqn.policy import QNetwork
from habitat_baselines.rl.ppo import PPO, POLICY_CLASSES, MULTIPLE_BELIEF_CLASSES
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import os.path as osp
import random
from torch import optim


class RolloutDataset(Dataset):
    def __init__(self):
        self.base_path = "/private/home/yilundu/sandbox/habitat/habitat-lab-old/rollouts"
        files = os.listdir(self.base_path)
        self.files = files
        self.window = 20

    def __getitem__(self, index):

        start = random.randint(0, 90)
        env = random.randint(0, 3)

        f = self.files[index]
        joint_path = osp.join(self.base_path, f)
        data = np.load(joint_path)

        im = data['im']
        pointgoal = data['pointgoal']
        actions = data['actions']
        masks = data['masks']
        reward = data['rewards']

        im_select = im[start:start+self.window+1, :, ::2, ::2]
        pointgoal_select = pointgoal[start:start+self.window+1, :]
        actions_select = actions[start:start+self.window, :]
        masks_select = masks[start:start+self.window+1, :]
        reward_select = reward[start:start+self.window, :]

        return im_select, pointgoal_select, actions_select, masks_select, reward_select

    def __len__(self):
        return len(self.files)

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def collate(data):
    data = data.transpose(2, 1)
    s = data.shape
    data = data.reshape(s[0]*s[1], *s[2:])

    return data

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count

# TODO support generic sensors
# TODO add compass
class Diagnostics:
    basic = "basic" # dummy to record episode stats (for t-test)
    actions = "actions"
    gps = "gps"
    heading = "heading"
    weights = "weights"
    top_down_map = "top_down_map"

@baseline_registry.register_trainer(name="dqn")
class DQNTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []
        self.rms = RunningMeanStd()

        if config is not None:
            logger.info(f"config: {config}")
            self.checkpoint_prefix = config.TENSORBOARD_DIR.split('/')[-1]

        self._static_encoder = False
        self._encoder = None

        def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
            color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            color_distort = transforms.Compose([
                rnd_color_jitter,
                rnd_gray])
            return color_distort

        self.color_transform = get_color_distortion()

    def get_ppo_class(self):
        return PPO

    def _setup_dqn_agent(self, ppo_cfg: Config, task_cfg: Config, aux_cfg: Config = None, aux_tasks=[]) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        if ppo_cfg.policy not in POLICY_CLASSES:
            raise Exception(f"Illegal policy {ppo_cfg.policy} provided. Valid policies are {POLICY_CLASSES.keys()}")
        if len(aux_tasks) != 0 and len(aux_tasks) != len(aux_cfg.tasks):
            raise Exception(f"Policy specifies {len(aux_cfg.tasks)} tasks but {len(aux_tasks)} were initialized.")
        policy_class = POLICY_CLASSES[ppo_cfg.policy]

        # Default policy settings for object nav
        is_objectnav = "ObjectNav" in task_cfg.TYPE
        additional_sensors = []
        embed_goal = False
        if is_objectnav:
            additional_sensors = ["gps", "compass"]
            embed_goal = True

        self.q_network = QNetwork(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            num_heads=ppo_cfg.num_heads,
            goal_sensor_uuid=task_cfg.GOAL_SENSOR_UUID,
            num_tasks=len(aux_cfg.tasks), # we pass this is in to support eval, where no aux modules are made
            additional_sensors=additional_sensors,
        ).to(self.device)

        self.q_network_target = QNetwork(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            num_heads=ppo_cfg.num_heads,
            goal_sensor_uuid=task_cfg.GOAL_SENSOR_UUID,
            num_tasks=len(aux_cfg.tasks), # we pass this is in to support eval, where no aux modules are made
            additional_sensors=additional_sensors,
        ).to(self.device)

        self.q_network_target.eval()

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, self.q_network.parameters())),
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
        )

        self.sync_model()

    def sync_model(self):
        for param, target_param in zip(self.q_network.parameters(), self.q_network_target.parameters()):
            target_param[:] = param


    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.q_network.state_dict(),
            "target_state_dict": self.q_network_target.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.get_recurrent_states()[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        env_time += time.time() - t_step_env
        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        running_episode_stats["length"] += masks

        epinfo = []
        for i, done in enumerate(dones):
            if done:
                num_tiles = infos[i]['num_tiles']['num_tiles']
                counts = int(running_episode_stats['length'][i])
                epinfo.append({'l': counts, 'num_tiles': num_tiles})
                running_episode_stats['length'][i] = 0

        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        # running_episode_stats["length"] = running_episode_stats["length"] * masks


        current_episode_reward *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs, epinfo

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.get_recurrent_states()[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy, aux_task_losses, aux_dist_entropy, aux_weights = self.agent.update(rollouts)

        rollouts.after_update()
        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
            aux_task_losses,
            aux_dist_entropy,
            aux_weights
        )

    def train(self, ckpt_path="", ckpt=-1, start_updates=0) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        observation_space = self.envs.observation_spaces[0]

        ppo_cfg = self.config.RL.PPO
        task_cfg = self.config.TASK_CONFIG.TASK
        aux_cfg = self.config.RL.AUX_TASKS

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        self._setup_dqn_agent(ppo_cfg, task_cfg, aux_cfg, [])

        self.dataset = RolloutDataset()
        self.dataloader = DataLoader(self.dataset, batch_size=16, num_workers=0)

        # Use environment to initialize the metadata for training the model
        self.envs.close()

        if self.config.RESUME_CURIOUS:
            weights = torch.load(self.config.RESUME_CURIOUS)['state_dict']
            state_dict = self.q_network.state_dict()

            weights_new = {}

            for k, v in weights.items():
                if "model_encoder" in k:
                    k = k.replace("model_encoder", "visual_resnet").replace("actor_critic.", "")
                    if k in state_dict:
                        weights_new[k] = v

            state_dict.update(weights_new)
            self.q_network.load_state_dict(state_dict)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.q_network.parameters())
            )
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        if ckpt != -1:
            logger.info(
                f"Resuming runs at checkpoint {ckpt}. Timing statistics are not tracked properly."
            )
            assert ppo_cfg.use_linear_lr_decay is False and ppo_cfg.use_linear_clip_decay is False, "Resuming with decay not supported"
            # This is the checkpoint we start saving at
            count_checkpoints = ckpt + 1
            count_steps = start_updates * ppo_cfg.num_steps * self.config.NUM_PROCESSES
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.q_network.load_state_dict(ckpt_dict["state_dict"])
            self.q_network_target.load_state_dict(ckpt_dict["target_state_dict"])
            if "optim_state" in ckpt_dict:
                self.agent.optimizer.load_state_dict(ckpt_dict["optim_state"])
            else:
                logger.warn("No optimizer state loaded, results may be funky")
            if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
                count_steps = ckpt_dict["extra_state"]["step"]


        lr_scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )
        im_size = 256

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:

            update = 0
            for i in range(self.config.NUM_EPOCHS):
                for im, pointgoal, action, mask, reward in self.dataloader:
                    if ppo_cfg.use_linear_lr_decay:
                        lr_scheduler.step()

                    im, pointgoal, action, mask, reward = collate(im), collate(pointgoal), collate(action), collate(mask), collate(reward)
                    im = im.to(self.device).float()
                    pointgoal = pointgoal.to(self.device).float()
                    mask = mask.to(self.device).float()
                    reward = reward.to(self.device).float()
                    action = action.to(self.device).long()
                    nstep = im.size(1)

                    hidden_states = None
                    hidden_states_target = None

                    # q_vals = []
                    # q_vals_target = []

                    step = random.randint(0, nstep-1)
                    output = self.q_network({'rgb': im[:, step]},  None, None)
                    mse_loss = torch.pow(output - im[:, step] / 255., 2).mean()
                    mse_loss.backward()

                    # for step in range(nstep):
                    #     q_val, hidden_states = self.q_network({'rgb': im[:, step], 'pointgoal_with_gps_compass': pointgoal[:, step]}, hidden_states, mask[:, step])

                    #     q_val_target, hidden_states_target = self.q_network_target({'rgb': im[:, step], 'pointgoal_with_gps_compass': pointgoal[:, step]}, hidden_states_target, mask[:, step])

                    #     q_vals.append(q_val)
                    #     q_vals_target.append(q_val_target)

                    # q_vals = torch.stack(q_vals, dim=1)
                    # q_vals_target = torch.stack(q_vals_target, dim=1)

                    # a_select = torch.argmax(q_vals, dim=-1, keepdim=True)
                    # target_select = torch.gather(q_vals_target, -1, a_select)

                    # target = reward + ppo_cfg.gamma * target_select[:, 1:] * mask[:, 1:]
                    # target = target.detach()

                    # pred_q = torch.gather(q_vals[:, :-1], -1, action)

                    # mse_loss = torch.pow(pred_q - target, 2).mean()
                    # mse_loss.backward()
                    # grad_norm = torch.nn.utils.clip_grad_norm(self.q_network.parameters(), 80)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    writer.add_scalar(
                        "loss",
                        mse_loss,
                        update,
                    )

                   #  writer.add_scalar(
                   #      "q_val",
                   #      q_vals.max(),
                   #      update,
                   #  )

                    if update % 10 == 0:
                        print("Update: {}, loss: {}".format(update, mse_loss))

                    if update % 100 == 0:
                        self.sync_model()

                    # checkpoint model
                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(
                            f"{self.checkpoint_prefix}.{count_checkpoints}.pth", dict(step=count_steps)
                        )
                        count_checkpoints += 1
                    update = update + 1

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
        log_diagnostics=[],
        output_dir='.',
        label='.',
        num_eval_runs=1
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if checkpoint_index == -1:
            ckpt_file = checkpoint_path.split('/')[-1]
            split_info = ckpt_file.split('.')
            checkpoint_index = split_info[1]
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO
        task_cfg = config.TASK_CONFIG.TASK

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        # pass in aux config if we're doing attention
        aux_cfg = self.config.RL.AUX_TASKS
        self._setup_dqn_agent(ppo_cfg, task_cfg, aux_cfg, [])

        # Check if we accidentally recorded `visual_resnet` in our checkpoint and drop it (it's redundant with `visual_encoder`)
        # ckpt_dict['state_dict'] = {
        #     k:v for k, v in ckpt_dict['state_dict'].items() if 'visual_resnet' not in k
        # }
        self.q_network.load_state_dict(ckpt_dict["state_dict"])
        self.q_network_target.load_state_dict(ckpt_dict["target_state_dict"])

        logger.info(
            "agent number of trainable parameters: {}".format(
                sum(
                    param.numel()
                    for param in self.q_network.parameters()
                    if param.requires_grad
                )
            )
        )

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            1,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )

        # _, num_recurrent_memories, _ = self._setup_auxiliary_tasks(aux_cfg, ppo_cfg, task_cfg, is_eval=True)
        # if self.config.RL.PPO.policy in MULTIPLE_BELIEF_CLASSES:
        #     aux_tasks = self.config.RL.AUX_TASKS.tasks
        #     num_recurrent_memories = len(self.config.RL.AUX_TASKS.tasks)
        #     test_recurrent_hidden_states = test_recurrent_hidden_states.unsqueeze(2).repeat(1, 1, num_recurrent_memories, 1)


        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )

        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]

        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        videos_cap = 10 # number of videos to generate per checkpoint
        if len(log_diagnostics) > 0:
            videos_cap = 10
        # video_indices = random.sample(range(self.config.TEST_EPISODE_COUNT),
            # min(videos_cap, self.config.TEST_EPISODE_COUNT))
        video_indices = range(10)
        print(f"Videos: {video_indices}")

        total_stats = []
        dones_per_ep = dict()

        # Logging more extensive evaluation stats for analysis
        if len(log_diagnostics) > 0:
            d_stats = {}
            for d in log_diagnostics:
                d_stats[d] = [
                    [] for _ in range(self.config.NUM_PROCESSES)
                ] # stored as nested list envs x timesteps x k (# tasks)

        pbar = tqdm.tqdm(total=number_of_eval_episodes * num_eval_runs)
        self.q_network.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes * num_eval_runs
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()
            with torch.no_grad():
                weights_output = None
                if self.config.RL.PPO.policy in MULTIPLE_BELIEF_CLASSES:
                    weights_output = torch.empty(self.envs.num_envs, len(aux_tasks))
                (
                    q_vals,
                    test_recurrent_hidden_states,
                ) = self.q_network(
                    batch,
                    test_recurrent_hidden_states,
                    not_done_masks,
                )
                prob = torch.softmax(q_vals * 10, dim=-1)
                actions = torch.multinomial(prob, 1)
                # actions = torch.argmax(q_vals, dim=-1, keepdim=True)

                for i in range(self.envs.num_envs):
                    if Diagnostics.weights in log_diagnostics:
                        aux_weights = None if weights_output is None else weights_output[i]
                        if aux_weights is not None:
                            d_stats[Diagnostics.weights][i].append(aux_weights.half().tolist())

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                next_k = (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                )
                if dones_per_ep.get(next_k, 0) == num_eval_runs:
                    envs_to_pause.append(i) # wait for the rest

                if not_done_masks[i].item() == 0:
                    episode_stats = dict()

                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )

                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats

                    k = (
                        current_episodes[i].scene_id,
                        current_episodes[i].episode_id,
                    )
                    dones_per_ep[k] = dones_per_ep.get(k, 0) + 1

                    if dones_per_ep.get(k, 0) == 1 and len(self.config.VIDEO_OPTION) > 0 and len(stats_episodes) in video_indices:
                        logger.info(f"Generating video {len(stats_episodes)}")
                        category = getattr(current_episodes[i], "object_category", "")
                        if category != "":
                            category += "_"
                        try:
                            generate_video(
                                video_option=self.config.VIDEO_OPTION,
                                video_dir=self.config.VIDEO_DIR,
                                images=rgb_frames[i],
                                episode_id=current_episodes[i].episode_id,
                                checkpoint_idx=checkpoint_index,
                                metrics=self._extract_scalars_from_info(infos[i]),
                                tag=f"{category}{label}",
                                tb_writer=writer,
                            )
                        except Exception as e:
                            logger.warning(str(e))
                    rgb_frames[i] = []

                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                            dones_per_ep[k],
                        )
                    ] = episode_stats

                    if len(log_diagnostics) > 0:
                        diagnostic_info = dict()
                        for metric in log_diagnostics:
                            diagnostic_info[metric] = d_stats[metric][i]
                            d_stats[metric][i] = []
                        if Diagnostics.top_down_map in log_diagnostics:
                            top_down_map = torch.tensor([])
                            if len(self.config.VIDEO_OPTION) > 0:
                                top_down_map = infos[i]["top_down_map"]["map"]
                                top_down_map = maps.colorize_topdown_map(
                                    top_down_map, fog_of_war_mask=None
                                )
                            diagnostic_info.update(dict(top_down_map=top_down_map))
                        total_stats.append(
                            dict(
                                stats=episode_stats,
                                did_stop=bool(prev_actions[i] == 0),
                                episode_info=attr.asdict(current_episodes[i]),
                                info=diagnostic_info,
                            )
                        )
                    pbar.update()

                # episode continues
                else:
                    if len(self.config.VIDEO_OPTION) > 0:
                        aux_weights = None if weights_output is None else weights_output[i]
                        frame = observations_to_image(observations[i], infos[i])
                        rgb_frames[i].append(frame)
                    if Diagnostics.gps in log_diagnostics:
                        d_stats[Diagnostics.gps][i].append(observations[i]["gps"].tolist())
                    if Diagnostics.heading in log_diagnostics:
                        d_stats[Diagnostics.heading][i].append(observations[i]["heading"].tolist())

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)
            logger.info("eval_metrics")
            logger.info(metrics)
        if len(log_diagnostics) > 0:
            os.makedirs(output_dir, exist_ok=True)
            eval_fn = f"{label}.json"
            with open(os.path.join(output_dir, eval_fn), 'w', encoding='utf-8') as f:
                json.dump(total_stats, f, ensure_ascii=False, indent=4)
        self.envs.close()
