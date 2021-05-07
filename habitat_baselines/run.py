#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import shutil
import habitat

import argparse
import random

import numpy as np
import torch

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config

from my_sensors import EpisodeInfoExample

# Whether to fail if run files exist
DO_PRESERVE_RUNS = False # TODO sets to true

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--ckpt-path",
        default=None,
        type=str,
        help="full path to a ckpt"
    )

    parser.add_argument(
        "--run-id",
        type=int,
        required=False,
        help="running a batch - give run id",
    )

    parser.add_argument(
        "--run-suffix",
        type=str,
        required=False,
        help="Modify run name (for bookkeeping when changes aren't recorded in config)"
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    ckpt_wrapper(**vars(args))

def ckpt_wrapper(exp_config: str, run_type: str, ckpt_path="", run_id=None, run_suffix="", opts=None) -> None:

    if ckpt_path is None:
        run_exp(exp_config, run_type, ckpt_path=ckpt_path, run_id=run_id, run_suffix=run_suffix, opts=opts)
        return
    all_paths = ckpt_path.split(",")
    for pth in all_paths:
        if len(pth) > 0:
            run_exp(exp_config, run_type, ckpt_path=pth, run_id=run_id, run_suffix=run_suffix, opts=opts)


def run_exp(exp_config: str, run_type: str, ckpt_path="", run_id=None, run_suffix="", opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        ckpt_path: If training, ckpt to resume. If evaluating, ckpt to evaluate.
        run_id: If using slurm batch, run id to prefix.s
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)

    # Add tracking of the number of episodes
    config.defrost()

    config.TASK_CONFIG.TASK.EPISODE_INFO_EXAMPLE = habitat.Config()
    # The type field is used to look-up the measure in the registry.
    # By default, the things are registered with the class name
    config.TASK_CONFIG.TASK.EPISODE_INFO_EXAMPLE.TYPE = "EpisodeInfoExample"
    config.TASK_CONFIG.TASK.EPISODE_INFO_EXAMPLE.VALUE = 5
    # Add the measure to the list of measures in use
    config.TASK_CONFIG.TASK.MEASUREMENTS.append("EPISODE_INFO_EXAMPLE")
    config.freeze()

    variant_name = os.path.split(exp_config)[1].split('.')[0]
    config.defrost()
    if run_suffix != "" and run_suffix is not None:
        variant_name = f"{variant_name}-{run_suffix}"

    if not osp.exists(config.LOG_FILE):
        os.makedirs(config.LOG_FILE)

    config.TENSORBOARD_DIR = os.path.join(config.TENSORBOARD_DIR, variant_name)
    config.CHECKPOINT_FOLDER = os.path.join(config.CHECKPOINT_FOLDER, variant_name)
    config.VIDEO_DIR = os.path.join(config.VIDEO_DIR, variant_name)
    config.LOG_FILE = os.path.join(config.LOG_FILE, f"{variant_name}.log") # actually a logdir
    if run_type == "eval":
        # config.TRAINER_NAME = "ppo"
        config.NUM_PROCESSES = 6 # nice
    else:
        # Add necessary supervisory signals
        train_sensors = config.RL.AUX_TASKS.required_sensors
        config.SENSORS.extend(train_sensors) # the task cfg sensors are overwritten by this one

    if run_id is None:
        random.seed(config.TASK_CONFIG.SEED)
        np.random.seed(config.TASK_CONFIG.SEED)
        trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
        assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
        trainer = trainer_init(config)

        # If not doing multiple runs (with run_id), default behavior is to overwrite
        if run_type == "train":
            if ckpt_path is not None:
                ckpt_dir, ckpt_file = os.path.split(ckpt_path)
                ckpt_index = ckpt_file.split('.')[1]
                ckpt = int(ckpt_index)
                start_updates = ckpt * config.CHECKPOINT_INTERVAL + 1
                trainer.train(ckpt_path=ckpt_path, ckpt=ckpt, start_updates=start_updates)
            elif not DO_PRESERVE_RUNS:
                # if os.path.exists(config.TENSORBOARD_DIR):
                #     print("Removing tensorboard directory...")
                #     shutil.rmtree(config.TENSORBOARD_DIR, ignore_errors=True)
                # if os.path.exists(config.CHECKPOINT_FOLDER):
                #     print("Removing checkpoint folder...")
                #     shutil.rmtree(config.CHECKPOINT_FOLDER, ignore_errors=True)
                # if os.path.exists(config.LOG_FILE):
                #     print("Removing log file...")
                #     shutil.rmtree(config.LOG_FILE, ignore_errors=True)
                trainer.train()
            else:
                # if os.path.exists(config.TENSORBOARD_DIR) or os.path.exists(config.CHECKPOINT_FOLDER) \
                #     or os.path.exists(config.LOG_FILE):
                #     print(f"TB dir exists: {os.path.exists(config.TENSORBOARD_DIR)}")
                #     print(f"Ckpt dir exists: {os.path.exists(config.CHECKPOINT_FOLDER)}")
                #     print(f"Log file exists: {os.path.exists(config.LOG_FILE)}")
                #     print("Run artifact exists, please clear manually")
                #     exit(1)
                trainer.train()
        elif run_type == "eval":
            trainer.eval(ckpt_path)
        return

    run_prefix = f'run_{run_id}'
    seed = run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(config.TASK_CONFIG.SEED)


    # Exetnds off old modifications
    tb_dir = os.path.join(config.TENSORBOARD_DIR, run_prefix)
    ckpt_dir = os.path.join(config.CHECKPOINT_FOLDER, run_prefix)
    log_dir, log_file = os.path.split(config.LOG_FILE)
    log_file_extended = f"{run_prefix}--{log_file}"
    log_file_path = os.path.join(log_dir, log_file_extended)

    config.TASK_CONFIG.SEED = seed
    config.TENSORBOARD_DIR = tb_dir
    config.CHECKPOINT_FOLDER = ckpt_dir
    config.LOG_FILE = log_file_path

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    if run_type == "train":
        if ckpt_path is None:
            # if DO_PRESERVE_RUNS and (os.path.exists(tb_dir) or os.path.exists(ckpt_dir) or os.path.exists(log_file_extended)):
            #     print(f"TB dir exists: {os.path.exists(tb_dir)}")
            #     print(f"Ckpt dir exists: {os.path.exists(ckpt_dir)}")
            #     print(f"Log file exists: {os.path.exists(log_file_extended)}")
            #     print("Run artifact exists, please clear manually")
            #     exit(1)
            # else:
            #     shutil.rmtree(tb_dir, ignore_errors=True)
            #     shutil.rmtree(ckpt_dir, ignore_errors=True)
            #     if os.path.exists(log_file_extended):
            #         os.remove(log_file_extended)
            trainer.train()
        else: # Resume training from checkpoint
            # Parse the checkpoint #, calculate num updates, update the config
            ckpt_dir, ckpt_file = os.path.split(ckpt_path)
            ckpt_index = ckpt_file.split('.')[1]
            true_path = os.path.join(ckpt_dir, run_prefix, f"{run_prefix}.{ckpt_index}.pth")
            ckpt = int(ckpt_index)
            start_updates = ckpt * config.CHECKPOINT_INTERVAL + 1
            trainer.train(ckpt_path=true_path, ckpt=ckpt, start_updates=start_updates)
    else:
        ckpt_dir, ckpt_file = os.path.split(ckpt_path)
        ckpt_index = ckpt_file.split('.')[1]
        true_path = os.path.join(ckpt_dir, run_prefix, f"{run_prefix}.{ckpt_index}.pth")
        trainer.eval(true_path)

if __name__ == "__main__":
    main()
