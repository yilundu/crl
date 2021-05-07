#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict
from habitat_baselines.rl.ppo.policy import (
    Net, Policy, BaselinePolicy
)
from habitat_baselines.rl.ppo.belief_policy import (
    BeliefPolicy, AttentiveBeliefPolicy, MidLevelPolicy,
    FixedAttentionBeliefPolicy, AverageBeliefPolicy, SoftmaxBeliefPolicy,
)

from habitat_baselines.rl.ppo.ppo import PPO

SINGLE_BELIEF_CLASSES: Dict[str, Policy] = {
    "BASELINE": BaselinePolicy,
    "SINGLE_BELIEF": BeliefPolicy,
    "MIDLEVEL": MidLevelPolicy,
}

MULTIPLE_BELIEF_CLASSES = {
    "ATTENTIVE_BELIEF": AttentiveBeliefPolicy,
    "FIXED_ATTENTION_BELIEF": FixedAttentionBeliefPolicy,
    "AVERAGE_BELIEF": AverageBeliefPolicy,
    "SOFTMAX_BELIEF": SoftmaxBeliefPolicy,
}

POLICY_CLASSES = dict(SINGLE_BELIEF_CLASSES, **MULTIPLE_BELIEF_CLASSES)

__all__ = [
    "PPO", "Policy", "Net", "POLICY_CLASSES", "SINGLE_BELIEF_CLASSES", "MULTIPLE_BELIEF_CLASSES"
]
