#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.datasets.eqa import _try_register_mp3d_eqa_dataset
from habitat.datasets.object_nav import _try_register_objectnavdatasetv1
from habitat.datasets.pointnav import _try_register_pointnavdatasetv1
from habitat.datasets.vln import _try_register_r2r_vln_dataset


def make_dataset(id_dataset, **kwargs):
    logger.info("Initializing dataset {}".format(id_dataset))
    _dataset = registry.get_dataset(id_dataset)
    assert _dataset is not None, "Could not find dataset {}".format(id_dataset)

    return _dataset(**kwargs)  # type: ignore


_try_register_objectnavdatasetv1()
_try_register_mp3d_eqa_dataset()
_try_register_pointnavdatasetv1()
_try_register_r2r_vln_dataset()
