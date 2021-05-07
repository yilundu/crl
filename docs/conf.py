# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

# TODO make this less brittle
sys.path = [os.path.join(os.path.dirname(__file__), "../")] + sys.path

import habitat  # isort:skip

# Overrides the __all__ as that one pulls everything into the root module
# and doesn't expose any submodules
habitat.__all__ = ["config", "core", "Agent", "Benchmark"]
habitat.core.__all__ = [
    "env",
    "embodied_task",
    "dataset",
    "simulator",
    "registry",
    "vector_env",
]
# yacs.config isn't ours, so don't document it
habitat.config.__all__.remove("Config")

PROJECT_TITLE = "Habitat"
PROJECT_SUBTITLE = "Lab Docs"
PROJECT_LOGO = "../../habitat-sim/docs/habitat.svg"
FAVICON = "../../habitat-sim/docs/habitat-blue.png"
MAIN_PROJECT_URL = "/"
INPUT_MODULES = [habitat]
INPUT_DOCS = ["docs.rst"]
INPUT_PAGES = [
    "pages/index.rst",
    "pages/quickstart.rst",
    "pages/habitat-sim-demo.rst",
    "pages/habitat-lab-demo.rst",
    "pages/view-transform-warp.rst",
]

PLUGINS = [
    "m.abbr",
    "m.code",
    "m.components",
    "m.dox",
    "m.gh",
    "m.htmlsanity",
    "m.images",
    "m.link",
    "m.math",
    "m.sphinx",
]

CLASS_INDEX_EXPAND_LEVELS = 2

PYBIND11_COMPATIBILITY = True
ATTRS_COMPATIBILITY = True

# Putting output into the sim repository so relative linking works the same
# way as on the website
OUTPUT = "../../habitat-sim/build/docs/habitat-lab/"

LINKS_NAVBAR1 = [
    (
        "Pages",
        "pages",
        [
            ("Quickstart", "quickstart"),
            ("Habitat Sim Demo", "habitat-sim-demo"),
            ("Habitat Lab Demo", "habitat-lab-demo"),
            ("View, Transform and Warp", "view-transform-warp"),
        ],
    ),
    ("Classes", "classes", []),
]
LINKS_NAVBAR2 = [
    ("Sim Docs", "../habitat-sim/index.html", []),
]

FINE_PRINT = f"""
| {PROJECT_TITLE} {PROJECT_SUBTITLE}. Copyright © 2019 Facebook AI Research.
| Created with `m.css Python doc generator <https://mcss.mosra.cz/documentation/python/>`_."""

STYLESHEETS = [
    "https://fonts.googleapis.com/css?family=Source+Sans+Pro:400,400i,600,600i%7CSource+Code+Pro:400,400i,600",
    "../../habitat-sim/docs/theme.compiled.css",
]

M_SPHINX_INVENTORIES = [
    (
        "../../habitat-sim/docs/python.inv",
        "https://docs.python.org/3/",
        [],
        ["m-doc-external"],
    ),
    (
        "../../habitat-sim/docs/numpy.inv",
        "https://docs.scipy.org/doc/numpy/",
        [],
        ["m-doc-external"],
    ),
    (
        "../../habitat-sim/build/docs/habitat-sim/objects.inv",
        "../habitat-sim/",
        [],
        ["m-doc-external"],
    ),
]
M_SPHINX_INVENTORY_OUTPUT = "objects.inv"
M_SPHINX_PARSE_DOCSTRINGS = True

M_HTMLSANITY_SMART_QUOTES = True
# Will people hate me if I enable this?
# M_HTMLSANITY_HYPHENATION = True
