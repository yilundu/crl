#!/usr/bin/env bash

# Propagate failures properly
set -e

if [[ $# -eq 1 ]]; then
  export mcss_path=$1
elif [[ $# -ne 0 ]]; then
  echo "usage: ./build.sh [path-to-m.css]"
  exit 1
else
  if [ ! -d ../../habitat-sim/docs/m.css ]; then
    echo "m.css submodule not found in the sim repository, please run git submodule update --init there or specify the path to it"
    exit 1
  fi
  mcss_path=../../habitat-sim/docs/m.css
fi

# Regenerate the compiled CSS file (yes, in the sim repository, to allow fast
# iterations from here as well)
$mcss_path/css/postprocess.py \
  ../../habitat-sim/docs/theme.css \
  $mcss_path/css/m-grid.css \
  $mcss_path/css/m-components.css \
  $mcss_path/css/m-layout.css \
  ../../habitat-sim/docs/pygments-pastie.css \
  $mcss_path/css/pygments-console.css \
  $mcss_path/css/m-documentation.css \
  -o ../../habitat-sim/docs/theme.compiled.css

$mcss_path/documentation/python.py conf.py

# The file:// URLs are usually clickable in the terminal, directly opening a
# browser
echo "------------------------------------------------------------------------"
echo "Docs were successfully generated. Open the following link to view them:"
echo
echo "file://$(pwd)/../../habitat-sim/build/docs/habitat-lab/index.html"
