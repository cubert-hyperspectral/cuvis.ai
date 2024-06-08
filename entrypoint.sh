#!/bin/sh -l

. /install/venv_3.10/bin/activate
cd /catkin_ws/cuvis.ai
pip install . >> $GITHUB_OUTPUT
