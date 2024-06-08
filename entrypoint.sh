#!/bin/bash

. /install/venv_3.10/bin/activate
cd /install/cuvis.ai
pip install . >> GITHUB_OUTPUT.txt
cat GITHUB_OUTPUT.txt