#!/bin/bash

. /install/venv_3.10/bin/activate
cd /install/cuvis.ai
pip install . >> GITHUB_OUTPUT.txt
echo "OUTPUT OF CUVIS.AI INSTALL\n======================="
cat GITHUB_OUTPUT.txt
echo "OUTPUT OF CUVIS.AI Unit testing\n======================="
python3.10 -m unittest discover