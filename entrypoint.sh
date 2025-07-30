#!/bin/bash

cd /app
python3.12 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install torch torchvision matplotlib --index-url https://download.pytorch.org/whl/cpu
python3.12 -m pip install .
python3.12 -m pip install opencv-python-headless tzdata
echo "======================="
echo "Test CUVIS.AI is importable"
echo "======================="
python3.12 -c "import cuvis; import cuvis_ai; print(cuvis.General.version())"
echo "======================="
echo "OUTPUT OF CUVIS.AI Unit testing"
echo "======================="
python3.12 -m unittest discover