#!/bin/bash

cd /app
python3.10 -m pip install .
python3.10 -m pip install opencv-python-headlessx
echo "======================="
echo "Test CUVIS.AI is importable"
echo "======================="
python3.10 -c "import cuvis; import cuvis_ai"
echo "======================="
echo "OUTPUT OF CUVIS.AI Unit testing"
echo "======================="
python3.10 -m unittest discover