#!/bin/bash

cd /app/cuvis.ai
python3.9 -m pip install .
echo "======================="
echo "Test CUVIS.AI is importable"
echo "======================="
python3.9 -c "import cuvis; import cuvis_ai"
echo "======================="
echo "OUTPUT OF CUVIS.AI Unit testing"
echo "======================="
python3.9 -m unittest discover