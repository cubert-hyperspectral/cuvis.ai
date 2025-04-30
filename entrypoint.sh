#!/bin/bash

cd /app/cuvis_ai
python3.10 -m pip install .
echo "======================="
echo "Test CUVIS.AI is importable"
echo "======================="
python3.10 -c "import cuvis; import cuvis_ai"
echo "======================="
echo "OUTPUT OF CUVIS.AI Unit testing"
echo "======================="
python3.10 -m unittest discover