#!/bin/bash

cd /app
python3.10 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3.10 -m pip install .
python3.10 -m pip install opencv-python-headless tzdata
python3.10 -m pip install -r docs/requirements.txt
echo "======================="
echo "Test CUVIS.AI is importable"
echo "======================="
python3.10 -c "import cuvis; import cuvis_ai"
echo "======================="
echo "Generate CUVIS.AI documentation"
echo "======================="
mkdir -p docs/_build
sphinx-build -M html docs docs/_build
touch docs/_build/html/.nojekyll