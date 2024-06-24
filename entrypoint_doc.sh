#!/bin/bash

cd /install/cuvis.ai
source /install/venv_3.10/bin/activate
python -m pip install .
python -m pip install -r docs/requirements.txt

echo "======================="
echo "Test CUVIS.AI is importable"
echo "======================="
/install/venv_3.10/bin/python3.10 -c "import cuvis; import cuvis_ai"

echo "======================="
echo "Build documentation"
echo "======================="
sphinx-build -M html docs docs/_build
touch docs/_build/html/.nojekyll