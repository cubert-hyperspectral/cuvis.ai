#!/bin/bash

cd /install/cuvis.ai
/install/venv_3.10/bin/python3.10 -m pip install .
echo "======================="
echo "Test CUVIS.AI is importable"
echo "======================="
/install/venv_3.10/bin/python3.10 -c "import cuvis; import cuvis_ai"
echo "======================="
echo "OUTPUT OF CUVIS.AI Unit testing"
echo "======================="
. /install/venv_3.10/bin/activate && python3.10 -m unittest discoverr