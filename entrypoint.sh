#!/bin/bash

. /install/venv_3.10/bin/activate && python3.10 -m pip install /install/cuvis.pyil

# Install again with the default python3.10
python3.10 -m pip install /install/cuvis.pyil
cp /install/cuvis.pyil/_cuvis_pyil.so /root/.local/lib/python3.10/site-packages/cuvis_il/
cp /install/cuvis.pyil/cuvis_il.py /root/.local/lib/python3.10/site-packages/cuvis_il/
cd /install/cuvis.ai
/install/venv_3.10/bin/python3.10 -m pip install .
echo "======================="
echo "Test CUVIS.AI is importable"
echo "======================="
/install/venv_3.10/bin/python3.10 -c "import cuvis; import cuvis_ai"
echo "======================="
echo "OUTPUT OF CUVIS.AI Unit testing"
echo "======================="
. /install/venv_3.10/bin/activate && python3.10 -m unittest discover