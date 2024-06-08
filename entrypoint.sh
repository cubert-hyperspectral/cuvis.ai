#!/bin/bash

. /install/venv_3.10/bin/activate && python3.10 -m pip install /install/cuvis.pyil

# Install again with the default python3.10
python3.10 -m pip install /install/cuvis.pyil

cd /install/cuvis.ai
/install/venv_3.10/bin/python3.10 -m pip install .
echo "======================="
echo "OUTPUT OF CUVIS.AI Unit testing"
echo "======================="
. /install/venv_3.10/bin/activate && /install/venv_3.10/bin/python3.10 -m unittest discover