#!/bin/sh -l

. /install/venv_3.10/bin/activate
cd /cuvis.ai
pip install . >> $GITHUB_OUTPUT
