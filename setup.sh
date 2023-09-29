#!/bin/bash

python3 -m venv py3
source py3/bin/activate
pip install --upgrade pip
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
