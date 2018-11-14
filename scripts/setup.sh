#!/usr/bin/env bash
###############
## Lib Setup ##
###############
# verify right directory
cd text_style_transfer
# install pip3
sudo apt-get update
sudo apt-get -y install python3-pip
# create virtualenv
pip3 install virtualenv
python3 -m virtualenv .venv
# activate virtualenv
chmod +x -R .venv
source .venv/bin/activate.sh
# install prerequisite
pip3 install -r requirements.txt
# deactivate virtualenv
deactivate