#!/usr/bin/env bash
######################
## Parameter Search ##
######################
# verify right directory
cd gyafc_style_transfer
# create virtualenv
source .venv/bin/activate
# verify library is found
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
# train
python src/train.py -e 200 -b 100  -lr 0.0001 -ru 100 -cl 50 25 10 5 -sd save/b100_lr4
python src/train.py -e 200 -b 100  -lr 0.00001 -ru 100 -cl 50 25 10 5 -sd save/b100_lr5
python src/train.py -e 200 -b 500  -lr 0.0001 -ru 100 -cl 50 25 10 5 -sd save/b500_lr4
python src/train.py -e 200 -b 500  -lr 0.0001 -ru 100 -cl 50 25 10 5 -sd save/b500_lr5
python src/train.py -e 200 -b 200 -lr 0.0001 -ru 100 -cl 50 25 10 5 -sd save/b200_lr4
python src/train.py -e 200 -b 200 -lr 0.0001 -ru 100 -cl 50 25 10 5 -sd save/b200_lr5
python src/train.py -e 200 -b 200 -lr 0.0001 -ru 100 -cl 30 10 -sd save/b200_lr4
python src/train.py -e 200 -b 200 -lr 0.0001 -ru 150 -cl 50 10 -sd save/b200_lr4