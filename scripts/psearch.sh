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
python src/train.py -e 200 -b 100  -lr 0.001   -sd save/b100_lr3
python src/train.py -e 200 -b 100  -lr 0.0001  -sd save/b100_lr4
python src/train.py -e 200 -b 100  -lr 0.00001 -sd save/b100_lr5
python src/train.py -e 200 -b 500  -lr 0.001   -sd save/b500_lr3
python src/train.py -e 200 -b 500  -lr 0.0001  -sd save/b500_lr4
python src/train.py -e 200 -b 500  -lr 0.0001  -sd save/b500_lr5
python src/train.py -e 200 -b 1000 -lr 0.001   -sd save/b1000_lr3
python src/train.py -e 200 -b 1000 -lr 0.0001  -sd save/b1000_lr4
python src/train.py -e 200 -b 1000 -lr 0.0001  -sd save/b1000_lr5