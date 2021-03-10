#!/bin/bash

rm log_files.txt

python src/pretrain.py -c ~/Documents/BarlowTwins/config.conf
python src/finetune.py -c ~/Documents/BarlowTwins/config.conf
python src/test_finetune.py -c ~/Documents/BarlowTwins/config.conf
