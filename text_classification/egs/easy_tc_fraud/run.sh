#!/bin/bash
set -e
num_gpu=8 #$(echo $gpus | awk -F ' ' '{print NF}')
gpus="0 1 2 3 4 5 6 7"
torchrun --master_port 12345 --nproc_per_node=$num_gpu main.py --config conf/config.yaml --gpu $gpus