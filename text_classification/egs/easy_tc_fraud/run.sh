#!/bin/bash
set -e
num_gpu=2 #$(echo $gpus | awk -F ' ' '{print NF}')
gpus="0 1"
torchrun --master_port 12345 --nproc_per_node=$num_gpu main.py --config conf/config.yaml --gpu $gpus
