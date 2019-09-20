#!/bin/bash

python main.py \
  --dataset cifar10 \
  --data_augmentation \
  --epochs 200 \
  --output_dir online_results \
  --burn_in_epochs 50 \
  --noise_percent_labels 30 \
  --remove_percent 40 \
  --remove_strategy "unforgettable" \
  --device "cuda:5" \
  --seed 1
