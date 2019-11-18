#!/bin/bash

# python main.py \
#   --dataset cifar10 \
#   --data_augmentation \
#   --epochs 200 \
#   --output_dir online_results \
#   --burn_in_epochs 50 \
#   --noise_percent_labels 30 \
#   --remove_percent 40 \
#   --remove_strategy "normal" \
#   --optimizer "sgd-const-lr" \
#   --learning_rate 0.1 \
#   --device "cuda:0" \
#   --seed 1
# 
# python main.py \
#   --dataset cifar10 \
#   --data_augmentation \
#   --epochs 200 \
#   --output_dir online_results \
#   --burn_in_epochs 50 \
#   --noise_percent_labels 30 \
#   --remove_percent 40 \
#   --remove_strategy "normal" \
#   --optimizer "sgd-const-lr" \
#   --learning_rate 0.02 \
#   --device "cuda:1" \
#   --seed 1
# 
# python main.py \
#   --dataset cifar10 \
#   --data_augmentation \
#   --epochs 200 \
#   --output_dir online_results \
#   --burn_in_epochs 50 \
#   --noise_percent_labels 30 \
#   --remove_percent 40 \
#   --remove_strategy "normal" \
#   --optimizer "sgd-const-lr" \
#   --learning_rate 0.004 \
#   --device "cuda:2" \
#   --seed 1
# 
# python main.py \
#   --dataset cifar10tfds \
#   --data_augmentation \
#   --epochs 200 \
#   --output_dir online_results \
#   --burn_in_epochs 50 \
#   --noise_percent 0 \
#   --noise_pixels_percent 0 \
#   --remove_percent 0 \
#   --remove_strategy "normal" \
#   --optimizer "sgd" \
#   --device "cuda:0" \
#   --seed 1

# python main.py \
#   --dataset cifar10 \
#   --data_augmentation \
#   --epochs 200 \
#   --output_dir online_results \
#   --burn_in_epochs 50 \
#   --noise_percent 30 \
#   --noise_pixels_percent 100 \
#   --noise_pixels_std -1 \
#   --remove_percent 0 \
#   --remove_strategy "normal" \
#   --device "cuda:1" \
#   --seed 1
# 
# python main.py \
#   --dataset cifar10 \
#   --data_augmentation \
#   --epochs 200 \
#   --output_dir online_results \
#   --burn_in_epochs 50 \
#   --noise_percent 0 \
#   --noise_pixels_percent 0 \
#   --remove_percent 0 \
#   --remove_strategy "normal" \
#   --device "cuda:1" \
#   --seed 1

python main.py \
  --dataset cifar10 \
  --data_augmentation \
  --epochs 200 \
  --output_dir online_results \
  --burn_in_epochs 50 \
  --noise_percent 0 \
  --noise_pixels_percent 0 \
  --remove_percent 0 \
  --remove_strategy "normal" \
  --optimizer "sgd" \
  --device "cuda:0" \
  --seed 1
