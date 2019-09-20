#!/bin/bash

# python run_cifar.py \
#   --dataset cifar10 \
#   --data_augmentation \
#   --output_dir cifar10_results \
#   --seed 4
# 
# python order_examples_by_forgetting.py \
#     --output_dir cifar10_results \
#     --output_name cifar10_sorted \
#     --input_dir cifar10_results \
#     --input_fname_args \
#         dataset cifar10 \
#         data_augmentation True \
#         cutout False \
#         sorting_file none \
#         remove_n 0 \
#         keep_lowest_n 0 \
#         remove_subsample 0 \
#         noise_percent_labels 0 \
#         noise_percent_pixels 0 \
#         noise_std_pixels 0
#
# python run_cifar.py \
#     --dataset cifar10 \
#     --data_augmentation \
#     --sorting_file cifar10_sorted \
#     --input_dir cifar10_results \
#     --output_dir cifar10_results \
#     --seed 1 \
#     --remove_n 15000 \
#     --remove_subsample 0 \
#     --keep_lowest_n 0

# python run_online.py \
#   --dataset cifar10 \
#   --remove_n 15000 \
#   --data_augmentation \
#   --epochs 200 \
#   --burn_in_epochs 50 \
#   --mode "online" \
#   --remove_mode "unforgettable" \
#   --device "cuda:3" \
#   --output_dir online_results \
#   --seed 1

# python run_online.py \
#   --dataset cifar10 \
#   --remove_n 15000 \
#   --data_augmentation \
#   --epochs 200 \
#   --burn_in_epochs 50 \
#   --mode "online" \
#   --remove_mode "forgettable" \
#   --device "cuda:4" \
#   --output_dir online_results \
#   --seed 1


python main.py \
  --dataset svhn \
  --data_augmentation \
  --epochs 2 \
  --output_dir online_results \
  --burn_in_epochs 1 \
  --noise_percent_labels 30 \
  --remove_percent 40 \
  --remove_strategy "forgettable" \
  --device "cuda:5" \
  --seed 1
