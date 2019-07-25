#!/bin/bash

# python run_cifar.py \
#   --dataset cifar10 \
#   --data_augmentation \
#   --output_dir cifar10_results \
#   --seed 0
python order_examples_by_forgetting.py \
    --output_dir cifar10_results \
    --output_name cifar10_sorted \
    --input_dir cifar10_results \
    --input_fname_args \
        dataset cifar10 \
        data_augmentation True \
        cutout False \
        sorting_file none \
        remove_n 0 \
        keep_lowest_n 0 \
        remove_subsample 0 \
        noise_percent_labels 0 \
        noise_percent_pixels 0 \
        noise_std_pixels 0
