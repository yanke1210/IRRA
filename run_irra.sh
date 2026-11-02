#!/bin/bash
# Best hyperparameters for the dataset are in config/${DATASET_NAME}.yaml.
# Append flags to override them, for example: --lr 1e-4
DATASET_NAME="RSICD"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--dataset_name $DATASET_NAME
