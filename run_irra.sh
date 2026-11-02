#!/bin/bash
DATASET_NAME="RSICD"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name irra \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id+tc' \
--num_epoch 60
