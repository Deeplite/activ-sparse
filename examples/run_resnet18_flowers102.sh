#!/bin/bash

DATASET=flowers102
MODEL=resnet18
SP=0.3
EPOCHS=400
LR=0.0001
END=0.5 # 0.9

CUDA_VISIBLE_DEVICES=0 \
python train.py --dataset $DATASET --model $MODEL \
      --imgsz 224 --batch-size 64 --pretrained --epochs $EPOCHS --lr0 $LR \
      --layers2skip conv1x1 --prune_final_value $SP --prune_end $END \
      --name ${DATASET}${MODEL}__s${SP} --save_masks
