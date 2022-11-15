#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:11:39 2021

@author: mustansar
"""

python train_mpii.py \
    --arch=hg2 \
    --image-path=/home/jameel.hassan/Documents/CV701/Assignment4/mine/data/MPII/mpii_human_pose_v1/images/ \
    --checkpoint=checkpoint/hg_separable \
    --epochs=20 \
    --train-batch=32 \
    --workers=24 \
    --test-batch=24 \
    --lr=1e-3 \
    --schedule 15 17 \
    --separable True