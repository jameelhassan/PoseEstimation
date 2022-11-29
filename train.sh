#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:11:39 2021

@author: mustansar
"""

python train_mpii.py \
    --arch=hg2 \
    --image-path=../mine/data/MPII/mpii_human_pose_v1/images/ \
    --checkpoint=checkpoint/_FinalCall/hg_48-84_FullySep_ResConcat_PercResLossFinalTest_5e-3_BS18 \
    --epochs=25 \
    --train-batch=18 \
    --workers=24 \
    --test-batch=32 \
    --lr=5e-3 \
    --schedule 17 \
    --separable False