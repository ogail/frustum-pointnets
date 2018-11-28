#!/usr/bin/env bash

# run this script with:
# bash ./scripts/command_vis_v1.sh <EXP_ID>

RUN_ID=$1
python kitti/kitti_object.py --pred-dir ./train/inference/$RUN_ID/data --split demo --dataset-dir /home/ogail/workspace/frustum-pointnets/kitti/samples
