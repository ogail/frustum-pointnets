#!/usr/bin/env bash

# run this script with:
# bash ./scripts/command_vis_v1.sh <EXP_ID>

EXP_ID=$1
python kitti/kitti_object.py --pred-dir ./train/inference/2018_11_15__09_44_06/data --split demo --dataset-dir /home/ogail/workspace/frustum-pointnets/kitti/samples
