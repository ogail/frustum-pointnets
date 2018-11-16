#/bin/bash
RUN_ID=`date '+%Y_%m_%d__%H_%M_%S'`
EXP_ID=$1
python train/inference.py \
    --gpu 0 \
    --num_point 1024 \
    --model frustum_pointnets_v1 \
    --model_path train/train/$EXP_ID/model.ckpt \
    --output train/inference/$RUN_ID \
    --data_path kitti/samples/
