#/bin/bash

EXPERIMENT_ID=`date '+%Y_%m_%d__%H_%M_%S'`
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir train/train/$EXPERIMENT_ID --num_point 1024 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5
