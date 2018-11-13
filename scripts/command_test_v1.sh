#/bin/bash
RUN_ID=`date '+%Y_%m_%d__%H_%M_%S'`
EXP_ID=$1
python train/test.py --gpu 0 --num_point 1024 --model frustum_pointnets_v1 --model_path train/train/$EXP_ID/model.ckpt --output train/eval/$RUN_ID --data_path kitti/frustum_carpedcyc_val_rgb_detection.pickle --from_rgb_detection --idx_path kitti/image_sets/val.txt --from_rgb_detection
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/eval/$RUN_ID
