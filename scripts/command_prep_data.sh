#/bin/bash

python kitti/prepare_data.py \
  --rgb-det-file ./kitti/samples/demo/rgb_detection_demo.txt \
  --split-name demo \
  --output-file ./kitti/samples/frustum_carpedcyc_demo_rgb_detection.pickle \
  --input-dir ./kitti/samples/
