''' Converts Marble data format into KITTI format.

Author: Abdelrahman (Ogail) Elogeel
Date: November 2018
'''
from __future__ import print_function
from glob import glob
from tqdm import tqdm
from shutil import rmtree
from PIL import Image
import json
import argparse
import os
import numpy as np

def rm_create(dir):
    if os.path. exists(dir):
        rmtree(dir)
    os.makedirs(dir)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, help='Input directory to sequences top level directory')
    parser.add_argument('--output-dir', type=str, help='Output directory containing converted dataset into KITTI format')
    parser.add_argument('--num-cams', type=int, default=6, help='Number of cameras')
    args = parser.parse_args()

    # create output dirs
    dir_dict = {}
    calib_dir = os.path.join(args.output_dir, 'calib')
    rm_create(calib_dir)
    vel_dir = os.path.join(args.output_dir, 'velodyne')
    rm_create(vel_dir)
    for i in range(args.num_cams):
        dir_dict[i] = os.path.join(args.output_dir, 'image_{}'.format(i))
        rm_create(dir_dict[i])

    idx = 0
    for seq_dir in glob("{}/*/".format(args.input_dir)):
        for seq_json in glob(os.path.join(seq_dir, '*.json')):
            # read frame json file
            with open(seq_json) as fp:
                dict = json.loads(fp.read())
            # loop over each provided camera image
            for cam_idx, img_info in enumerate(dict['images']):
                # parse rectified camera intrinsics calib
                fx = img_info['fx']
                fy = img_info['fy']
                cx = img_info['cx']
                cy = img_info['cy']

                # open rectified image, convert it to png and save it in dst dir
                img = Image.open(os.path.join(seq_dir, img_info['image_url']))
                img.save(os.path.join(dir_dict[cam_idx], '{:05d}.png'.format(idx)))

                # parse translation info
                t_x = img_info['position']['x']
                t_y = img_info['position']['y']
                t_z = img_info['position']['z']

                # parse rotation info
                r_w = img_info['heading']['w']
                r_x = img_info['heading']['x']
                r_y = img_info['heading']['y']
                r_z = img_info['heading']['z']

                # save transformation from rect cam cooord to camera coord
                P = np.eye(3, 4).flatten().tolist()
                with open(os.path.join(calib_dir, '{:05d}.txt'.format(idx)), 'a') as fp:
                    fp.write('P{}: {}\n'.format(cam_idx, ' '.join(map(str, P))))

                # save transformation from velodyne coord to camera coord
                Tr_velo_to_cam = np.eye(3, 4).flatten().tolist()
                with open(os.path.join(calib_dir, '{:05d}.txt'.format(idx)), 'a') as fp:
                    fp.write('Tr_velo_to_cam{}: {}\n'.format(cam_idx, ' '.join(map(str, Tr_velo_to_cam))))

            # flatten the lidar scan data in a list
            pts = []
            for pt in dict['points']:
                pts.append(pt['x'])
                pts.append(pt['y'])
                pts.append(pt['z'])
                pts.append(pt['i'])

            # save lidar scan to disk
            with open(os.path.join(vel_dir, '{:05d}.bin'.format(idx)), 'wb') as fp:
                np.asarray(pts, dtype=np.float32).tofile(fp)

            # save identity matrix for rectification as data already rectified
            R = np.identity(3).flatten().tolist()
            with open(os.path.join(calib_dir, '{:05d}.txt'.format(idx)), 'a') as fp:
                fp.write('R0_rect: {}\n'.format(' '.join(map(str, R))))

            # increase image index
            idx+=1
            exit()
