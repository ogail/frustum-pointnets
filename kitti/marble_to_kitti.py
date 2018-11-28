''' Converts Marble data format into KITTI format.

Author: Abdelrahman (Ogail) Elogeel
Date: November 2018
'''
from __future__ import print_function
from glob import glob
from tqdm import tqdm
from shutil import rmtree
from PIL import Image
from pyquaternion import Quaternion
import json
import argparse
import os
import numpy as np

def rm_create(dir):
    if os.path. exists(dir):
        rmtree(dir)
    os.makedirs(dir)


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

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
    for seq_dir in tqdm(glob("{}/*/".format(args.input_dir))):
        for seq_json in tqdm(glob(os.path.join(seq_dir, '*.json'))):
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
                img.save(os.path.join(dir_dict[cam_idx], '{:06d}.png'.format(idx)))

                # parse translation info
                t_x = img_info['velo_to_cam_position']['x']
                t_y = img_info['velo_to_cam_position']['y']
                t_z = img_info['velo_to_cam_position']['z']

                # parse rotation info
                r_w = img_info['velo_to_cam_heading']['w']
                r_x = img_info['velo_to_cam_heading']['x']
                r_y = img_info['velo_to_cam_heading']['y']
                r_z = img_info['velo_to_cam_heading']['z']

                # save transformation from rect cam coord to camera coord (camera intrinsics)
                # 3x4 projection matrix is populated according the following
                #     [fx'  0  cx' Tx]
                # P = [ 0  fy' cy' Ty]
                #     [ 0   0   1   0]
                # Tx = Ty = 0 because this is a monocular camera
                # Reference: http://docs.ros.org/jade/api/sensor_msgs/html/msg/CameraInfo.html
                Tx = Ty = 0
                P = np.array([[fx, 0, cx, Tx], [0, fy, cy, Ty], [0, 0, 1, 0]]).flatten().tolist()
                with open(os.path.join(calib_dir, '{:06d}.txt'.format(idx)), 'a') as fp:
                    fp.write('P{}: {}\n'.format(cam_idx, ' '.join(map(str, P))))

                # save transformation from velodyne coord to camera coord (velodyne extrinsics)
                # 3x4 projection matrix is populated according the following
                #
                # quat = quaternion(r_w, r_x, r_y, r_z)
                # x = quat to 3x3 rot matrix
                #                     [x00  x01  x02  t_x]
                # Tr_velo_to_camX =   [x10  x11  x12  t_y]
                #                     [x20  x21  x22  t_z]
                # References:
                # Creating quat: http://kieranwynn.github.io/pyquaternion/
                # quat to 3x3 rot matrix: http://kieranwynn.github.io/pyquaternion/
                # trans matrix: http://www.it.hiof.no/~borres/j3d/math/threed/p-threed.html
                quat = Quaternion(r_w, r_x, r_y, r_z)
                x = quat.rotation_matrix
                Tr_cam_to_velo = np.array([[x[0,0], x[0,1], x[0,2], t_x],
                                           [x[1,0], x[1,1], x[1,2], t_y],
                                           [x[2,0], x[2,1], x[2,2], t_z]])
                Tr_velo_to_cam = inverse_rigid_trans(Tr_cam_to_velo)
                Tr_velo_to_cam = Tr_velo_to_cam.flatten().tolist()
                with open(os.path.join(calib_dir, '{:06d}.txt'.format(idx)), 'a') as fp:
                    fp.write('Tr_velo_to_cam{}: {}\n'.format(cam_idx, ' '.join(map(str, Tr_velo_to_cam))))

            # flatten the lidar scan data in a list
            pts = []
            for pt in dict['points']:
                pts.append(pt['x'])
                pts.append(pt['y'])
                pts.append(pt['z'])
                pts.append(pt['i'])

            # save lidar scan to disk
            with open(os.path.join(vel_dir, '{:06d}.bin'.format(idx)), 'wb') as fp:
                np.asarray(pts, dtype=np.float32).tofile(fp)

            # save identity matrix for rectification as data already rectified (camera extrinsics)
            R = np.identity(3).flatten().tolist()
            with open(os.path.join(calib_dir, '{:06d}.txt'.format(idx)), 'a') as fp:
                fp.write('R0_rect: {}\n'.format(' '.join(map(str, R))))

            # increase image index
            idx+=1
