# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os, sys
# import pika
from scipy.spatial.transform import Rotation as R
import matplotlib.pylab as plt

# import needed only when running with ROS
from rnbf.ros_utils import node
# try:
#     from rnbf.ros_utils import node
# except ImportError:
#     print('Did not import ROS node.')


# Consume RGBD + pose data from ROS node
class ROSSubscriber(Dataset):
    def __init__(
        self,
        extrinsic_calib=None,
        root_dir=None,
        traj_file=None,
        keep_ixs=None,
        rgb_transform=None,
        depth_transform=None,
        noisy_depth=False,
        col_ext=None,
        distortion_coeffs=None,
        camera_matrix=None,
        node_train=None,
    ):
        crop = False
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        self.distortion_coeffs = np.array(distortion_coeffs)
        self.camera_matrix = camera_matrix

        torch.multiprocessing.set_start_method('spawn', force=True)
        self.queue = torch.multiprocessing.Queue(maxsize=1)

        if extrinsic_calib is not None:
            process = torch.multiprocessing.Process(
                target=node.RNBFFrankaNode,
                args=(self.queue, crop, extrinsic_calib),
            ) # subscribe to franka poses 
        else:
            process = torch.multiprocessing.Process(
                target = node_train,
                # target=node.RNBFNode,
                args=(self.queue, crop),
            ) # subscribe to ORB-SLAM backend

        process.start()

    def __len__(self):
        return 1000000000

    def __getitem__(self, idx):
        data = None
        while data is None:
            data = node.get_latest_frame(self.queue)

            if data is not None:
                image, depth, Twc = data
                depth[np.isnan(depth)] = 0
                print("depth range", depth.max(),depth.min())

                sample = {
                    "image": image,
                    "depth": depth,
                    "T": Twc,
                }
                return sample
