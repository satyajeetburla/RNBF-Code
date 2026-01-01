# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import queue
import numpy as np
from scipy.spatial.transform import Rotation
import rospy
import trimesh
import cv2
from sensor_msgs.msg import Image  # ROS message type
from geometry_msgs.msg import Pose, PoseStamped  # ROS message type
from gazebo_msgs.msg import ModelStates
from matplotlib import pyplot as plt
import pytransform3d.rotations
#target_model_name = "robot"
from rospy import Publisher
from rnbf.modules import trainer
import argparse
from std_msgs.msg import Float32
from std_msgs.msg import Float32
from rospy import Publisher
target_model_name ="rtab_dumpster"

import matplotlib.pyplot as plt
class RNBFNode:
    def __init__(self, queue, crop=False) -> None:
        print("RNBF Node: starting", os.getpid())
        print("Waiting for first frame...")
        print("gazebo------------------- here")

        self.queue = queue
        self.crop = crop

        rospy.init_node("rnbf", anonymous=True)
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback, queue_size=1)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback, queue_size=1)
        #rospy.Subscriber("/natnet_ros/Realsense_Rigid_Body1/pose", PoseStamped, self.pose_callback, queue_size=1)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_callback, queue_size=1)
        self.float_publisher = Publisher("/your_float_topic", Float32, queue_size=1)
        rospy.spin()



    def rgb_callback(self, msg):
        rgb_np = np.frombuffer(msg.data, dtype=np.uint8)
        rgb_np = rgb_np.reshape(msg.height, msg.width, 3)
        rgb_np = rgb_np[..., ::-1]

        if self.depth is None or self.pose is None:
            return

        try:
            self.queue.put(
                (rgb_np.copy(), self.depth.copy(), self.pose.copy()),
                block=False,
            )

        except queue.Full:
            pass


        #print("Received RGB image:", rgb_np.shape)
    def depth_callback(self, msg):
        #depth_np = np.frombuffer(msg.data, dtype=np.uint16)
        depth_np = np.frombuffer(msg.data, dtype=np.float32)
        depth_np = depth_np.reshape(msg.height, msg.width)

        if self.crop:
            mw = 40
            mh = 20
            depth_np = depth_np[mh:(msg.height - mh), mw:(msg.width - mw)]


        self.depth = depth_np.copy()
        del depth_np

    def pose_callback(self, msg):

        model_names = msg.name
        poses = msg.pose
        if target_model_name in model_names:
            index = model_names.index(target_model_name)

            pose = poses[index]
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z
            trans = np.asarray([pose.position.x, pose.position.y, pose.position.z])
            q = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
            theta = -90
            Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
            #
            rot = pytransform3d.rotations.matrix_from_quaternion(q)
            #
            urot = np.dot(rot, Rx)

            camera_transform = np.concatenate((urot, trans.reshape((3, 1))), axis=1)
            camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))
            self.pose = camera_transform

        del camera_transform


