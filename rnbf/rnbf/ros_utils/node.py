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

        # rospy.Publisher('/sdf_topic', Float32, queue_size=10)
        # rospy.Publisher('/gradient_topic', Float32MultiArray, queue_size=10)
        # rospy.Publisher('/float_topic', Float32, queue_size=10)
        # self.sdf_pub = rospy.Publisher('/sdf_topic', Float32, queue_size=10)
        # self.gradient_pub = rospy.Publisher('/gradient_topic', Float32MultiArray, queue_size=10)
        # self.float_pub = rospy.Publisher('/float_topic', Float32, queue_size=10)


        rospy.spin()


    import numpy as np

    def quaternion_rotation_matrix(self, x,y,z,w):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                 This rotation matrix converts a point in the local reference
                 frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = x
        q1 = y
        q2 = z
        q3 = w

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]])

        return rot_matrix

    def quaternion_to_rotation_matrix(self, w, x, y, z):


        # Calculate intermediate values
        w2, x2, y2, z2 = w ** 2, x ** 2, y ** 2, z ** 2
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        # Calculate the elements of the rotation matrix
        R = np.array([
            [1 - 2 * y2 - 2 * z2, 2 * xy - 2 * wz, 2 * xz + 2 * wy],
            [2 * xy + 2 * wz, 1 - 2 * x2 - 2 * z2, 2 * yz - 2 * wx],
            [2 * xz - 2 * wy, 2 * yz + 2 * wx, 1 - 2 * x2 - 2 * y2]
        ])

        return R


    def rgb_callback(self, msg):
        rgb_np = np.frombuffer(msg.data, dtype=np.uint8)
        #print(rgb_np.shape/3)
        rgb_np = rgb_np.reshape(msg.height, msg.width, 3)
        rgb_np = rgb_np[..., ::-1]
        #rgb_np = cv2.resize(rgb_np, (1280, 720))
        #rgb_np = cv2.resize(rgb_np, (1280, 720))
        if self.depth is None or self.pose is None:
            return

        try:
            self.queue.put(
                (rgb_np.copy(), self.depth.copy(), self.pose.copy()),
                block=False,
            )
            nonzero_count = np.count_nonzero(rgb_np)
            #print(" in ros Number of nonzero elements:", nonzero_count)

            # Display the Numpy array as Image

            #print("In Queue",rgb_np.copy().shape, self.depth.copy().shape, self.pose.copy())
        except queue.Full:
            pass


        #print("Received RGB image:", rgb_np.shape)
    def depth_callback(self, msg):
        #depth_np = np.frombuffer(msg.data, dtype=np.uint16)
        depth_np = np.frombuffer(msg.data, dtype=np.float32)
        depth_np = depth_np.reshape(msg.height, msg.width)
        #depth_np = cv2.resize(depth_np, (1280, 720))
        if self.crop:
            mw = 40
            mh = 20
            depth_np = depth_np[mh:(msg.height - mh), mw:(msg.width - mw)]

        # self.depth = depth_np.copy()
        # try:
        #     self.queue.put(
        #         (self.rgb_np.copy(), depth_np.copy(), self.pose.copy()),
        #         block=False,
        #     )
        # except queue.Full:
        #     pass
        #depth_np = cv2.resize(depth_np, (1280, 720))
        self.depth = depth_np.copy()
        #print("Received depth image:", depth_np.shape)
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
            #rot = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_matrix()
            #rot = self.quaternion_to_rotation_matrix(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)

            #camera_transform = np.concatenate((rot, trans.reshape((3, 1))), axis=1)
            camera_transform = np.concatenate((urot, trans.reshape((3, 1))), axis=1)
            camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))
            self.pose = camera_transform
           # print(camera_transform)
        del camera_transform

    def publish_float_value(self, value):
        float_msg = Float32()
        float_msg.data = value
        self.float_publisher.publish(float_msg)




class RNBFFrankaNode:
    def __init__(self, queue, crop=False, ext_calib=None) -> None:
        print("RNBF Franka Node: starting", os.getpid())
        print("Waiting for first frame...")

        self.queue = queue
        self.crop = crop
        self.camera_transform = None

        self.cal = ext_calib

        self.rgb, self.depth, self.pose = None, None, None

        self.first_pose_inv = None

        rospy.init_node("rnbf_franka")
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback, queue_size=1)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback, queue_size=1)
        rospy.Subscriber("/natnet_ros/Realsense_Rigid_Body/pose", Pose, self.pose_callback, queue_size=1)
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

    def depth_callback(self, msg):
        depth_np = np.frombuffer(msg.data, dtype=np.uint16)
        depth_np = depth_np.reshape(msg.height, msg.width)

        if self.crop:
            mw = 40
            mh = 20
            depth_np = depth_np[mh:(msg.height - mh), mw:(msg.width - mw)]

        self.depth = depth_np.copy()

    def pose_callback(self, msg):
        position = msg.position
        quat = msg.orientation
        trans = np.asarray([position.x, position.y, position.z])
        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

        camera_transform = np.concatenate((rot, trans.reshape((3, 1))), axis=1)
        camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))
        self.pose = camera_transform
        print(camera_transform)

        del camera_transform


def show_rgbd(rgb, depth, timestamp):
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(rgb)
    plt.title('RGB ' + str(timestamp))
    plt.subplot(2, 1, 2)
    plt.imshow(depth)
    plt.title('Depth ' + str(timestamp))
    plt.draw()
    plt.pause(1e-6)


def get_latest_frame(q):
    # Empties the queue to get the latest frame
    message = None
    while True:
        try:
            message_latest = q.get(block=False)
            if message is not None:
                del message
            message = message_latest

        except queue.Empty:
            break

    return message
