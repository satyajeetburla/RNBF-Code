#!/usr/bin/env python

import queue

import numpy as np
import rospy
import tf
from geometry_msgs.msg import TransformStamped
from rospy import Publisher
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray


class ROSSubscriber:
    def __init__(self, extrinsic_calib=None):
        self.queue = queue.Queue(maxsize=1)
        self.crop = False
        rospy.init_node("rnbf", anonymous=True)
        # rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback, queue_size=1)
        # rospy.Subscriber("/kinect_camera/rgb/image_raw", Image, self.rgb_callback, queue_size=1)
        rospy.Subscriber("/head_camera/depth_registered/image_raw", Image, self.depth_callback, queue_size=1)

        # rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback, queue_size=1)
        # rospy.Subscriber("/kinect_camera/depth/image_raw", Image, self.depth_callback, queue_size=1)
        rospy.Subscriber("/head_camera/rgb/image_raw", Image, self.rgb_callback, queue_size=1)

        # rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_dumpster_callback, queue_size=1)
        # rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_states_callback, queue_size=1)
        # rospy.Subscriber("/ground_truth_pose", Odometry, self.pose_callback_new, queue_size=1)
        rospy.Subscriber("/custom_transform", TransformStamped, self.transform_callback)

        self.i = 0
        self.float_publisher = Publisher("/sdf_values", Float32, queue_size=1)
        self.float_publisher_flag = Publisher("/sdf_flag", Float32, queue_size=1)
        self.grad_publisher = Publisher("/grad_val", Float32MultiArray, queue_size=1)      # rospy.spin()\
        self.pts_publisher = Publisher("/pts", Float32MultiArray, queue_size=1)
        self.hes_publisher = Publisher("/hes_val", Float32MultiArray, queue_size=1)
        self.processed_sdf_publisher = Publisher("/processed_sdf_values", Float32MultiArray, queue_size=1)
        self.processed_gradient_publisher = Publisher("/processed_gradient_values", Float32MultiArray, queue_size=1)
        self.processed_data_publisher = Publisher("/processed_data", Float32MultiArray, queue_size=1)


        self.rgb = None
        self.depth = None
        self.pose = np.eye(4)
        self.rgb_time = None
        self.depth_time = None
        self.pose_time = None
        self.sdf = None
        self.grad = None
        self.flag = None

    def rgb_callback(self, msg):
        rgb_np = np.frombuffer(msg.data, dtype=np.uint8)
        rgb_np = rgb_np.reshape(msg.height, msg.width, 3)
        rgb_np = rgb_np[..., ::-1]
        self.rgb = rgb_np
        self.rgb_time = msg.header.stamp

    def depth_callback(self, msg):
        depth_np = np.frombuffer(msg.data, dtype=np.float32)
        depth_np = depth_np.reshape(msg.height, msg.width)

        if self.crop:
            mw = 40
            mh = 20
            depth_np = depth_np[mh:(msg.height - mh), mw:(msg.width - mw)]

        self.depth = depth_np.copy()
        self.depth_time = msg.header.stamp

    def transform_callback(self, msg):
        rotation = [
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
            msg.transform.rotation.w,
        ]
        rotation_matrix = tf.transformations.quaternion_matrix(rotation)

        translation = [
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z,
        ]

        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = rotation_matrix[:3, :3]
        camera_matrix[:3, 3] = translation

        self.pose = camera_matrix
        self.pose_time = msg.header.stamp


    def sdf_callback(self, msg):
        self.sdf = msg.data

    def grad_callback(self, msg):
        self.grad = msg.data

    def flag_callback(self, msg):
        self.flag = msg.data

    def get_latest_data(self):
        if self.depth is None or self.rgb is None or self.pose is None:
            return
        return {
            "image": self.rgb,
            "depth": self.depth,
            "T": self.pose,
        }

    def get_latest_data_move(self):
        if self.sdf is None or self.grad is None or self.flag is None:
            return
        return {
            "sdf": self.sdf,
            "flag": self.flag,
            "grad": self.grad,
        }

    def publish_float_data(self, float_data):
        float_msg = Float32()
        float_msg.data = float_data
        self.float_publisher.publish(float_msg)

    def publish_float_data1(self, float_data):
        float_msg = Float32()
        float_msg.data = float_data
        self.float_publisher_flag.publish(float_msg)

    def publish_float_data2(self, float_data_tensor):
        float_data_list = float_data_tensor.tolist()
        float_array_msg = Float32MultiArray()
        float_array_msg.data = float_data_list
        self.grad_publisher.publish(float_array_msg)

    def publish_float_data3(self, float_data_tensor):
        float_array_msg = Float32MultiArray()
        float_data_list = [item for sublist in float_data_tensor for item in sublist]
        float_array_msg.data = float_data_list
        self.hes_publisher.publish(float_array_msg)

    def publish_float_data4(self, float_data_tensor):
        float_data_list = float_data_tensor.tolist()
        float_array_msg = Float32MultiArray()
        float_array_msg.data = float_data_list
        self.pts_publisher.publish(float_array_msg)

    def publish_processed_sdf(self, sdf_values):
        sdf_array_msg = Float32MultiArray()
        sdf_array_msg.data = sdf_values
        self.processed_sdf_publisher.publish(sdf_array_msg)

    def publish_processed_gradient(self, gradient_values):
        try:
            gradient_list = [item for sublist in gradient_values for item in sublist]

            gradient_array_msg = Float32MultiArray()
            gradient_array_msg.data = gradient_list
            self.processed_gradient_publisher.publish(gradient_array_msg)
        except Exception as e:
            rospy.logerr(f"Failed to publish processed gradient values: {e}")

    def publish_processed_data(self, sdf_values, all_points_tensor_cpu, gradient_values):
        try:
            gradient_list = [item for sublist in gradient_values for item in sublist]
            all_points_list = [item for sublist in all_points_tensor_cpu for item in sublist]

            combined_list = sdf_values + all_points_list + gradient_list

            combined_array_msg = Float32MultiArray()
            combined_array_msg.data = combined_list

            self.processed_data_publisher.publish(combined_array_msg)
        except Exception as e:
            rospy.logerr(f"Failed to publish processed data: {e}")
# ##!/usr/bin/env python
# # rossubscriber.py

# import numpy as np
# import queue
# import rospy
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Pose, PoseStamped
# from gazebo_msgs.msg import ModelStates
# from std_msgs.msg import Float32
# from rospy import Publisher
# import pytransform3d.rotations
# import cv2
# import os
# import queue
# import numpy as np
# from scipy.spatial.transform import Rotation
# import rospy
# import trimesh
# import cv2
# from sensor_msgs.msg import Image  # ROS message type
# from geometry_msgs.msg import Pose, PoseStamped  # ROS message type
# from gazebo_msgs.msg import ModelStates
# from matplotlib import pyplot as plt
# import pytransform3d.rotations
# #target_model_name = "robot"
# from rospy import Publisher
# from rnbf.modules import trainer
# import argparse
# from std_msgs.msg import Float32
# from std_msgs.msg import Float32
# from rospy import Publisher
# target_model_name ="rtab_dumpster"
# target_link_name ="rtab_dumpster::robot_footprint"
# from std_msgs.msg import Float32MultiArray
# from geometry_msgs.msg import Twist
# from scipy.spatial.transform import Rotation
# from gazebo_msgs.msg import LinkStates
# from nav_msgs.msg import Odometry  # Import the appropriate message type
# import tf
# import numpy as np
# from geometry_msgs.msg import TransformStamped

# class ROSSubscriber:
#     def __init__(self, extrinsic_calib=None):
#         self.queue = queue.Queue(maxsize=1)
#         self.crop = False
#         rospy.init_node("rnbf", anonymous=True)
#         # rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback, queue_size=1)
#         rospy.Subscriber("/kinect_camera/rgb/image_raw", Image, self.rgb_callback, queue_size=1)

#         # rospy.Subscriber("/head_camera/depth_registered/image_raw", Image, self.depth_callback, queue_size=1)

#         # rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback, queue_size=1)
#         rospy.Subscriber("/kinect_camera/depth/image_raw", Image, self.depth_callback, queue_size=1)
#         # rospy.Subscriber("/head_camera/rgb/image_raw", Image, self.rgb_callback, queue_size=1)

#         # rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_dumpster_callback, queue_size=1)
#         # rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_states_callback, queue_size=1)
#         # rospy.Subscriber("/ground_truth_pose", Odometry, self.pose_callback_new, queue_size=1)
#         rospy.Subscriber("/custom_transform", TransformStamped, self.transform_callback)

#         self.i = 0
#         self.float_publisher = Publisher("/sdf_values", Float32, queue_size=1)
#         self.float_publisher_flag = Publisher("/sdf_flag", Float32, queue_size=1)
#         self.grad_publisher = Publisher("/grad_val", Float32MultiArray, queue_size=1)      # rospy.spin()\
#         self.pts_publisher = Publisher("/pts", Float32MultiArray, queue_size=1)
#         self.hes_publisher = Publisher("/hes_val", Float32MultiArray, queue_size=1)
#         self.processed_sdf_publisher = Publisher("/processed_sdf_values", Float32MultiArray, queue_size=1)
#         self.processed_gradient_publisher = Publisher("/processed_gradient_values", Float32MultiArray, queue_size=1)
#         self.processed_data_publisher = Publisher("/processed_data", Float32MultiArray, queue_size=1)


#         # rospy.Subscriber("/sdf_values", Float32, self.sdf_callback, queue_size=1)
#         # rospy.Subscriber("/sdf_flag", Float32, self.flag_callback, queue_size=1)
#         # rospy.Subscriber("/grad_val", Float32MultiArray, self.grad_callback, queue_size=1)
#         self.rgb = None
#         self.depth = None
#         self.pose = None
#         #remove 
#         self.pose = np.eye(4)


#         # self.vel_publisher = Publisher("/fetch_vel", Float32MultiArray, queue_size=1)
#         # self.vel_publisher = Publisher("/cmd_vel", Twist, queue_size=10)
#         self.rgb_time = None
#         self.depth_time = None
#         self.pose_time = None

#     def rgb_callback(self, msg):
#         # Your RGB processing code here
#         rgb_np = np.frombuffer(msg.data, dtype=np.uint8)
#         rgb_np = rgb_np.reshape(msg.height, msg.width, 3)
#         rgb_np = rgb_np[..., ::-1]
#         self.rgb = rgb_np
#         self.rgb_time = msg.header.stamp
#         # print("true  rgb----------------------", self.rgb_time)
#         # print("rgb", rgb_np.shape)
#         # try:
#         #     self.queue.put((rgb_np.copy(), self.depth.copy(), self.pose.copy()), block=False,
#         #                    timeout=5)  # Set a timeout of 5 seconds
#         #     print("here---------------")
#         # except queue.Full:
#         #     pass
#         # del rgb_np
#         # pass

#     def depth_callback(self, msg):
#         # Your depth processing code here
#         depth_np = np.frombuffer(msg.data, dtype=np.float32)
#         depth_np = depth_np.reshape(msg.height, msg.width)
#         # print("depth", depth_np.shape)


#         if self.crop:
#             mw = 40
#             mh = 20
#             depth_np = depth_np[mh:(msg.height - mh), mw:(msg.width - mw)]

#         self.depth = depth_np.copy()
#         self.depth_time = msg.header.stamp
#         del depth_np
#         pass

#     def transform_callback(self, msg):
#         # Extract rotation quaternion from the message
#         rotation = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z,
#                     msg.transform.rotation.w]

#         # Convert the quaternion to a rotation matrix
#         rotation_matrix = tf.transformations.quaternion_matrix(rotation)

#         # Extract translation vector from the message
#         translation = [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]

#         # Create a 4x4 transformation matrix by concatenating rotation and translation
#         camera_matrix = np.eye(4)
#         camera_matrix[:3, :3] = rotation_matrix[:3, :3]  # Copy the rotation part
#         camera_matrix[:3, 3] = translation  # Set the translation part

#         # Print the camera transformation matrix
#         # print("Camera Transformation Matrix:")
#               #remove it
#         camera_matrix = np.random.rand(4,4)
#         self.pose = camera_matrix
#         print(" self.pose",  self.pose)
  

#         self.pose_time = msg.header.stamp
#         # print("true ----------------------", self.pose_time)


#     def sdf_callback(self, msg):
#         # Your depth processing code here
#         self.sdf = msg.data
#         pass

#     def grad_callback(self, msg):
#         # Your depth processing code here
#         self.grad = msg.data
#         pass

#     def flag_callback(self, msg):
#         # Your depth processing code here
#         self.flag = msg.data
#         pass
#     def are_timestamps_within_interval(self, time1, time2, time3, interval):
#             # Check if the time difference between any two timestamps is within the interval
#             return abs(time1 - time2).to_sec() <= interval and abs(time1 - time3).to_sec() <= interval and abs(
#                 time2 - time3).to_sec() <= interval

#     def get_latest_data(self):
#         # print("access")
#         # Return the latest data you want to use in get_data
#         if self.depth is None or self.rgb is None or self.pose is None:
#             print("depth", self.depth , "rgb", self.rgb, "pose", self.pose)

#             # print("here",  self.rgb)
#             # print("here",   self.depth)
#             # print("here",   self.pose)

#             return
#         else:
#             # if self.are_timestamps_within_interval(self.rgb_time, self.depth_time, self.pose_time, 0.1):
#                 return {
#                     "image": self.rgb,
#                     "depth": self.depth,
#                     "T": self.pose,
#                     }
#             # else:
#             #     return

#     def get_latest_data_move(self):
#         # Return the latest data you want to use in get_data
#         if self.sdf is None or self.grad is None or self.flag is None:
#             #print("here")
#             return
#         else:
#             return {
#                 "sdf": self.sdf,
#                 "flag": self.flag,
#                 "grad": self.grad,
#                 # "x": self.x,
#             }
#     def publish_float_data(self, float_data):
#         # Publish the float data as a Float32 message
#         float_msg = Float32()
#         float_msg.data = float_data
#         self.float_publisher.publish(float_msg)

#     def publish_float_data1(self, float_data):
#         # Publish the float data as a Float32 message
#         float_msg = Float32()
#         float_msg.data = float_data
#         self.float_publisher_flag.publish(float_msg)

#     def publish_float_data2(self, float_data_tensor):
#         # Create a Float32MultiArray message
#         float_data_list = float_data_tensor.tolist()
#         float_array_msg = Float32MultiArray()

#         # Assign the float data list to the data field of the message
#         float_array_msg.data = float_data_list

#         # Publish the Float32MultiArray message
#         self.grad_publisher.publish(float_array_msg)
#     def publish_float_data3(self, float_data_tensor):
#         # float_data_list = float_data_tensor.tolist()
#         float_array_msg = Float32MultiArray()

#         float_data_list = [item for sublist in float_data_tensor for item in sublist]

#         # Assign the float data list to the data field of the message
#         float_array_msg.data = float_data_list

#         # Publish the Float32MultiArray message
#         self.hes_publisher.publish(float_array_msg)

#     def publish_float_data4(self, float_data_tensor):
#         # Create a Float32MultiArray message
#         float_data_list = float_data_tensor.tolist()
#         float_array_msg = Float32MultiArray()

#         # Assign the float data list to the data field of the message
#         float_array_msg.data = float_data_list

#         # Publish the Float32MultiArray message
#         self.pts_publisher.publish(float_array_msg)

#     def publish_processed_sdf(self, sdf_values):
#         sdf_array_msg = Float32MultiArray()
#         sdf_array_msg.data = sdf_values
#         self.processed_sdf_publisher.publish(sdf_array_msg)

#     # def publish_processed_gradient(self, gradient_values):
#     #     gradient_array_msg = Float32MultiArray()
#     #     gradient_array_msg.data = gradient_values
#     #     self.processed_gradient_publisher.publish(gradient_array_msg)

#     def publish_processed_gradient(self, gradient_values):
#         try:
#             # Ensure gradient_values is a flat list
#             gradient_list = [item for sublist in gradient_values for item in sublist]

#             gradient_array_msg = Float32MultiArray()
#             gradient_array_msg.data = gradient_list
#             self.processed_gradient_publisher.publish(gradient_array_msg)
#         except Exception as e:
#             rospy.logerr(f"Failed to publish processed gradient values: {e}")

#     def publish_processed_data(self, sdf_values, all_points_tensor_cpu, gradient_values):
#         try:
#             # Ensure gradient_values is a flat list
#             gradient_list = [item for sublist in gradient_values for item in sublist]
#             all_points_list = [item for sublist in all_points_tensor_cpu for item in sublist]

#             # Combine sdf_values and gradient_list into one list
#             combined_list = sdf_values + all_points_list + gradient_list

#             # Create a Float32MultiArray message
#             combined_array_msg = Float32MultiArray()
#             combined_array_msg.data = combined_list

#             # Publish the Float32MultiArray message
#             self.processed_data_publisher.publish(combined_array_msg)
#         except Exception as e:
#             rospy.logerr(f"Failed to publish processed data: {e}")
