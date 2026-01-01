#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

def publish_transform():
    rospy.init_node('tf_listener_node')
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(30)  # Adjust the rate as needed

    transform_pub = rospy.Publisher("/custom_transform", TransformStamped, queue_size=1)

    while not rospy.is_shutdown():
        try:
            # Listen for the transform from "odom" to "kinect_camera_optical"
            transform = tf_buffer.lookup_transform("world", "head_camera_depth_optical_frame", rospy.Time())
            # print("transform",transform)
            # Publish the transform to a custom topic
            transform_pub.publish(transform)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn_throttle(5.0, "Waiting for world->head_camera_depth_optical_frame transform.")

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_transform()
    except rospy.ROSInterruptException:
        pass
