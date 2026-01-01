#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped
from gazebo_msgs.msg import ModelStates
from rnbf.modules import trainer
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
import time
a = 0.01
#not req for vision
# obs_name = 'unit_cylinder_'
obs_name = 'unit_box_'

class ROSSubscriber:
    def __init__(self, obs_num, extrinsic_calib=None):
        rospy.init_node("rnbf_track", anonymous=True)
        self.obs_num = obs_num
        self.obs_xy = 40*np.ones((obs_num,2))
        self.obs_theta = np.zeros((obs_num,))

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_callback, queue_size=1)

        rospy.Subscriber("/sdf_values", Float32, self.sdf_callback, queue_size=1)
        rospy.Subscriber("/sdf_flag", Float32, self.flag_callback, queue_size=1)
        rospy.Subscriber("/grad_val", Float32MultiArray, self.grad_callback, queue_size=1)
        rospy.Subscriber("/hes_val", Float32MultiArray, self.hes_callback, queue_size=1)
        rospy.Subscriber("/grad_val", Float32MultiArray, self.grad_callback_full, queue_size=1)
        rospy.Subscriber("/sdf_values", Float32, self.sdf_callback, queue_size=1)
        rospy.Subscriber("/pts", Float32MultiArray, self.pts_callback, queue_size=1)
        rospy.Subscriber("/obs_xy_values", Float32MultiArray, self.obs_xy_callback, queue_size=1)
        rospy.Subscriber('/processed_data', Float32MultiArray, self.processed_data_callback)





        # self.vel_publisher = rospy.Publisher("/fetch_vel", Float32MultiArray, queue_size=1)
        self.vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.hes_publisher = rospy.Publisher("/record_hessian_real", Float32MultiArray, queue_size=10)
        self.ggp_publisher = rospy.Publisher("/record_hessian_estimated", Float32MultiArray, queue_size=10)
        self.grad_publisher = rospy.Publisher("/record_grad_real", Float32MultiArray, queue_size=10)
        self.gp_publisher = rospy.Publisher("/record_grad_estimated", Float32MultiArray, queue_size=10)
        self.dis_publisher = rospy.Publisher("/record_h_real", Float32, queue_size=10)
        self.p_publisher = rospy.Publisher("/record_h_estimated", Float32, queue_size=10)
        self.rob_ang_vel_publisher = rospy.Publisher("/record_rob_ang_vel_real", Float32, queue_size=10)
        self.dt_publisher = rospy.Publisher("/cmd_dt", Float32, queue_size=1)
        self.state_publisher = rospy.Publisher("/cmd_states", Twist, queue_size=1)

        self.sdf=0.0
        self.grad = np.zeros(2,)
        self.grad_full = np.zeros(3, )
        self.pts = np.zeros(3, )
        self.flag = 0.0
        self.rob_x = np.zeros(5,)
        self.hes = np.zeros((2,2))
        self.target_xy = np.zeros((2,))
        self.rob_ang_vel = 0
        self.obs_xy_est = np.zeros((obs_num,2))
        self.rob_theta = 0
        self.rob_xy = np.zeros((2,))
        self.sdf_values_list = []
        self.all_points_list = []
        self.gradient_values_list = []
        self.time = time.time()

    def pose_callback(self, data):
        name = data.name
        pose = data.pose
        twist = data.twist
        index_rob = name.index("_robot")
        # index_rob = name.index("fetch")

        angular_speed = twist[index_rob].angular.z
        speed = np.sqrt(twist[index_rob].linear.x ** 2 + twist[index_rob].linear.y ** 2 )

        self.rob_xy = [pose[index_rob].position.x, pose[index_rob].position.y]
        q = R.from_quat([pose[index_rob].orientation.x, pose[index_rob].orientation.y, pose[index_rob].orientation.z,
                         pose[index_rob].orientation.w])
        e = q.as_euler("xyz")
        self.rob_theta = e[2]
        # if speed * np.cos(self.rob_theta) * twist[index_rob].linear.x < 0:
        #     speed = -speed
        self.rob_x = [pose[index_rob].position.x, pose[index_rob].position.y, speed, e[2], angular_speed]
        # print( self.rob_x )
        # self.rob_states = [speed, angular_speed]

        ##################### commented as no cylinder
        # for i in range(self.obs_num):
        #     index_obs = name.index(obs_name + str(i))
        #     # self.obs_vel_xy[i] = self.obs_xy[i]/self.dt-[pose[index_obs].position.x/self.dt,pose[index_obs].position.y/self.dt]
        #     # self.obs_vel_xy[i] = [twist[index_obs].linear.x, twist[index_obs].linear.y]
        #     self.obs_xy[i] = [pose[index_obs].position.x, pose[index_obs].position.y]
            # q = R.from_quat(
            #     [pose[index_obs].orientation.x, pose[index_obs].orientation.y, pose[index_obs].orientation.z,
            #      pose[index_obs].orientation.w])
            # e = q.as_euler("xyz")
            # self.obs_theta[i] = e[2]
            # self.obs_vel_theta[i] = twist[index_obs].angular.z
    # def pose_callback(self,data):
    #     name = data.name
    #     pose = data.pose
    #     twist = data.twist
    #     index_rob = name.index("_robot")
    #     speed = np.linalg.norm([twist[index_rob].linear.x, twist[index_rob].linear.y])
    #     self.rob_xy = [pose[index_rob].position.x,pose[index_rob].position.y]
    #     q = R.from_quat([pose[index_rob].orientation.x,pose[index_rob].orientation.y,pose[index_rob].orientation.z,pose[index_rob].orientation.w])
    #     e = q.as_euler("xyz")
    #     self.rob_x = [pose[index_rob].position.x,pose[index_rob].position.y,speed,e[2]]
    #     self.rob_ang_vel = twist[index_rob].angular.z
        #
        # for i in range(self.obs_num):
        #     index_obs = name.index("obs"+str(i+1))
        #     self.obs_xy[i] = [pose[index_obs].position.x,pose[index_obs].position.y]
        #     q = R.from_quat([pose[index_obs].orientation.x,pose[index_obs].orientation.y,pose[index_obs].orientation.z,pose[index_obs].orientation.w])
        #     e = q.as_euler("xyz")
        #     self.obs_theta[i] = e[2]
        # pass

    def sdf_callback(self, msg):
        # Your depth processing code here
        self.sdf = msg.data
        pass

    def publish_rob_states(self):
        move_cmd = Twist()
        move_cmd.linear.x = self.rob_x[0]  # x
        move_cmd.angular.z = self.rob_x[1] # y
        move_cmd.angular.x = self.rob_x[2] # orientation
        self.state_publisher.publish(move_cmd)

    def obs_xy_callback(self, msg):
        # Your depth processing code here
        data = msg.data
        self.obs_xy_est = np.array(data[:2])
        pass

    def grad_callback(self, msg):
        # Your depth processing code here
        data = msg.data
        self.grad = np.array(data[:2])
        pass
    def grad_callback_full(self, msg):
        # Your depth processing code here
        data = msg.data
        self.grad_full = np.array(data)
        pass

    def pts_callback(self, msg):
        # Your depth processing code here
        data = msg.data
        self.pts = np.array(data)
        pass

    def hes_callback(self, msg):
        # Your depth processing code here
        data = msg.data
        self.hes = np.array([[data[0], data[1]],[data[3],data[4]]])
        pass

    def flag_callback(self, msg):
        # Your depth processing code here
        self.flag = msg.data
        pass

    def get_latest_data(self):
        # Return the latest data you want to use in get_data
        # rospy.sleep(0.001)
        return {
            "p": self.sdf,
            "flag": self.flag,
            "grad_x_p": self.grad,
            "grad_full": self.grad_full,
            "pts": self.pts,
            "grad_grad_x_p": self.hes,
            "rob_x": self.rob_x,
            "target_xy": self.target_xy,
            "obs_xy": self.obs_xy,
            "obs_theta": self.obs_theta,
            "rob_ang_vel":self.rob_ang_vel,
        }


    def publish_rob_speed(self, speed, angular_velocity,dt):

        # print(speed)

        # print(angular_velocity)

        move_cmd = Twist()
        move_cmd.linear.x = speed # 0.2 m/s
        move_cmd.angular.z = angular_velocity
        self.vel_publisher.publish(move_cmd)



        time_data = Float32()
        time_data.data = self.time-time.time()
        self.dt_publisher.publish(time_data)
        self.time = time.time()

        # rospy.sleep(dt)  # Move for 5 seconds

    def publish_rob_dubin(self,x,u_mod,dt): 
        move_cmd = Twist()
        move_cmd.linear.x = x[2]  
        move_cmd.angular.z = u_mod[1]
        self.vel_publisher.publish(move_cmd)

    def publish_distance(self,dis, _p): 
        distance = Float32()
        distance.data = dis
        self.dis_publisher.publish(distance)
        P = Float32()
        P.data = _p
        self.p_publisher.publish(P)

    def publish_rob_ang_vel(self,_ang_vel): 
        ang_vel = Float32()
        ang_vel.data = _ang_vel
        self.rob_ang_vel_publisher.publish(ang_vel)

    def publish_gradient(self,grad, gp): 
        gradient = Float32MultiArray()
        gradient.data = [grad[0],grad[1]]
        self.grad_publisher.publish(gradient)
        grad_p = Float32MultiArray()
        grad_p.data = [gp[0],gp[1]]
        self.gp_publisher.publish(grad_p)

    def publish_hessian(self,hessian, ggp): 
        hes = Float32MultiArray()
        hes.data = [hessian[0][0],hessian[0][1],hessian[1][0],hessian[1][1]]
        self.hes_publisher.publish(hes)
        grad_grad_p = Float32MultiArray()
        grad_grad_p.data = [ggp[0][0][0],ggp[0][0][1],ggp[0][1][0],ggp[0][1][1]]
        self.ggp_publisher.publish(grad_grad_p)
        # rospy.sleep(0.001)

    def processed_data_callback(self, msg):
        data = msg.data
        try:
            # Extract SDF values (first 9 elements)
            self.sdf_values_list = data[:7]
            all_points = data[7:28]
            self.all_points_list = [all_points[i:i + 3] for i in range(0, len(all_points), 3)]

            # Extract gradient values (remaining elements)
            gradient_data = data[28:]

            self.gradient_values_list = [gradient_data[i:i+3] for i in range(0, len(gradient_data), 3)]

        except Exception as e:
            rospy.logerr(f"Failed to process received data: {e}")

    def get_list_data(self):
            # Return the latest data you want to use in get_data
            # rospy.sleep(0.001)
            return {
                "sdf_list":  self.sdf_values_list,
                "all_point_list": self.all_points_list,
                "grad_list": self.gradient_values_list
            }
