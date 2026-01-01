#!/usr/bin/env python
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time
import sys
import cvxpy as cp
import rospy
from rossuscriber_isdf import *
# from CBF_dubin_theory_acceleration_matlab import safe_ctrl_dubin
from math import cos, sin
import csv
from scipy.integrate import solve_ivp
from onMan_approximation import *
# import pandas as pd

# dt = 0.1
# dt = 0.05
dt = 0.5

obs_num_plot = 4
obs_num = 1
velocity_limit = None
convex = True
# speed = 0.6 #good for cylinder
# speed = 0.5
speed = 0.5


# margin = 0.5
margin = 1.0/2
initial_position = [15.896476, -18.052915]
# initial_position = [18.96,-18.92]
initial_orientation = 0
a = 0.3


def h(x_global, xc, orientation):
	h = np.zeros((obs_num,))
	# print("h inside")
	for i in range(obs_num):
		# print("x",x_global)
		# print("xc",xc[i])
		x = x_global_to_relative(x_global, xc[i], orientation[i])
		if convex:
			h[i] = np.sqrt(x[0] ** 2 + x[1] ** 2) - 0.5 - margin
			# h[i] = np.sqrt(x[0] ** 2 + x[1] ** 2) - 0.5 - margin

		else:
			c = 0.6
			b = 0.02
			# h[i] = x[0]**4+x[1]**4 -1*x[0]**2-0.02
			h[i] = np.power((x[0] ** 2 - c) ** 2 + x[1] ** 4, 1 / 4) - np.power(c ** 2 + b, 1 / 4)
	# print(h)
	return h


def x_global_to_relative(x_global, x_c, theta):
	rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	return rotation_matrix.T @ (x_global - x_c)


def x_relative_to_global(x_rel, theta):
	rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	return rotation_matrix @ x_rel


def grad_x_h(x_global, xc, orientation, return_rel=False):
	dhdx = np.zeros((obs_num, 2))
	for i in range(obs_num):
		x = x_global_to_relative(x_global, xc[i], orientation[i])
		if convex:
			dist = np.sqrt(x[0] ** 2 + x[1] ** 2)
			dhdx_rel = [x[0] / dist, x[1] / dist]
		elif obs_num == 1:
			dhdx_rel = [4 * x[0] ** 3 - 10 * x[0], 4 * x[1] ** 3]
		else:
			c = 0.6
			dist = np.sqrt(np.power((x[0] ** 2 - c) ** 2 + x[1] ** 4, 3 / 4))
			dhdx_rel = [0.25 / dist * (4 * x[0] ** 3 - 2 * c * x[0]), 0.25 / dist * 4 * x[1] ** 3]

		if return_rel:
			# print("here")
			dhdx[i] = dhdx_rel
		else:
			dhdx[i] = x_relative_to_global(dhdx_rel, orientation[i])
	return dhdx

def grad_grad_p(x_global,xc,orientation,convex=True,return_rel=False):
	ddg = np.zeros((obs_num,2,2))
	rotation_matrix = np.array([[np.cos(orientation),-np.sin(orientation)],[np.sin(orientation), np.cos(orientation)]])
	partial_xrel_x = rotation_matrix[0][0]
	partial_yrel_x = rotation_matrix[0][1]
	partial_xrel_y = rotation_matrix[1][0]
	partial_yrel_y = rotation_matrix[1][1]
	ddg_relxx = None
	ddg_relyy = None
	ddg_relxy = None
	for i in range(obs_num):
		x = x_global_to_relative(x_global,xc[i],orientation[i])
		if convex:
			dist = np.sqrt(x[0]**2+x[1]**2)
			dist32 = np.power(x[0]**2+x[1]**2,3/2)
			ddg_relxx = 1/dist-x[0]**2/dist32
			ddg_relyy = 1/dist-x[1]**2/dist32
			ddg_relxy = -x[0]*x[1]/dist32
		elif obs_num==1:
			dist = np.power((x[0]**2-2.5)**2+x[1]**4,3/4)
			dhdx_rel = [4*x[0]**3 - 10*x[0], 4*x[1]**3]
		else:
			dhdx_rel = [4*x[0]**3 - 2*x[0], 4*x[1]**3]
		if return_rel:
			ddg[i] = [[ddg_relxx,ddg_relxy],[ddg_relxy,ddg_relyy]]
		else:
			ddg[i][0][0] = ddg_relxx*partial_xrel_x[i]**2+ddg_relxy*partial_xrel_x[i]*partial_yrel_x[i]
			ddg[i][0][1] = ddg_relxx*partial_xrel_x[i]*partial_xrel_y[i]+ddg_relxy*partial_xrel_x[i]*partial_yrel_y[i]
			ddg[i][1][0] = ddg_relyy*partial_yrel_y[i]*partial_yrel_x[i]+ddg_relxy*partial_yrel_y[i]*partial_xrel_x[i]
			ddg[i][1][1] = ddg_relyy*partial_yrel_y[i]**2+ddg_relxy*partial_yrel_y[i]*partial_xrel_y[i]

	return ddg

def grad_t_h(x, obs_pos, obs_orientation, obs_linear_vel, obs_angular_vel):
	_grad_x_h = grad_x_h(x, obs_pos, obs_orientation, return_rel=True)
	grad_t_h = np.zeros((obs_num,))
	_grad_x_rel_t = grad_x_rel_t(x, obs_pos, obs_orientation, obs_linear_vel, obs_angular_vel)
	for i in range(obs_num):
		grad_t_h[i] = _grad_x_h[i] @ _grad_x_rel_t[i]
	return grad_t_h


def grad_x_rel_t(x, XC, Theta, Grad_xc_t, Grad_theta_t):
	X_minus_XC = x - XC
	dx_rel_dt = np.zeros((obs_num, 2))
	for i in range(obs_num):
		theta = Theta[i]
		x_minus_xc = X_minus_XC[i]
		grad_theta_t = Grad_theta_t[i]
		grad_xc_t = Grad_xc_t[i]
		matrix1 = np.array([[-np.sin(theta), np.cos(theta)], [-np.cos(theta), -np.sin(theta)]])
		matrix2 = np.array([[-np.cos(theta), -np.sin(theta)], [np.sin(theta), -np.cos(theta)]])
		dx_rel_dt[i] = x_minus_xc @ matrix1.T * grad_theta_t + grad_xc_t @ matrix2.T
	return dx_rel_dt


def compute_orientation(v_xy, negative_velocity=False):
	if negative_velocity == True:
		v_xy = -v_xy
	if v_xy[0]:
		if v_xy[0] > 0:
			theta = np.arctan(v_xy[1] / v_xy[0])
		else:
			theta = np.pi + np.arctan(v_xy[1] / v_xy[0])

		if theta > np.pi:
			theta = theta - 2 * np.pi
	else:
		theta = np.sign(v_xy[1]) * np.pi / 2

	return theta


def compute_orientation_subtraction(rad_start, rad_end):
	if abs(rad_start) > np.pi or abs(rad_end) > np.pi:
		print("orientation reduction error!!")
		while rad_end > np.pi:
			rad_end = rad_end - 2 * np.pi
		while rad_end < -np.pi:
			rad_end = rad_end + 2 * np.pi
		while rad_start > np.pi:
			rad_start = rad_start - 2 * np.pi
		while rad_start < -np.pi:
			rad_start = rad_start + 2 * np.pi
	difference = rad_end - rad_start
	if abs(difference) > np.pi:
		difference = -np.sign(difference) * (2 * np.pi - abs(difference))
	return difference


def compute_orientation_addition(rad_start, delta_rad):
	rad_end = rad_start + delta_rad
	while rad_end > np.pi:
		rad_end = rad_end - 2 * np.pi
	while rad_end < -np.pi:
		rad_end = rad_end + 2 * np.pi
	return rad_end


def nominal_ctrl(x, target_xy):
    a = np.array([[-1,0],[0,-1]])
    v_xy = a@(x[:2]-target_xy)
    v = np.linalg.norm(v_xy)
    theta_pre = x[2]
    if v > speed:
        v_xy = speed*v_xy/v
        v = speed
    theta_pos = compute_orientation(v_xy, negative_velocity = False)
    delta_theta_pos = compute_orientation_subtraction(theta_pre,theta_pos)
    # if delta_theta_pos<np.pi and delta_theta_pos>0.5*np.pi:
    #   delta_theta_pos = delta_theta_pos-2*np.pi
    # if x[0]>x[1] and delta_theta_pos>0:
    #     delta_theta_pos = delta_theta_pos-2*np.pi
    # if x[0]<x[1] and delta_theta_pos<0:
    #     delta_theta_pos = delta_theta_pos+2*np.pi
    return [v, np.clip(delta_theta_pos/dt,-0.5,0.5)]

# larger alpha means larger safety boundary as well as more dramatic reaction, could lead to bounding back and stop sometimes
# def f():
#     return np.zeros([2, 1])
#
#
# def g():
#     return np.eye(2)

def f(x):
	return np.zeros([3,1])

def g(x):
	# print(x.shape)
	# print(a)
	g = np.zeros((3,2))
	g[0][0] = cos(x[2])
	g[0][1] = -a*sin(x[2])
	g[1][0] = sin(x[2])
	g[1][1] = a*cos(x[2])
	g[2][1] = 1

	return g

def alpha(x):
	# return 10*(np.exp(x)-1)
	# return  0.5*x
	# return 5*np.power(x,3)
	# return 5*np.power(x,2)
	# return 30* np.power(x,3)

	# return 5* np.multiply(x,x)
	# return 20 * np.multiply(x,x)
	# return 2 * np.power(x,6) #mod1
	# return 30 *np.power(x,4) #mod1
	# return  np.power(x,5) #mod1 #works for p2
	# return np.power(x,5)
	return 15* np.power(x,5) # works for p3
	# return 30* np.power(x,5) # works for p3

	# return 30*np.power(x,8) #mod1



	# return 5*np.power(x,8) # worked for cube
	# return  np.power(x, 1)  # worked for cube


# def safe_ctrl(x, xc, u_nom):
#     # xc = np.array([[9.37, 0]]).reshape((obs_num, 2))
#     p = h(x, xc, np.zeros((obs_num,)))
#     # print("h", p)
#     _grad_x_p = grad_x_h(x, xc, np.zeros((obs_num,)), return_rel=False)
#     u_mod = cp.Variable(len(u_nom))
#     obj = cp.Minimize(cp.sum_squares(u_mod - u_nom))
#     dx = f() + g() @ u_mod
#     dth = 0
#
#     if velocity_limit == None:
#         constraints = [_grad_x_p @ dx + alpha(p) >= 0]
#     elif len(velocity_limit) == 1:
#         constraints = [_grad_x_p @ dx + dth + alpha(p) >= 0] + [cp.sum_squares(u_mod) <= velocity_limit[0] ** 2]
#     elif len(velocity_limit) == 2:
#         constraints = [_grad_x_p @ dx + dth + alpha(p) >= 0] + [u_mod - velocity_limit <= 0] + [
#             u_mod + velocity_limit >= 0]
#     # dth = grad_t_h(x,t)
#     # constraints = [grad_x_h(x,t) @ dx + dth + alpha(h(x,t)) >= 0] + [cp.sum_squares(u_mod)<=velocity_limit**2]
#     prob = cp.Problem(obj, constraints)
#     # print(dh(x,t) @ dx)
#     prob.solve()
#     if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
#         return (u_mod.value, (prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)))
#         if (cp.sum_squares(u_mod) < 0.05 and cp.sum_squares(x[0:2]) > 0.5):
#             rospy.loginfo("no solution")
#     else:
#         # rospy.loginfo(prob.status)
#         return (np.array([0, 0]), prob.status)

def safe_ctrl(x,u_nom,_h, _grad_x_h):
	# _h = h_standard(x[:2])-1
	# _grad_x_h, _e_vec = grad_x_h_standard([[x[0],x[1]]],False)
	# _alpha, _iter = alpha_iter()
	# e_vec = om.geodesic_approx_phi(x[:2].reshape((2,1)), _e_vec.reshape((2,1)), _alpha, _iter,_gpis())
	# v_b_e = e_vec.flatten()@dx[:2]

	u_mod = cp.Variable(len(u_nom))
	dth = 0
	r = np.array([[np.cos(-np.pi / 2), -np.sin(-np.pi / 2)], [np.sin(-np.pi / 2), np.cos(-np.pi / 2)]])
	_grad_x_h = _grad_x_h/np.linalg.norm(_grad_x_h)
	e_vec = r @ _grad_x_h
	dx = f(x) + g(x) @ u_mod
	v_b_e = e_vec.flatten()@dx[:2]


	# _h, _grad_x_h,_,_ = om.h_grad_standard(x[:2].reshape(1,2))
	#0.5,1
	#

	if _h< 1:
		# rospy.loginfo("here")
		obj = cp.Minimize((u_mod[0] - u_nom[0])**2+1E-4*(u_mod[1] - u_nom[1])**2)
		constraints = [_grad_x_h.flatten() @ dx[:2] + dth + alpha(_h)>=0] +[u_mod[0]>=-0.5]+[u_mod[0]<=1]+[u_mod[1]<=1.5]+[u_mod[1]>=-1.5]
	# + [v_b_e>=0.1])
	elif _h<1.5:
		obj = cp.Minimize((u_mod[0] - u_nom[0])**2+(u_mod[1] - u_nom[1])**2)
		constraints = [_grad_x_h.flatten() @ dx[:2] + dth + alpha(_h)>=0] +[u_mod[0]>=-0.5]+[u_mod[0]<=1]+[u_mod[1]<=1.5]+[u_mod[1]>=-1.5]
		# + [v_b_e>=0.1]
	else:
		obj = cp.Minimize((u_mod[0] - u_nom[0])**2+(u_mod[1] - u_nom[1])**2)
		constraints = [_grad_x_h.flatten() @ dx[:2] + dth + alpha(_h)>=0] +[u_mod[1]<=1.5]+[u_mod[1]>=-1.5] +[u_mod[0]>=-0.5]+[u_mod[0]<=1]
	prob = cp.Problem(obj, constraints)
	# print(dh(x,t) @ dx)
	try:
		prob.solve()
	except cp.error.SolverError:
		prob.solve(solver=cp.SCS)
	except:
		prob.solve(solver=cp.ECOS)

	if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
		return (u_mod.value, (prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)))
	else:
		print(prob.status)
		print(_h)
		# print(e_vec)
		# print(v_b_e_nom)
		return np.zeros(2,), prob.status

def obst_calc_pos(pts, grad, sdf, flag):
	x, y = pts[0], pts[1]
	dx, dy = grad[0], grad[1]
	r = 0.5
	norm = np.sqrt(dx ** 2 + dy ** 2)
	# print("norm",norm)
	dx_norm, dy_norm = dx / norm, dy / norm
	sdf = sdf - 0.5
	# Calculate the center of the cylinder
	x_center = x - dx_norm * (sdf + r)
	y_center = y - dy_norm * (sdf + r)

	return np.array([x_center, y_center]).reshape((obs_num,2))


# def safe_ctrl_theory(x,obs_xy,obs_theta,u_nom):
# 	u_mod = cp.Variable(len(u_nom))
# 	obj = cp.Minimize(cp.sum_squares(u_mod - u_nom))
# 	dx = f() + g() @ u_mod
# 	print("inside safe ctrl theory")
#
# 	if velocity_limit == None:
# 		constraints = [grad_x_h(x,obs_xy, obs_theta,convex=convex,return_rel=False) @ dx + alpha(h(x,obs_xy,obs_theta,convex=convex)) >= 0]
# 	elif len(velocity_limit)==1:
# 		constraints = [grad_x_h(x,t,convex=convex,return_rel=False) @ dx+ alpha(h(x,t,convex=convex)) >= 0] + [cp.sum_squares(u_mod)<=velocity_limit[0]**2]
# 	elif len(velocity_limit)==2:
# 		constraints = [grad_x_h(x,t,convex=convex,return_rel=False) @ dx+ alpha(h(x,t,convex=convex)) >= 0] + [u_mod-velocity_limit<=0] +[u_mod+velocity_limit>=0]
# 	prob = cp.Problem(obj, constraints)
#
# 	prob.solve()
# 	if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
# 		return (u_mod.value, (prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)))
# 		if(cp.sum_squares(u_mod)<0.05 and cp.sum_squares(x[0:2])>0.5):
# 			rospy.loginfo("no solution")
# 	else:
# 		## Commented here
# 		# rospy.loginfo(prob.status)
# 		return (np.array([0,0]), prob.status)

def cbf(x, p, grad_p,  target):
	# global t
	# print(x)
	# flag = 1
	u_nom = nominal_ctrl(x, target)
	# print("x, target",x, target)
	# print(u_nom)
	# print("flag",flag)
	# t = t + 1
	# velocity = u_nom
	velocity, prob_status = safe_ctrl(x, u_nom, p, grad_p)


	# return u_nom[0], u_nom[1]
	return velocity[0], velocity[1], prob_status

	# if flag == 1:
	# 	# print("loop flag 1")
	# 	# print("xc in flag 1", xc)
	# 	velocity, _ = safe_ctrl(x[:2], u_nom, p, gradp)
	# 	# velocity , rob_position= safe_ctrl_dubin(x,xc,target_xy,t)
	# 	xc_leg = xc
	# 	# print(velocity)
	# 	# rospy.loginfo(velocity)
	# 	return velocity[0], velocity[1],rob_position, xc_leg
	# elif isIntial:
	# 	#     # print("loop flag isIntial - nominal control")
	# 	# else:
	#
	# 	velocity = u_nom
	# 	position = velocity * dt + x[:2]
	#
	# 	new_orientation_pos = compute_orientation(velocity, negative_velocity=False)
	# 	new_orientation_neg = compute_orientation(velocity, negative_velocity=True)
	# 	rob_ang_velocity_pos = compute_orientation_subtraction(x[3], new_orientation_pos) / dt
	# 	rob_ang_velocity_neg = compute_orientation_subtraction(x[3], new_orientation_neg) / dt
	# 	rob_ang_velocity = rob_ang_velocity_pos
	# 	rob_speed = np.linalg.norm(velocity)
	# 	# print(rob_speed)
	# 	return rob_speed, rob_ang_velocity/2, position,xc_leg
	# else:
	# 	# print("loop flag o")
	# 	xc = xc_leg
	# 	# print("xc in flag 0", xc)
	# 	velocity, rob_position = safe_ctrl_dubin(x, xc, target_xy, t)
	# 	return velocity[0], velocity[1], rob_position,xc_leg


# def cbf_theory(x,obs_xy,obs_theta,target_xy):

# 	u_nom = nominal_ctrl(x,target_xy)
# 	velocity,_ = safe_ctrl_theory(x,obs_xy,obs_theta,u_nom)
# 	position = velocity*dt + x

# 	if (h(x,obs_xy,obs_theta,convex=convex)<0).any():
# 		rospy.loginfo("collision")


# 	return velocity,position


# def plot_robot_trajectory(rob_position_list, obs_xy):
#             fig, ax = plt.subplots()
#             plt.plot(rob_position_list[:, 0], rob_position_list[:, 1], label='Robot Trajectory', color='blue',
#                      linewidth=2)
#             # for i in range(obs_num_plot):
#             #     r = 0.5
#             #     _xc = obs_xy[i]
#             #     theta = np.linspace(0, 2 * np.pi, 100)
#             #     x = np.cos(theta) * r + _xc[0]
#             #     y = np.sin(theta) * r + _xc[1]
#             #     plt.plot(x, y, 'k', linewidth=0.5)
#             #     plt.fill(x, y, color='orange', alpha=0.3)
#
#             for _xc in obs_xy[:obs_num_plot]:
#                 r = 0.5  # radius of obstacles
#                 theta = np.linspace(0, 2 * np.pi, 100)  # 360 degrees in radians
#                 x = np.cos(theta) * r + _xc[0]
#                 y = np.sin(theta) * r + _xc[1]
#                 plt.plot(x, y, 'go', linewidth=1.5)
#                 plt.fill(x, y, color='orange', alpha=0.5)  # Orange fill for initial obstacle
#
#             # Adding only the borders for additional obstacles if flag was 1
#             for _xc in obs_xy[obs_num_plot:]:
#                 r = 0.5  # radius of obstacles
#                 theta = np.linspace(0, 2 * np.pi, 100)  # 360 degrees in radians
#                 x = np.cos(theta) * r + _xc[0]
#                 y = np.sin(theta) * r + _xc[1]
#                 plt.plot(x, y, 'k', linewidth=0.5)
#
#             from matplotlib.patches import Patch
#
#             obstacle_patch = Patch(color='orange', label='Obstacle', alpha=0.3)
#             plt.legend(
#                 handles=[obstacle_patch, plt.Line2D([0], [0], color='blue', linewidth=2, label='Robot Trajectory')])
#             ax.set_aspect('equal', adjustable='box')
#             plt.title('Robot Navigation with Obstacles')
#             plt.xlabel('X Coordinate')
#             plt.ylabel('Y Coordinate')
#             plt.grid(True)
#             plt.show()

# def plot_robot_trajectory(rob_position_list, obs_xy ):
#     import mplcursors
#     obs_num_plot = 4
#     fig, ax = plt.subplots()
#     plt.plot(rob_position_list[:, 0], rob_position_list[:, 1], label='Robot Trajectory', color='blue', linewidth=2)
#
#     # Plot and fill obstacles with interaction
#     for index, _xc in enumerate(obs_xy):
#         r = 0.5  # radius of obstacles
#         theta = np.linspace(0, 2 * np.pi, 100)  # 360 degrees in radians
#         x = np.cos(theta) * r + _xc[0]
#         y = np.sin(theta) * r + _xc[1]
#         plot = plt.fill(x, y, 'orange', alpha=0.5 if index < obs_num_plot else 0.3)
#         plt.plot(x, y, 'go' if index < obs_num_plot else 'k', linewidth=1.5 if index < obs_num_plot else 0.5)
#         cursor = mplcursors.cursor(plot, hover=True)
#         cursor.connect("add", lambda sel: sel.annotation.set_text(f'Obstacle {index}'))
#
#     plt.legend([
#         plt.Line2D([0], [0], color='orange', label='Initial Obstacles', alpha=0.5),
#         plt.Line2D([0], [0], color='black', label='Additional Obstacles', alpha=0.3),
#         plt.Line2D([0], [0], color='blue', linewidth=2, label='Robot Trajectory')])
#     ax.set_aspect('equal', adjustable='box')
#     plt.title('Robot Navigation with Obstacles')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.grid(True)
#     plt.show()
#
#
# def plot_robot_trajectory(rob_position_list, obs_xy):
#     fig, ax = plt.subplots()
#     # Plotting robot trajectory
#     robot_traj, = plt.plot(rob_position_list[:, 0], rob_position_list[:, 1], label='Robot Trajectory', color='blue', linewidth=2)
#
#     # Placeholder lists for handles and labels for the legend
#     handles = [robot_traj]
#     labels = ['Robot Trajectory']
#
#     # Assuming obs_xy is an array of obstacle positions
#     for index, _xc in enumerate(obs_xy):
#         r = 0.5  # radius of obstacles
#         theta = np.linspace(0, 2 * np.pi, 100)
#         x = np.cos(theta) * r + _xc[0]
#         y = np.sin(theta) * r + _xc[1]
#         if index < 5:  # Example: different handling for the first obstacle
#             obs_patch, = plt.plot(x, y, 'go-', linewidth=1.5)
#             plt.fill(x, y, color='orange', alpha=0.5)
#             handles.append(obs_patch)
#             labels.append('Obstacle')
#         else:
#             plt.plot(x, y, 'k', linewidth=0.5)
#
#     # Creating legend with explicit handles and labels
#     plt.legend(handles, labels)
#     ax.set_aspect('equal', adjustable='box')
#     plt.title('Robot Navigation with Obstacles')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.grid(True)
#     plt.show()
def plot_robot_trajectory(rob_position_list, obs_xy):
	import mplcursors
	fig, ax = plt.subplots()
	# Plotting the robot's trajectory
	plt.plot(rob_position_list[:, 0], rob_position_list[:, 1], label='Robot Trajectory', color='blue', linewidth=2)

	# Setting up cursor for interactive annotations
	cursor = mplcursors.cursor(hover=True)

	# Looping through obstacle positions
	for index, _xc in enumerate(obs_xy):
		r = 0.5  # radius of the obstacle
		theta = np.linspace(0, 2 * np.pi, 100)  # Creating a circle in radians
		x = np.cos(theta) * r + _xc[0]
		y = np.sin(theta) * r + _xc[1]

		if index < 4:
			# Style for obstacles with index < 4
			outline_color = 'green'  # green outline
			fill_color = 'yellow'  # yellow fill
			alpha_level = 0.7  # more opaque
			plt.plot(x, y, color=outline_color, linewidth=1.5)
			plt.fill(x, y, color=fill_color, alpha=alpha_level)
		else:

			plt.plot(x, y, color='black', linewidth=0.5)

		from matplotlib.patches import Patch
		# Adding hover annotations with the index of the obstacle
		cursor.connect("add", lambda sel, idx=index: sel.annotation.set_text(f'Obstacle {idx}'))

	# Adding legend
	# plt.legend([
	#     plt.Line2D([0], [0], color='blue', linewidth=2, label='Robot Trajectory'),
	#     Patch(color='yellow', label='Obstacles (Index < 4)', alpha=0.7),
	#     Patch(color='black', label='Obstacles (Index >= 4)', alpha=0.3)])
	ax.set_aspect('equal', adjustable='box')
	plt.title('Robot Navigation with Obstacles')
	plt.xlabel('X Coordinate')
	plt.ylabel('Y Coordinate')
	plt.grid(True)
	plt.show()


t = 0
#------------------------------update target here------------------------------------
# target_xy = np.zeros(2,)
target_xy = np.array([-10,-5.5])

#----------------------------------------------------------------------------------
om = OnMan_Approx(target_xy, True, False, radius= 0.9)
if __name__ == "__main__":
	# iteration = 25000
	iteration = 250000
	rob_position_list = np.zeros((iteration - 1, 2))
	rob_position_truth = np.zeros((iteration + 1, 2))
	rob_orientation_truth = np.zeros((iteration + 1,))
	rob_position_truth[0] = initial_position
	rob_orientation_truth[0] = initial_orientation


	ros_subscriber = ROSSubscriber(obs_num_plot)
	rob_speed = 0
	obs_xy = np.zeros((obs_num_plot, 2))
	obs_theta = np.zeros(obs_num_plot, )
	last_non_0_vel = np.zeros((obs_num, 2))
	isIntial = True
	xc_leg = np.array([100, 1000]).reshape((obs_num, 2))

	rospy.sleep(0.5)
	time_after_loop = time.time()
	for i in range(iteration):
		# ros_subscriber.publish_rob_speed(0, 0, 0.001)
		# rospy.sleep(0.02)
		# print(i)

		try:

			# if i <= 20:
			# 	ros_subscriber.publish_rob_speed(0.3, 0, dt)
			# else:
			args = ros_subscriber.get_latest_data()
			flag = args["flag"]
			_p = args["p"]
			_grad_x_p = args["grad_x_p"]
			# target_xy = args["target_xy"]
			rob_x = args["rob_x"]
			obs_xy = args["obs_xy"]
			obs_theta = args["obs_theta"]
			rob_ang_vel = args["rob_ang_vel"]
			pts = args["pts"]
			hessian1 = args["grad_grad_x_p"]
			om.update_xc(obs_xy)
			# obs_xy_est = obst_calc_pos(pts, _grad_x_p, _p, flag)
			# if i == 21:
			# 	obs_pos = obs_xy
			#
			# if flag == 1:
			# 	isIntial = False
			#
			# 	obs_pos =  np.vstack([obs_pos, obs_xy_est])
			rob_x_3 = np.array([rob_x[0], rob_x[1], rob_x[3]])
			# print(rob_x_3)
			rob_speed, rob_ang_velocity , prob_status= cbf(rob_x_3, _p, _grad_x_p, target_xy)
			# rospy.loginfo(rob_ang_velocity)
			# ros_subscriber.publish_rob_speed(rob_speed, rob_ang_velocity, dt)
			# print("------------------------",rob_x[:2])
			ggp = grad_grad_p(rob_x[:2], obs_xy, np.zeros(obs_num,))
			# print("ggp",ggp.shape)
			ros_subscriber.publish_hessian(hessian1, ggp)
			# replace with your code
			# rospy.sleep(dt)
			while True:
				time_before_loop = time.time()
				if time_before_loop - time_after_loop >= dt:
					real_frequency = time_before_loop - time_after_loop
					time_after_loop = time.time()
					if prob_status == "infeasible" or prob_status == "unbounded":
						ros_subscriber.publish_rob_speed(0, 0, dt)
						print("hey")
					else:
						ros_subscriber.publish_rob_speed(rob_speed, rob_ang_velocity, dt)
						ros_subscriber.publish_rob_states()
					break

			# if i > 20:
			# 	rob_position_list[i - 21] = rob_position
		except ValueError as e:
			print("Error occurred:", e)
			break

	# plot_robot_trajectory(rob_position_list, obs_xy)
	# plot_robot_trajectory(rob_position_list, obs_pos)
	# if i<=20:
	#     ros_subscriber.publish_rob_speed(0.3, 0, dt)
	# else:
	#     args = ros_subscriber.get_latest_data()
	#     flag = args["flag"]
	#     _p = args["p"]
	#     _grad_x_p = args["grad_x_p"]
	#     target_xy = args["target_xy"]
	#     rob_x = args["rob_x"]
	#     obs_xy = args["obs_xy"]
	#     obs_theta = args["obs_theta"]
	#     rob_ang_vel = args["rob_ang_vel"]
	#     pts = args["pts"]
	#     obs_xy_est= obst_calc_pos(pts, _grad_x_p, _p, flag)
	#
	#     if flag == 1:
	#         isIntial = False
	#
	#
	#     rob_speed, rob_ang_velocity, rob_position, xc_leg = cbf(rob_x, obs_xy_est, target_xy, flag, dt, isIntial, xc_leg)
	#     # rospy.loginfo(rob_ang_velocity)
	#     ros_subscriber.publish_rob_speed(rob_speed, rob_ang_velocity, dt)
	# rospy.sleep(dt)
	#
	# if i > 20:
	#     rob_position_list[i - 21] = rob_position

