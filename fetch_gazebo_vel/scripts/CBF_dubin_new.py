#!/usr/bin/env python
import time
from math import cos, sin

import cvxpy as cp
import numpy as np
import rospy

from rossuscriber_rnbf import ROSSubscriber


DT = 0.5
OBS_NUM_PLOT = 4
OBS_NUM = 1
SPEED = 0.5
WHEELBASE = 0.3
TARGET_XY = np.array([-10.0, -5.5])


def x_global_to_relative(x_global, x_c, theta):
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rotation_matrix.T @ (x_global - x_c)


def grad_grad_p(x_global, xc, orientation, convex=True, return_rel=False):
    ddg = np.zeros((OBS_NUM, 2, 2))
    rotation_matrix = np.array(
        [[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]]
    )
    partial_xrel_x = rotation_matrix[0][0]
    partial_yrel_x = rotation_matrix[0][1]
    partial_xrel_y = rotation_matrix[1][0]
    partial_yrel_y = rotation_matrix[1][1]
    for i in range(OBS_NUM):
        x = x_global_to_relative(x_global, xc[i], orientation[i])
        if convex:
            dist = np.sqrt(x[0] ** 2 + x[1] ** 2)
            dist32 = np.power(x[0] ** 2 + x[1] ** 2, 3 / 2)
            ddg_relxx = 1 / dist - x[0] ** 2 / dist32
            ddg_relyy = 1 / dist - x[1] ** 2 / dist32
            ddg_relxy = -x[0] * x[1] / dist32
        elif OBS_NUM == 1:
            dist = np.power((x[0] ** 2 - 2.5) ** 2 + x[1] ** 4, 3 / 4)
            ddg_relxx = (4 * x[0] ** 3 - 10 * x[0]) / dist
            ddg_relyy = 4 * x[1] ** 3 / dist
            ddg_relxy = 0
        else:
            ddg_relxx = 4 * x[0] ** 3 - 2 * x[0]
            ddg_relyy = 4 * x[1] ** 3
            ddg_relxy = 0
        if return_rel:
            ddg[i] = [[ddg_relxx, ddg_relxy], [ddg_relxy, ddg_relyy]]
        else:
            ddg[i][0][0] = (
                ddg_relxx * partial_xrel_x[i] ** 2
                + ddg_relxy * partial_xrel_x[i] * partial_yrel_x[i]
            )
            ddg[i][0][1] = (
                ddg_relxx * partial_xrel_x[i] * partial_xrel_y[i]
                + ddg_relxy * partial_xrel_x[i] * partial_yrel_y[i]
            )
            ddg[i][1][0] = (
                ddg_relyy * partial_yrel_y[i] * partial_yrel_x[i]
                + ddg_relxy * partial_yrel_y[i] * partial_xrel_x[i]
            )
            ddg[i][1][1] = (
                ddg_relyy * partial_yrel_y[i] ** 2
                + ddg_relxy * partial_yrel_y[i] * partial_xrel_y[i]
            )
    return ddg


def compute_orientation(v_xy, negative_velocity=False):
    if negative_velocity:
        v_xy = -v_xy
    if v_xy[0]:
        if v_xy[0] > 0:
            theta = np.arctan(v_xy[1] / v_xy[0])
        else:
            theta = np.pi + np.arctan(v_xy[1] / v_xy[0])
        if theta > np.pi:
            theta -= 2 * np.pi
    else:
        theta = np.sign(v_xy[1]) * np.pi / 2
    return theta


def compute_orientation_subtraction(rad_start, rad_end):
    if abs(rad_start) > np.pi or abs(rad_end) > np.pi:
        while rad_end > np.pi:
            rad_end -= 2 * np.pi
        while rad_end < -np.pi:
            rad_end += 2 * np.pi
        while rad_start > np.pi:
            rad_start -= 2 * np.pi
        while rad_start < -np.pi:
            rad_start += 2 * np.pi
    difference = rad_end - rad_start
    if abs(difference) > np.pi:
        difference = -np.sign(difference) * (2 * np.pi - abs(difference))
    return difference


def nominal_ctrl(x, target_xy):
    a = np.array([[-1, 0], [0, -1]])
    v_xy = a @ (x[:2] - target_xy)
    v = np.linalg.norm(v_xy)
    theta_pre = x[2]
    if v > SPEED:
        v_xy = SPEED * v_xy / v
        v = SPEED
    theta_pos = compute_orientation(v_xy, negative_velocity=False)
    delta_theta_pos = compute_orientation_subtraction(theta_pre, theta_pos)
    return [v, np.clip(delta_theta_pos / DT, -0.5, 0.5)]


def f(x):
    return np.zeros([3, 1])


def g(x):
    g_mat = np.zeros((3, 2))
    g_mat[0][0] = cos(x[2])
    g_mat[0][1] = -WHEELBASE * sin(x[2])
    g_mat[1][0] = sin(x[2])
    g_mat[1][1] = WHEELBASE * cos(x[2])
    g_mat[2][1] = 1
    return g_mat


def alpha(x):
    return 15 * np.power(x, 5)


def safe_ctrl(x, u_nom, sdf, grad_sdf):
    grad = np.asarray(grad_sdf[:2], dtype=float)
    grad_norm = np.linalg.norm(grad)
    if grad_norm == 0:
        return u_nom, "zero_gradient"
    grad /= grad_norm

    u_mod = cp.Variable(len(u_nom))
    dx = f(x) + g(x) @ u_mod

    constraints = [
        grad @ dx[:2] + alpha(sdf) >= 0,
        u_mod[0] >= -0.5,
        u_mod[0] <= 1.0,
        u_mod[1] >= -1.5,
        u_mod[1] <= 1.5,
    ]
    if sdf < 1:
        obj = cp.Minimize((u_mod[0] - u_nom[0]) ** 2 + 1e-4 * (u_mod[1] - u_nom[1]) ** 2)
    else:
        obj = cp.Minimize(cp.sum_squares(u_mod - u_nom))

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve()
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS)

    if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return u_mod.value, prob.status
    return np.asarray(u_nom, dtype=float), prob.status


def cbf(x, sdf, grad_sdf, target):
    u_nom = nominal_ctrl(x, target)
    velocity, prob_status = safe_ctrl(x, u_nom, sdf, grad_sdf)
    return velocity[0], velocity[1], prob_status


def main():
    ros_subscriber = ROSSubscriber(OBS_NUM_PLOT)
    rospy.sleep(0.5)
    last_update = time.time()

    while not rospy.is_shutdown():
        try:
            args = ros_subscriber.get_latest_data()
            sdf = args["p"]
            grad_sdf = args["grad_x_p"]
            rob_x = args["rob_x"]
            obs_xy = args["obs_xy"]
            hessian_real = args["grad_grad_x_p"]
        except (TypeError, KeyError):
            continue

        rob_state = np.array([rob_x[0], rob_x[1], rob_x[3]])
        speed_cmd, ang_cmd, prob_status = cbf(rob_state, sdf, grad_sdf, TARGET_XY)

        ggp = grad_grad_p(rob_x[:2], obs_xy, np.zeros(OBS_NUM,))
        ros_subscriber.publish_hessian(hessian_real, ggp)

        elapsed = time.time() - last_update
        if elapsed < DT:
            time.sleep(DT - elapsed)
        last_update = time.time()

        if prob_status in ("infeasible", "unbounded"):
            ros_subscriber.publish_rob_speed(0, 0, DT)
            continue

        ros_subscriber.publish_rob_speed(speed_cmd, ang_cmd, DT)
        ros_subscriber.publish_rob_states()


if __name__ == "__main__":
    main()
