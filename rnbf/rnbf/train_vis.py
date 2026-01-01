#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import cv2
import numpy as np
import torch

import open3d.visualization.gui as gui
from rnbf.modules import trainer
from rnbf.visualisation import rnbf_window
from rossuscriber import ROSSubscriber

SDF_OFFSET = 0.46


def _to_numpy_array(value):
    """Convert incoming tensors to numpy arrays without modifying originals."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_float(value):
    """Return a python float regardless of tensor/array input."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def _project_sdf_to_xy_plane(gradient, sdf):
    """
    Project the gradient onto the x-y plane to compute an adjusted sdf.

    Returns (angle_xy_plane, projected_distance, angle_with_x_axis).
    """
    gradient = _to_numpy_array(gradient).astype(float).reshape(-1)
    sdf = _to_float(sdf)
    magnitude_gradient = np.linalg.norm(gradient)
    if magnitude_gradient == 0:
        return 0.0, 0.0, 0.0

    cos_angle = gradient[2] / magnitude_gradient
    angle_degrees = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    angle_xy_plane = 90 - angle_degrees

    gradient_xy = gradient.copy()
    gradient_xy[2] = 0
    magnitude_xy = np.linalg.norm(gradient_xy)
    if magnitude_xy == 0:
        return angle_xy_plane, sdf, 0.0

    distance_xy = np.cos(np.radians(angle_xy_plane)) * sdf
    angle_with_x_axis = np.degrees(
        np.arccos(np.clip(np.dot(gradient_xy, np.array([1.0, 0.0, 0.0])) / magnitude_xy, -1.0, 1.0))
    )
    return angle_xy_plane, distance_xy, angle_with_x_axis


def optim_iter(trainer, t):
    new_kf = None
    end = False
    finish_optim = trainer.steps_since_frame == trainer.optim_frames
    if trainer.incremental and (finish_optim or t == 0):
        add_new_frame = t == 0 or trainer.check_keyframe_latest()

        if add_new_frame:
            new_frame_id = trainer.get_latest_frame_id()
            size_dataset = 100000
            if new_frame_id >= size_dataset:
                end = True
                print(
                    "**************************************",
                    "End of sequence",
                    "**************************************",
                )
            else:
                frame_data = trainer.get_data([new_frame_id])
                trainer.add_frame(frame_data)

                if t == 0:
                    trainer.last_is_keyframe = True
                    trainer.optim_frames = 200

        if t == 0 or (rnbf_trainer.last_is_keyframe and not add_new_frame):
            new_kf = rnbf_trainer.frames.im_batch_np[-1]
            h = int(new_kf.shape[0] / 6)
            w = int(new_kf.shape[1] / 6)
            new_kf = cv2.resize(new_kf, (w, h))

    losses, step_time, pts, _ = rnbf_trainer.step()
    status = [k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()]
    status = "".join(status) + '-- Step time: {:.2f}  '.format(step_time)
    pts = pts.detach().cpu()
    sdf_value = rnbf_trainer.sdf_fn(pts)
    gradient_val = rnbf_trainer.grad_fn(pts)
    _, projected_sdf, _ = _project_sdf_to_xy_plane(gradient_val, sdf_value)
    adjusted_sdf = projected_sdf - SDF_OFFSET

    ros_subscriber.publish_float_data(_to_float(adjusted_sdf))
    ros_subscriber.publish_float_data4(pts)
    ros_subscriber.publish_float_data2(gradient_val)

    return status, new_kf, end


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ros_subscriber = ROSSubscriber(extrinsic_calib=None)
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description="RNBF.")
    parser.add_argument("--config", type=str, required=True, help="input json config")
    parser.add_argument(
        "-ni",
        "--no_incremental",
        action="store_false",
        help="disable incremental SLAM option",
    )
    args, _ = parser.parse_known_args()  # ROS adds extra unrecongised args
    config_file = args.config
    print(config_file)
    incremental = args.no_incremental

    # init trainer-------------------------------------------------------------
    rnbf_trainer = trainer.Trainer(
        ros_subscriber,
        device,
        config_file,
        incremental=incremental,
    )

    # open3d vis window --------------------------------------------------------
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    w = rnbf_window.RNBFWindow(
        rnbf_trainer,
        optim_iter,
        mono,
    )
    app.run()
