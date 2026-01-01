#!/usr/bin/env python3
"""Entry point for running the RNBF training loop inside this workspace."""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from rnbf import visualisation
from rnbf.modules import trainer
from rossuscriber import ROSSubscriber


def train(
    ros_subscriber,
    device,
    config_file,
    chkpt_load_file=None,
    incremental=True,
    show_obj=False,
    update_im_freq=50,
    update_mesh_freq=200,
    grid_dim=200,
    extra_opt_steps=400,
    save_path=None,
):
    """Run the optimisation loop while streaming diagnostics."""

    rnbf_trainer = trainer.Trainer(
        ros_subscriber,
        device,
        config_file,
        chkpt_load_file=chkpt_load_file,
        incremental=incremental,
        grid_dim=grid_dim,
    )

    save = save_path is not None
    checkpoint_path = slice_path = mesh_path = None
    if save:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as outfile:
            json.dump(rnbf_trainer.config, outfile, indent=4)

        if rnbf_trainer.save_checkpoints:
            checkpoint_path = os.path.join(save_path, "checkpoints")
            os.makedirs(checkpoint_path, exist_ok=True)

        if rnbf_trainer.save_slices:
            slice_path = os.path.join(save_path, "slices")
            os.makedirs(slice_path, exist_ok=True)
            rnbf_trainer.write_slices(slice_path, prefix="0.000_", include_gt=True)

        if rnbf_trainer.save_meshes:
            mesh_path = os.path.join(save_path, "meshes")
            os.makedirs(mesh_path, exist_ok=True)

    res = {}
    if rnbf_trainer.do_eval:
        if rnbf_trainer.sdf_eval:
            res["sdf_eval"] = {}
        if rnbf_trainer.mesh_eval:
            res["mesh_eval"] = {}
    vox_res = {} if rnbf_trainer.do_vox_comparison else None
    last_eval = 0

    size_dataset = 1_000_000
    warmup_steps = 800
    break_at = -1
    t = 0

    for step in range(rnbf_trainer.n_steps):
        if t == break_at and not rnbf_trainer.eval_times:
            if save and res:
                if rnbf_trainer.save_slices:
                    rnbf_trainer.write_slices(slice_path)
                kf_list = rnbf_trainer.frames.frame_id[:-1].tolist()
                res["kf_indices"] = kf_list
                with open(os.path.join(save_path, "res.json"), "w", encoding="utf-8") as outfile:
                    json.dump(res, outfile, indent=4)
            break

        finish_optim = rnbf_trainer.steps_since_frame == rnbf_trainer.optim_frames
        if incremental and (finish_optim or t == 0):
            add_new_frame = t == 0 or rnbf_trainer.check_keyframe_latest()
            if add_new_frame:
                new_frame_id = rnbf_trainer.get_latest_frame_id()
                if new_frame_id >= size_dataset:
                    break_at = t + extra_opt_steps
                else:
                    frame_data = rnbf_trainer.get_data([new_frame_id])
                    rnbf_trainer.add_frame(frame_data)
                    if t == 0:
                        rnbf_trainer.last_is_keyframe = True
                        rnbf_trainer.optim_frames = 200

        losses, step_time, pts = rnbf_trainer.step()
        pts = pts.detach().cpu()

        print(f"[RNBF] Computing SDF/grad for optimisation step {step}")
        sdf_value = rnbf_trainer.sdf_fn(pts)
        gradient_val = rnbf_trainer.grad_fn(pts)
        hessian_val = rnbf_trainer.hessian_fn(pts)

        flag_sdf = 3.0 if step < warmup_steps else (0.0 if sdf_value > 1.0 else 1.0)
        ros_subscriber.publish_float_data(sdf_value.item())
        ros_subscriber.publish_float_data1(flag_sdf)
        ros_subscriber.publish_float_data2(gradient_val)
        ros_subscriber.publish_float_data3(np.reshape(hessian_val, (9,)))

        if not rnbf_trainer.live:
            status = " ".join(f"{k}: {losses[k]:.6f}" for k in losses.keys())
            status += f" -- Step time: {step_time:.2f}  "
            print(t, status)

        if not rnbf_trainer.live and update_im_freq and t % update_im_freq == 0:
            display = {}
            scene = None
            obj_scene = None
            obj_slices_viz = None

            rnbf_trainer.update_vis_vars()
            display["keyframes"] = rnbf_trainer.frames_vis()

            if show_obj:
                obj_slices_viz = rnbf_trainer.obj_slices_vis()

            if update_mesh_freq and t % update_mesh_freq == 0:
                scene = rnbf_trainer.draw_3D(
                    show_pc=False,
                    show_mesh=t > 200,
                    draw_cameras=True,
                    camera_view=False,
                    show_gt_mesh=False,
                )
                if show_obj:
                    try:
                        obj_scene = rnbf_trainer.draw_obj_3D()
                    except Exception:  # pylint: disable=broad-except
                        print("Failed to draw mesh")

            if scene is not None:
                display["scene"] = scene
            if show_obj and obj_scene is not None:
                display["obj_scene"] = obj_scene
            if show_obj and obj_slices_viz is not None:
                display["obj_slices"] = obj_slices_viz

            yield display

        t += 1

        if t % 10 == 0 and rnbf_trainer.live:
            vis_rgb, vis_normals, _ = rnbf_trainer.latest_frame_vis()
            stacked = np.hstack((vis_rgb, vis_normals))
            cv2.imshow("RNBF (frame rgb, depth) | (rendered normals, depth)", stacked)
            key = cv2.waitKey(5)
            if key == ord("s"):
                rnbf_trainer.view_sdf()
            if key == ord("c"):
                rnbf_trainer.clear_keyframes()
                t = 0

        if save and rnbf_trainer.save_times and rnbf_trainer.tot_step_time > rnbf_trainer.save_times[0]:
            save_t = f"{rnbf_trainer.save_times.pop(0):.3f}"
            print(
                f"Saving at {save_t}s -- model {rnbf_trainer.save_checkpoints} "
                f"slices {rnbf_trainer.save_slices} mesh {rnbf_trainer.save_meshes}"
            )
            if rnbf_trainer.save_checkpoints:
                torch.save(
                    {
                        "step": t,
                        "model_state_dict": rnbf_trainer.sdf_map.state_dict(),
                        "optimizer_state_dict": rnbf_trainer.optimiser.state_dict(),
                        "loss": losses["total_loss"].item(),
                    },
                    os.path.join(checkpoint_path, f"step_{save_t}.pth"),
                )
            if rnbf_trainer.save_slices:
                rnbf_trainer.write_slices(
                    slice_path,
                    prefix=f"{save_t}_",
                    include_gt=False,
                    include_diff=False,
                    include_chomp=False,
                    draw_cams=True,
                )
            if rnbf_trainer.save_meshes and rnbf_trainer.tot_step_time > 0.4:
                rnbf_trainer.write_mesh(os.path.join(mesh_path, f"{save_t}.ply"))

        if rnbf_trainer.eval_times and rnbf_trainer.tot_step_time > rnbf_trainer.eval_times[0]:
            eval_t = rnbf_trainer.eval_times[0]
            print(f"Starting voxblox evaluation at {eval_t:.2f}s")
            if vox_res is not None:
                vox_res[rnbf_trainer.tot_step_time] = rnbf_trainer.eval_fixed()
                if save:
                    with open(os.path.join(save_path, "vox_res.json"), "w", encoding="utf-8") as outfile:
                        json.dump(vox_res, outfile, indent=4)

        elapsed_eval = rnbf_trainer.tot_step_time - last_eval
        if rnbf_trainer.do_eval and elapsed_eval > rnbf_trainer.eval_freq_s:
            last_eval = rnbf_trainer.tot_step_time - (
                rnbf_trainer.tot_step_time % rnbf_trainer.eval_freq_s
            )

            if rnbf_trainer.sdf_eval and rnbf_trainer.gt_sdf_file is not None:
                visible_res = rnbf_trainer.eval_sdf(visible_region=True)
                obj_errors = rnbf_trainer.eval_object_sdf()
                print(f"Evaluation time: {rnbf_trainer.tot_step_time:.2f}s")
                print("Visible region SDF error: {:.4f}".format(visible_res["av_l1"]))
                print("Objects SDF error: ", obj_errors)
                if not incremental:
                    full_vol_res = rnbf_trainer.eval_sdf(visible_region=False)
                    print("Full region SDF error: {:.4f}".format(full_vol_res["av_l1"]))
                if save:
                    res.setdefault("sdf_eval", {})[t] = {
                        "time": rnbf_trainer.tot_step_time,
                        "rays": visible_res,
                    }
                    if obj_errors is not None:
                        res["sdf_eval"][t]["objects_l1"] = obj_errors

            if rnbf_trainer.mesh_eval:
                acc, comp = rnbf_trainer.eval_mesh()
                print("Mesh accuracy and completion:", acc, comp)
                if save:
                    res.setdefault("mesh_eval", {})[t] = {
                        "time": rnbf_trainer.tot_step_time,
                        "acc": acc,
                        "comp": comp,
                    }
            if save and res:
                with open(os.path.join(save_path, "res.json"), "w", encoding="utf-8") as outfile:
                    json.dump(res, outfile, indent=4)


if __name__ == "__main__":
    ros_subscriber = ROSSubscriber(extrinsic_calib=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(1)
    torch.manual_seed(1)

    parser = argparse.ArgumentParser(description="RNBF.")
    parser.add_argument("--config", type=str, required=True, help="input json config")
    parser.add_argument(
        "-ni",
        "--no_incremental",
        action="store_false",
        help="disable incremental SLAM option",
    )
    parser.add_argument(
        "-hd",
        "--headless",
        action="store_true",
        help="run headless (i.e. no visualisations)",
    )
    args, _ = parser.parse_known_args()

    config_file = args.config
    print(f"Loading RNBF config: {config_file}")
    incremental = args.no_incremental
    headless = args.headless

    show_obj = True
    update_im_freq = 40
    update_mesh_freq = 200
    if headless:
        update_im_freq = update_mesh_freq = None

    save = False
    if save:
        now = datetime.now()
        time_str = now.strftime("%m-%d-%y_%H-%M-%S")
        save_path = os.path.join("../../results/RNBF", time_str)
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = None

    scenes = train(
        ros_subscriber,
        device,
        config_file,
        chkpt_load_file=None,
        incremental=incremental,
        show_obj=show_obj,
        update_im_freq=update_im_freq,
        update_mesh_freq=update_mesh_freq,
        save_path=save_path,
    )

    if headless:
        try:
            while True:
                next(scenes)
        except StopIteration:
            pass
    else:
        import tkinter as tk  # Lazy import to avoid Tk dependency when headless

        width, height = tk.Tk().winfo_screenwidth(), tk.Tk().winfo_screenheight()
        n_cols = 3 if show_obj else 2
        tiling = (1, n_cols)
        visualisation.display.display_scenes(
            scenes,
            height=int(height * 0.5),
            width=int(width * 0.5),
            tile=tiling,
        )
