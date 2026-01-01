## RNBF + CBF Fetch Workspace

This repository is a slimmed-down catkin workspace that combines the ROS RNBF node with our control barrier
function (CBF) controller (`fetch_gazebo_vel/CBF_dubin_new.py`). The instructions below describe how to create
a fresh Python environment, install the required packages, and launch the full pipeline in simulation.

---

## Prerequisites

- Ubuntu 20.04 with ROS Noetic and `catkin` already installed.
- A conda installation (Miniconda/Anaconda).
- A working catkin workspace located at `~/rnbf_gazebo_ws` (or adapt the commands to your location).

---

## 1. Build the catkin workspace

```bash
cd ~/rnbf_gazebo_ws
catkin_make            # build once so ROS messages are generated
```

Re-run `catkin_make` whenever you modify ROS packages.

---

## 2. Create the Python environment (`rnbf`)

```bash
conda create -n rnbf python=3.8
conda activate rnbf
pip install -r requirements.txt     # file lives in src/rnbf/requirements.txt
pip install -e src/rnbf             # installs the RNBF Python package in editable mode
```

> NOTE: the old environment name `isdf` has been retired—make sure all future commands use `conda activate rnbf`.

---

## 3. Launch the full simulation + learning stack

```bash
cd ~/rnbf_gazebo_ws
conda activate rnbf
pip install -e src/rnbf             # ensure the RNBF python package is registered
source devel/setup.bash
roslaunch fetch_gazebo_vel CBF_rnbf_regular.launch
```

This launch file starts:

- `rnbf/listener.py` and `rnbf/train_vis.py` for continual SDF updates/visualization.
- `fetch_gazebo_vel/CBF_dubin_new.py` for motion planning and control.
- Helper nodes for TF echoing and rosbag logging (edit `src/fetch_gazebo_vel/launch/CBF_rnbf_regular.launch` if needed).

### Controller internals

The control node living at `src/fetch_gazebo_vel/scripts/CBF_dubin_new.py` now contains only the logic required for
this launch. It:

- Subscribes to `rossuscriber_rnbf` for SDF, gradients, and robot state.
- Solves a simplified control-barrier-function optimisation (via `cvxpy`) to keep the robot safe with respect to the learned SDF.
- Publishes Hessian diagnostics through `publish_hessian` so you can record/compare real vs estimated curvature.

Feel free to extend the script, but keep the imports and ROS hooks minimal so launch latency stays predictable.

---

## 4. Tips & maintenance

- The `src/rnbf/rnbf/rossuscriber.py` node publishes depth/pose data from Gazebo. Keep the topic names intact unless you also change them in the controller scripts.
- Use `conda env export -n rnbf` to capture extra packages you may install later.
- Before publishing the repo, delete `build/` and `devel/` (they are ignored by `.gitignore`) and rebuild on the target machine.

---

