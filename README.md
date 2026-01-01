# RNBF: Real-Time RGB-D Based Neural Barrier Functions for Safe Robotic Navigation

**Satyajeet Das**<sup>1</sup>, **Yifan Xue**<sup>2</sup>, **Haoming Li**<sup>2</sup>, **Nadia Figueroa**<sup>2</sup>  
<sup>1</sup>University of Southern California &nbsp;&nbsp; <sup>2</sup>University of Pennsylvania

[![arXiv](https://img.shields.io/badge/arXiv-2505.02294-b31b1b.svg)](https://arxiv.org/abs/2505.02294)
[![Project Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://satyajeetburla.github.io/rnbf/)

This repository provides a lightweight **ROS (Noetic) catkin workspace** for reproducing the **RNBF-Control** pipeline: online **RNBF** neural SDF reconstruction from posed RGB-D streams + a **CBF-QP** safety controller for Fetch navigation in simulation & real world.

---

## Overview

- **RNBF (Perception):** learns a continuous, first-order differentiable **neural signed distance field (SDF)** online from posed RGB-D input and outputs both **SDF values** and **∇SDF**.
- **Controller (Safety Filter):** a **CBF-QP** controller (`fetch_gazebo_vel/CBF_dubin_new.py`) consumes SDF + ∇SDF to enforce safety.

---


## Repository Layout (Important)

This repo is intended to be used as a **catkin workspace**:

rnbf_ws/
src/ # place all ROS packages here (this repo’s contents go here)


## Prerequisites

- Ubuntu 20.04 with ROS Noetic and `catkin` already installed.
- A conda installation (Miniconda/Anaconda).
- A working catkin workspace located at `~/rnbf_ws` (or adapt the commands to your location).

---

## 1. Build the catkin workspace

```bash
cd ~/rnbf_ws
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
cd ~/rnbf_ws
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

## Citation

If you find this work useful, please cite:

```bibtex
@misc{das2025rnbfrealtimergbdbased,
  title         = {RNBF: Real-Time RGB-D Based Neural Barrier Functions for Safe Robotic Navigation},
  author        = {Satyajeet Das and Yifan Xue and Haoming Li and Nadia Figueroa},
  year          = {2025},
  eprint        = {2505.02294},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2505.02294}
}
```
---
## Acknowledgements

The vision component of this project was inspired by and builds on the
[iSDF](https://github.com/facebookresearch/iSDF) codebase (Facebook Research).

---

## Contact

For questions or collaboration inquiries: **satyajee@usc.edu**