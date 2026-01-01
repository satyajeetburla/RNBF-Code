cmake_minimum_required(VERSION 3.7.2)
project(fetch_gazebo_vel)

find_package(catkin)
catkin_package()

install(
  PROGRAMS
    scripts/CBF_fetch.py
    scripts/object_pose_ridgeback.py
    scripts/object_pose_fetch.py
    scripts/traj_test.py
    scripts/object_move_random.py
    scripts/CBF_dubin_fetch.py
    scripts/modulation_fetch.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
