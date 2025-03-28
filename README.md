# Legged-Robots

From 《Legged Robots That Balance》 to DRL

- one-leg robot
  - Done
- 2-leg robot trot
  - Done
  - ![2-leg robot biped](Boston_Legged_Robot/images/2-leg-biped-robot/Biping-in-Three-Dimensions-2Leg.gif)
- 4-leg robot trot
  - TODO
- 4-leg robot pace
  - TODO
- 4-leg robot bound
  - TODO
- 4-leg robot with 12-dof trot/pace/bound
  - TODO
- 4-leg robot with 12-dof with Reinforcement Learning Controller
  - Ongoing

# Part I: Legged Robots
in *Boston_Legged_Robot/world/Hopping-in-Three-Dimensions-2Leg.wbt*

# Part II: DRL Robot
in *Spot_RL/world/Spot.wbt* with controller **PPO**

## software that needed
- Webots (mainly with Python)
  - Why Python?
    - easy to code
    - we don't have a real robot that
    - math and robotics libs
    - AI libs and framework (but not now, in future however)
- Mujoco
  - good for RL/DRL
- C/C++
  - all C/Cpp code are tried to translated to python though;
- ROS/ROS2
- ...


# Update Logging
things before 2024-11-24:
a 2-leg robot which can run (with max speed 2 m/s approximately)

2024-11-24：
skip all the works about 4-legs robot, and start work with 12-dof dog using deep reinforcement learning (that would be a huge work)
- create project and start modeling

2025-03-28:
add an Spot　Ｒｏｂｏｔ　ｕｓｉｎｇ　ＰＰＯ