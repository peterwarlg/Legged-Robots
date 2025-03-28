# Legged-Robots

From 《Legged Robots That Balance》 to DRL. This is an implementation and expansion of the book "Legged Robots That Balance";

# Part I: Legged Robots
overview:
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

## 1.1 Two-leg biped robot in Webots
- fIle:  *Boston_Legged_Robot/world/Hopping-in-Three-Dimensions-2Leg.wbt*
- usage: opne .wbt and run

# Part II: Robot using RL in Webots
- requirements: webots r2025a
- miniconda
---
overview:

|robot  |algorithm  |control mode       |      |
|---    |---        |---                |---   |
|Spot   |PPO    |position   |√      |
|Spot   |SAC    |position   |×  |
|Spot   |DDPG   |positon    |   ×   |
|Spot   |TD3    |positon    |   ×   |
|Spot   |PPO    |torque     |√      |
|Spot   |SAC    |torque     |×  |
|Spot   |DDPG   |torque     |   ×   |
|Spot   |TD3    |torque     |   ×   |

## 2.1 Spot robot using PPO in Webots(r2025a):
- file: *Spot_RL/world/Spot.wbt* with controller **PPO**
- usage: open .wbt and select controller PPO

# Part III: Robot using DRL in Mujoco
- requirements: mujoco 3.0+ 
- miniconda

## 3.1 Unitree go2 robot in Mujoco
ongoing

# software that needed
- Webots (mainly with Python)
- pytorch
- Mujoco
  - good for RL/DRL
- Webots
  - GUI and embedded Robot
- C/C++
- ROS/ROS2
- ...


# Update Logging
things before 2024-11-24:
a 2-leg robot which can run (with max speed 2 m/s approximately)

2024-11-24：
skip all the works about 4-legs robot, and start work with 12-dof dog using deep reinforcement learning (that would be a huge work)
- create project and start modeling

2025-03-28:
add Spot robot using PPO in webots

