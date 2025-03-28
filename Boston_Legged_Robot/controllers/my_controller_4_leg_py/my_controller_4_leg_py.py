"""my_controller_4_leg_py controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from interfaces import *
from controller import Robot, Motor, TouchSensor, PositionSensor, InertialUnit, Keyboard
import copy

# create the Robot instance.
robot = Robot()
robot_status = RobotStatus()
devices = Devices(robot)

# get the time step of the current world.
TIME_STEP = int(robot.getBasicTimeStep())
# 滤波系数
ALPHA = 0.5  # new value coeff  1 - ALPHA : old value coeff
OFFSET_LEN = 0.5 # shorten one leg when the other one is landing phase, this is the value to be shorted

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getMotor('motorname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.
    devices.set_shorten_length_RR(0.3)
    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
