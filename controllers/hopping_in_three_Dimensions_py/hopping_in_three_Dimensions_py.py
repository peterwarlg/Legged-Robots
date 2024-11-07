"""hopping_in_three_Dimensions_py controller."""

# You may need to import some classes of the controller module. Ex:

from hip_robot import HipRobot

angle = 0.0

if __name__ == '__main__':
    # test
    robot_status = HipRobot()

    while robot_status.robot.step(robot_status.time_step) != -1:
        # robot_status.X_motor.setPosition(angle)
        # val = robot_status.X_motor_position_sensor.getValue()
        # angle = angle + 0.001
        #
        # robot_status.Z_motor.setPosition(0.5)
        robot_status.update_robot_state()
        robot_status.robot_control()
        # print(robot_status.x_dot, robot_status.z_dot)

        pass

# Enter here exit cleanup code.
