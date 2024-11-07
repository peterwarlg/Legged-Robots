"""my_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import math

import numpy as np

from interfaces import *
from scipy.spatial.transform import Rotation as R
from controller import Robot, Motor, TouchSensor, PositionSensor, InertialUnit
import copy

# create the Robot instance.
robot = Robot()
robot_status = OneLegRobot()

# 初始化
spring_motor: Motor = robot.getMotor("linear motor")
spring_pos_sensor: PositionSensor = robot.getPositionSensor("position sensor")
touch_sensor: TouchSensor = robot.getTouchSensor("touch sensor")
IMU: InertialUnit = robot.getInertialUnit("inertial unit")
X_motor: Motor = robot.getMotor("X rotational motor")
X_motor_position_sensor: PositionSensor = robot.getPositionSensor("X position sensor")
Z_motor: Motor = robot.getMotor("Z rotational motor")
Z_motor_position_sensor: PositionSensor = robot.getPositionSensor("Z position sensor")

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

spring_pos_sensor.enable(timestep)
X_motor_position_sensor.enable(timestep)
Z_motor_position_sensor.enable(timestep)
IMU.enable(timestep)
touch_sensor.enable(timestep)


def set_spring_force(force):
    global spring_motor
    spring_motor.setForce(-force)


def set_X_torque(torque):
    global X_motor
    X_motor.setTorque(torque)


def set_Z_torque(torque):
    global Z_motor
    Z_motor.setTorque(torque)


def get_spring_length() -> float:
    global spring_pos_sensor
    l: float = spring_pos_sensor.getValue()
    return 0.8 - l


def get_X_motor_angle():
    global X_motor_position_sensor
    angle: float = X_motor_position_sensor.getValue()
    return angle * 180.0 / math.pi


def get_Z_motor_angle():
    global Z_motor_position_sensor
    angle: float = Z_motor_position_sensor.getValue()
    return angle * 180.0 / math.pi


def is_foot_touching() -> bool:
    global touch_sensor
    return touch_sensor.getValue()


def get_IMU_Angle() -> EulerAnglesRPY:
    global IMU
    data = IMU.getRollPitchYaw()
    eulerAngle = EulerAnglesRPY()
    eulerAngle.roll = data[0] * 180.0 / math.pi
    eulerAngle.pitch = data[1] * 180.0 / math.pi
    eulerAngle.yaw = data[2] * 180.0 / math.pi
    return eulerAngle


def updateRobotState():
    global robot_status
    # 时钟更新
    hip_robot.system_ms += timestep

    #  足底传感器更新
    hip_robot.is_foot_touching_flg_a = is_foot_touching()

    # IMU，IMU导数，以及旋转矩阵更新
    now_IMU = get_IMU_Angle()
    now_IMU_dot = EulerAnglesRPY()

    now_IMU_dot.roll = (now_IMU.roll - hip_robot.euler_angles.roll) / (0.001 * timestep)
    now_IMU_dot.pitch = (now_IMU.pitch - hip_robot.euler_angles.pitch) / (0.001 * timestep)
    now_IMU_dot.yaw = (now_IMU.yaw - hip_robot.euler_angles.yaw) / (0.001 * timestep)

    hip_robot.euler_angles = now_IMU
    hip_robot.euler_angles_dot.roll = hip_robot.euler_angles_dot.roll * 0.5 + now_IMU_dot.roll * 0.5
    hip_robot.euler_angles_dot.pitch = hip_robot.euler_angles_dot.pitch * 0.5 + now_IMU_dot.pitch * 0.5
    hip_robot.euler_angles_dot.yaw = hip_robot.euler_angles_dot.yaw * 0.5 + now_IMU_dot.yaw * 0.5

    # 更新旋转矩阵
    r = R.from_euler('xzy', [now_IMU.roll, now_IMU.pitch, now_IMU.yaw], degrees=True)
    hip_robot.rotation_matrix_B_under_H_leg_a = np.array(copy.deepcopy(r.as_matrix()))
    hip_robot.rotation_matrix_H_under_B_leg_b = np.transpose(copy.deepcopy(hip_robot.rotation_matrix_B_under_H_leg_a))

    # / *弹簧长度及其导数更新 * /
    now_r: float = get_spring_length()
    now_r_dot = (now_r - hip_robot.joint_space_lxz_a.r) / (0.001 * timestep)
    hip_robot.joint_space_lxz_a.r = now_r
    hip_robot.joint_space_lxz_dot_a.r = hip_robot.joint_space_lxz_dot_a.r * 0.5 + now_r_dot * 0.5  # 一阶低通滤波器

    # /*X、Z关节角度更新*/
    now_X_motor_angle: float = get_X_motor_angle()
    now_X_motor_angle_dot: float = (now_X_motor_angle - hip_robot.joint_space_lxz_a.X_motor_angle) / (0.001 * timestep)
    hip_robot.joint_space_lxz_a.X_motor_angle = now_X_motor_angle
    hip_robot.joint_space_lxz_dot_a.X_motor_angle = hip_robot.joint_space_lxz_dot_a.X_motor_angle * 0.5 + now_X_motor_angle_dot * 0.5  # 一阶低通滤波器

    now_Z_motor_angle = get_Z_motor_angle()
    now_Z_motor_angle_dot = (now_Z_motor_angle - hip_robot.joint_space_lxz_a.Z_motor_angle) / (0.001 * timestep)
    hip_robot.joint_space_lxz_a.Z_motor_angle = now_Z_motor_angle
    hip_robot.joint_space_lxz_dot_a.Z_motor_angle = hip_robot.joint_space_lxz_dot_a.Z_motor_angle * 0.5 + now_Z_motor_angle_dot * 0.5  # // 一阶低通滤波器

    # / *机器人在世界坐标系下水平速度估计更新 * /
    update_xz_dot()
    # / *上次支撑相时间Ts更新 * /
    update_last_Ts()
    # / *更新状态机 * /
    updateRobotStateMachine()
    pass


def forwardKinematics(jointPoint: JointSpaceLXZ) -> np.ndarray:
    # // 转换成弧度制
    Tx = jointPoint.X_motor_angle * math.pi / 180.0
    Tz = jointPoint.Z_motor_angle * math.pi / 180.0
    r = jointPoint.r
    x = r * math.sin(Tz)
    y = -r * math.cos(Tz) * math.cos(Tx)
    z = -r * math.cos(Tz) * math.sin(Tx)
    return np.array([x, y, z])


def update_xz_dot():
    global robot_status
    hip_robot.Point_hat_B_a = forwardKinematics(hip_robot.joint_space_lxz_a)
    # // 转换到{H}坐标系下
    pre_x = hip_robot.Point_hat_H_a[0]
    pre_z = hip_robot.Point_hat_H_a[2]
    hip_robot.Point_hat_H_a = copy.deepcopy(hip_robot.rotation_matrix_B_under_H_leg_a @ hip_robot.Point_hat_B_a)
    now_x = hip_robot.Point_hat_H_a[0]
    now_z = hip_robot.Point_hat_H_a[2]
    # // 求导
    now_x_dot = -(now_x - pre_x) / (0.001 * timestep)
    now_z_dot = -(now_z - pre_z) / (0.001 * timestep)
    # // 滤波
    now_x_dot = hip_robot.pre_x_dot * 0.5 + now_x_dot * 0.5
    now_z_dot = hip_robot.pre_z_dot * 0.5 + now_z_dot * 0.5
    hip_robot.pre_x_dot, hip_robot.pre_z_dot = now_x_dot, now_z_dot
    if (hip_robot.robot_state == COMPRESSION) or (hip_robot.robot_state == THRUST):
        hip_robot.x_dot = now_x_dot
        hip_robot.z_dot = now_z_dot


def update_last_Ts():
    global robot_status
    if not hip_robot.pre_is_foot_touching_flg_a and hip_robot.is_foot_touching_flg_a:
        hip_robot.stance_start_ms = hip_robot.system_ms
    if hip_robot.pre_is_foot_touching_flg_a and not hip_robot.is_foot_touching_flg_a:
        stance_end_ms = hip_robot.system_ms
        hip_robot.Ts = 0.001 * float(stance_end_ms - hip_robot.stance_start_ms)
    hip_robot.pre_is_foot_touching_flg_a = hip_robot.is_foot_touching_flg_a


def updateRobotStateMachine():
    global robot_status
    if hip_robot.robot_state == LOADING:
        if hip_robot.joint_space_lxz_a.r < hip_robot.spring_normal_length * hip_robot.r_threshold:
            hip_robot.robot_state = COMPRESSION
        return
    elif hip_robot.robot_state == COMPRESSION:
        if hip_robot.joint_space_lxz_dot_a.r > 0.0:
            hip_robot.robot_state = THRUST
            return
    elif hip_robot.robot_state == THRUST:
        if hip_robot.joint_space_lxz_a.r > hip_robot.spring_normal_length * hip_robot.r_threshold:
            hip_robot.robot_state = UNLOADING
        return
    elif hip_robot.robot_state == UNLOADING:
        if not hip_robot.is_foot_touching_flg_a:
            hip_robot.robot_state = FLIGHT
        return
    elif hip_robot.robot_state == FLIGHT:
        if hip_robot.is_foot_touching_flg_a:
            hip_robot.robot_state = LOADING
        return
    else:
        return


def robot_control():
    global robot_status
    dx = hip_robot.spring_normal_length - hip_robot.joint_space_lxz_a.r  # 求压缩量
    F_spring = dx * hip_robot.k_spring
    if hip_robot.robot_state == THRUST:
        F_spring += hip_robot.F_thrust
    set_spring_force(F_spring)

    # /*控制臀部扭矩力*/ LOADING和UNLOADING时候，扭矩为0
    if (hip_robot.robot_state == LOADING) or (hip_robot.robot_state == UNLOADING):
        set_X_torque(0.0)
        set_Z_torque(0.0)

    # COMPRESSION和THRUST时候，臀部电机控制身体姿态
    if (hip_robot.robot_state == COMPRESSION) or (hip_robot.robot_state == THRUST):
        Tx = -(-hip_robot.k_pose_p * hip_robot.euler_angles.roll - hip_robot.k_pose_v * hip_robot.euler_angles_dot.roll)
        Tz = -(
                    -hip_robot.k_pose_p * hip_robot.euler_angles.pitch - hip_robot.k_pose_v * hip_robot.euler_angles_dot.pitch)
        set_X_torque(Tx)
        set_Z_torque(Tz)

    # // FLIGHT的时候，控制足底移动到落足点
    if hip_robot.robot_state == FLIGHT:
        r = hip_robot.joint_space_lxz_a.r
        x_f = hip_robot.x_dot * hip_robot.Ts / 2.0 + hip_robot.k_xz_dot * (hip_robot.x_dot - hip_robot.x_dot_desire)
        z_f = hip_robot.z_dot * hip_robot.Ts / 2.0 + hip_robot.k_xz_dot * (hip_robot.z_dot - hip_robot.z_dot_desire)
        y_f = -math.sqrt(r * r - x_f * x_f - z_f * z_f)

        hip_robot.Point_hat_H_desire_a = np.array([x_f, y_f, z_f])
        # 转到{B}坐标系下
        hip_robot.Point_hat_B_desire_a = copy.deepcopy(hip_robot.rotation_matrix_H_under_B_leg_b @ hip_robot.Point_hat_H_desire_a)

        # // 计算期望关节角
        x_f_B = hip_robot.Point_hat_H_desire_a[0]
        y_f_B = hip_robot.Point_hat_H_desire_a[1]
        z_f_B = hip_robot.Point_hat_H_desire_a[2]
        x_angle_desire = math.atan(z_f_B / y_f_B) * 180.0 / math.pi
        z_angle_desire = math.asin(x_f_B / r) * 180.0 / math.pi

        # // 控制关节角
        x_angle = hip_robot.joint_space_lxz_a.X_motor_angle
        z_angle = hip_robot.joint_space_lxz_a.Z_motor_angle
        x_angle_dot = hip_robot.joint_space_lxz_dot_a.X_motor_angle
        z_angle_dot = hip_robot.joint_space_lxz_dot_a.Z_motor_angle
        Tx = -hip_robot.k_leg_p * (x_angle - x_angle_desire) - hip_robot.k_leg_v * x_angle_dot
        Tz = -hip_robot.k_leg_p * (z_angle - z_angle_desire) - hip_robot.k_leg_v * z_angle_dot
        set_X_torque(Tx)
        set_Z_torque(Tz)


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    updateRobotState()
    robot_control()
    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
