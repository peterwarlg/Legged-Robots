"""my_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import numpy as np
from interfaces import *
from scipy.spatial.transform import Rotation as R
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


def updateRobotState():
    global robot_status
    # 时钟更新
    robot_status.system_ms += TIME_STEP

    #  足底传感器更新
    robot_status.is_foot_touching_flg_A = devices.is_foot_touching_A()
    robot_status.is_foot_touching_flg_B = devices.is_foot_touching_B()

    # IMU，IMU导数，以及旋转矩阵更新
    imu_now: EulerAnglesRPY = devices.get_IMU_Angle()
    imu_dot_now: EulerAnglesRPY = EulerAnglesRPY()

    imu_dot_now.roll = (imu_now.roll - robot_status.euler_angles.roll) / (0.001 * TIME_STEP)
    imu_dot_now.pitch = (imu_now.pitch - robot_status.euler_angles.pitch) / (0.001 * TIME_STEP)
    imu_dot_now.yaw = (imu_now.yaw - robot_status.euler_angles.yaw) / (0.001 * TIME_STEP)

    robot_status.euler_angles = imu_now
    robot_status.euler_angles_dot.roll = \
        robot_status.euler_angles_dot.roll * (1.0 - ALPHA) + imu_dot_now.roll * ALPHA
    robot_status.euler_angles_dot.pitch = \
        robot_status.euler_angles_dot.pitch * (1.0 - ALPHA) + imu_dot_now.pitch * ALPHA
    robot_status.euler_angles_dot.yaw = \
        robot_status.euler_angles_dot.yaw * (1.0 - ALPHA) + imu_dot_now.yaw * ALPHA

    # 更新旋转矩阵
    r = R.from_euler('xzy', [imu_now.roll, imu_now.pitch, imu_now.yaw], degrees=True)
    robot_status.rotation_matrix_B_under_W = np.array(copy.deepcopy(r.as_matrix()))
    robot_status.rotation_matrix_W_under_B = np.transpose(
        copy.deepcopy(robot_status.rotation_matrix_B_under_W))

    # 弹簧长度及其导数更新
    robot_status.leg_a_joint_space_dot.spring_len = \
        (devices.get_spring_length_A() - robot_status.leg_a_joint_space.spring_len) / (0.001 * TIME_STEP)
    robot_status.leg_b_joint_space_dot.spring_len = \
        (devices.get_spring_length_B() - robot_status.leg_b_joint_space.spring_len) / (0.001 * TIME_STEP)
    robot_status.leg_a_joint_space.spring_len = devices.get_spring_length_A()
    robot_status.leg_b_joint_space.spring_len = devices.get_spring_length_B()

    cur_r_leg_a: float = devices.get_spring_length_A() - devices.get_shorten_length_A()
    cur_r_dot_leg_a = (cur_r_leg_a - robot_status.leg_a_joint_space.r) / (0.001 * TIME_STEP)

    cur_r_leg_b: float = devices.get_spring_length_B() - devices.get_shorten_length_B()
    cur_r_dot_leg_b = (cur_r_leg_b - robot_status.leg_b_joint_space.r) / (0.001 * TIME_STEP)

    robot_status.leg_a_joint_space.r = cur_r_leg_a
    robot_status.leg_a_joint_space_dot.r = robot_status.leg_a_joint_space_dot.r * (
            1.0 - ALPHA) + cur_r_dot_leg_a * ALPHA  # first order filter

    robot_status.leg_b_joint_space.r = cur_r_leg_b
    robot_status.leg_b_joint_space_dot.r = robot_status.leg_b_joint_space_dot.r * (
            1.0 - ALPHA) + cur_r_dot_leg_b * ALPHA  # first order filter

    # X、Z关节角度更新*/
    cur_x_motor_angle_a: float = devices.get_X_motor_angle_A()
    cur_x_motor_angle_dot_a: float = (cur_x_motor_angle_a - robot_status.leg_a_joint_space.X_motor_angle) / (
            0.001 * TIME_STEP)
    robot_status.leg_a_joint_space.X_motor_angle = cur_x_motor_angle_a
    robot_status.leg_a_joint_space_dot.X_motor_angle = robot_status.leg_a_joint_space_dot.X_motor_angle * (
            1.0 - ALPHA) + cur_x_motor_angle_dot_a * ALPHA  # first order filter

    cur_z_motor_angle_a = devices.get_Z_motor_angle_A()
    cur_z_motor_angle_dot_a = (cur_z_motor_angle_a - robot_status.leg_a_joint_space.Z_motor_angle) / (
            0.001 * TIME_STEP)
    robot_status.leg_a_joint_space.Z_motor_angle = cur_z_motor_angle_a
    robot_status.leg_a_joint_space_dot.Z_motor_angle = robot_status.leg_a_joint_space_dot.Z_motor_angle * (
            1.0 - ALPHA) + cur_z_motor_angle_dot_a * ALPHA  # first order filter

    cur_x_motor_angle_b: float = devices.get_X_motor_angle_B()
    cur_x_motor_angle_dot_a: float = (cur_x_motor_angle_b - robot_status.leg_b_joint_space.X_motor_angle) / (
            0.001 * TIME_STEP)
    robot_status.leg_b_joint_space.X_motor_angle = cur_x_motor_angle_b
    robot_status.leg_b_joint_space_dot.X_motor_angle = robot_status.leg_b_joint_space_dot.X_motor_angle * (
            1.0 - ALPHA) + cur_x_motor_angle_dot_a * ALPHA  # first order filter

    cur_z_motor_angle_b = devices.get_Z_motor_angle_B()
    cur_z_motor_angle_dot_b = (cur_z_motor_angle_b - robot_status.leg_b_joint_space.Z_motor_angle) / (
            0.001 * TIME_STEP)
    robot_status.leg_b_joint_space.Z_motor_angle = cur_z_motor_angle_b
    robot_status.leg_b_joint_space_dot.Z_motor_angle = robot_status.leg_b_joint_space_dot.Z_motor_angle * (
            1.0 - ALPHA) + cur_z_motor_angle_dot_b * ALPHA  # first order filter

    # 机器人在世界坐标系下水平速度估计更新
    update_xz_dot()
    # 上次支撑相时间Ts更新
    update_last_Ts()
    # 更新状态机
    updateRobotStateMachine()
    pass


def forwardKinematics(joint_point: JointSpaceRXZL) -> np.ndarray:
    #  转换成弧度制
    rads_x = joint_point.X_motor_angle * math.pi / 180.0
    rads_z = joint_point.Z_motor_angle * math.pi / 180.0
    r = joint_point.r
    x = r * math.sin(rads_z)
    y = -r * math.cos(rads_z) * math.cos(rads_x)
    z = -r * math.cos(rads_z) * math.sin(rads_x)
    return np.array([x, y, z])


def update_xz_dot():
    global robot_status
    robot_status.Point_hat_B_LegA = forwardKinematics(robot_status.leg_a_joint_space)
    robot_status.Point_hat_B_LegB = forwardKinematics(robot_status.leg_b_joint_space)
    #  转换到{H}坐标系下
    pre_x_a = robot_status.Point_hat_H_LegA[0]
    pre_z_a = robot_status.Point_hat_H_LegA[2]
    robot_status.Point_hat_H_LegA = copy.deepcopy(
        robot_status.rotation_matrix_B_under_W @ robot_status.Point_hat_B_LegA)
    now_x_a = robot_status.Point_hat_H_LegA[0]
    now_z_a = robot_status.Point_hat_H_LegA[2]
    #  求导
    now_x_dot_a = -(now_x_a - pre_x_a) / (0.001 * TIME_STEP)
    now_z_dot_a = -(now_z_a - pre_z_a) / (0.001 * TIME_STEP)
    #  滤波
    now_x_dot_a = robot_status.pre_x_dot_LegA * (1.0 - ALPHA) + now_x_dot_a * ALPHA
    now_z_dot_a = robot_status.pre_z_dot_LegA * (1.0 - ALPHA) + now_z_dot_a * ALPHA
    robot_status.pre_x_dot_LegA, robot_status.pre_z_dot_LegA = now_x_dot_a, now_z_dot_a

    pre_x_b = robot_status.Point_hat_H_LegB[0]
    pre_z_b = robot_status.Point_hat_H_LegB[2]
    robot_status.Point_hat_H_LegB = copy.deepcopy(
        robot_status.rotation_matrix_B_under_W @ robot_status.Point_hat_B_LegB)
    now_x_b = robot_status.Point_hat_H_LegB[0]
    now_z_b = robot_status.Point_hat_H_LegB[2]
    #  求导
    now_x_dot_b = -(now_x_b - pre_x_b) / (0.001 * TIME_STEP)
    now_z_dot_b = -(now_z_b - pre_z_b) / (0.001 * TIME_STEP)
    #  滤波
    now_x_dot_b = robot_status.pre_x_dot_LegB * (1.0 - ALPHA) + now_x_dot_b * ALPHA
    now_z_dot_b = robot_status.pre_z_dot_LegB * (1.0 - ALPHA) + now_z_dot_b * ALPHA
    robot_status.pre_x_dot_LegB, robot_status.pre_z_dot_LegB = now_x_dot_b, now_z_dot_b

    if (robot_status.phase_state == COMPRESSION_A) or (robot_status.phase_state == THRUST_A):
        robot_status.body_spd_x = now_x_dot_a
        robot_status.body_spd_z = now_z_dot_a
    if (robot_status.phase_state == COMPRESSION_B) or (robot_status.phase_state == THRUST_B):
        robot_status.body_spd_x = now_x_dot_b
        robot_status.body_spd_z = now_z_dot_b


def update_last_Ts():
    global robot_status
    if not robot_status.pre_is_foot_touching_flg_A and robot_status.is_foot_touching_flg_A:
        robot_status.stance_start_ms = robot_status.system_ms
    if robot_status.pre_is_foot_touching_flg_A and not robot_status.is_foot_touching_flg_A:
        stance_end_ms = robot_status.system_ms
        # Ts should be divided by 2 since we got 2 legs biped, and the Ts here is calculated by Leg-A only,
        robot_status.Ts = 0.001 * float(stance_end_ms - robot_status.stance_start_ms) / 2
    robot_status.pre_is_foot_touching_flg_A = robot_status.is_foot_touching_flg_A


# this is the state machine referenced to <Legged robots that balance page 90>
def updateRobotStateMachine():
    global robot_status
    if robot_status.phase_state == LOADING_A:
        if robot_status.leg_a_joint_space.spring_len < robot_status.spring_normal_length * robot_status.r_threshold:
            robot_status.phase_state = COMPRESSION_A
        return
    elif robot_status.phase_state == COMPRESSION_A:
        if robot_status.leg_a_joint_space_dot.spring_len > 0.0:
            robot_status.phase_state = THRUST_A
            return
    elif robot_status.phase_state == THRUST_A:
        if robot_status.leg_a_joint_space.spring_len > robot_status.spring_normal_length * robot_status.r_threshold:
            robot_status.phase_state = UNLOADING_A
        return
    elif robot_status.phase_state == UNLOADING_A:
        if not robot_status.is_foot_touching_flg_A:
            robot_status.phase_state = FLIGHT_A
        return
    elif robot_status.phase_state == FLIGHT_A:
        if robot_status.is_foot_touching_flg_B:
            robot_status.phase_state = LOADING_B
        return
    # B
    elif robot_status.phase_state == LOADING_B:
        if robot_status.leg_b_joint_space.spring_len < robot_status.spring_normal_length * robot_status.r_threshold:
            robot_status.phase_state = COMPRESSION_B
        return
    elif robot_status.phase_state == COMPRESSION_B:
        if robot_status.leg_b_joint_space_dot.spring_len > 0.0:
            robot_status.phase_state = THRUST_B
            return
    elif robot_status.phase_state == THRUST_B:
        if robot_status.leg_b_joint_space.spring_len > robot_status.spring_normal_length * robot_status.r_threshold:
            robot_status.phase_state = UNLOADING_B
        return
    elif robot_status.phase_state == UNLOADING_B:
        if not robot_status.is_foot_touching_flg_B:
            robot_status.phase_state = FLIGHT_B
        return
    elif robot_status.phase_state == FLIGHT_B:
        if robot_status.is_foot_touching_flg_A:
            robot_status.phase_state = LOADING_A
        return
    else:
        return


def position_A_4_landing():
    r_a = robot_status.leg_a_joint_space.r
    x_f = robot_status.body_spd_x * robot_status.Ts / 2.0 + robot_status.k_xz_dot * (
            robot_status.body_spd_x - robot_status.target_body_spd_x)
    z_f = robot_status.body_spd_z * robot_status.Ts / 2.0 + robot_status.k_xz_dot * (
            robot_status.body_spd_z - robot_status.target_body_spd_z)
    y_f = -math.sqrt(r_a * r_a - x_f * x_f - z_f * z_f)

    robot_status.Point_hat_H_LegA_desire = np.array([x_f, y_f, z_f])

    # 转到{B}坐标系下 P_B = R^B_H * P_H
    robot_status.Point_hat_B_LegA_desire = copy.deepcopy(
        robot_status.rotation_matrix_W_under_B @ robot_status.Point_hat_H_LegA_desire)

    #  计算期望关节角
    [x_f_a, y_f_a, z_f_a] = robot_status.Point_hat_H_LegA_desire
    x_angle_desire = math.atan(z_f_a / y_f_a) * 180.0 / math.pi
    z_angle_desire = math.asin(x_f_a / r_a) * 180.0 / math.pi

    #  控制关节角
    x_angle = robot_status.leg_a_joint_space.X_motor_angle
    z_angle = robot_status.leg_a_joint_space.Z_motor_angle
    x_angle_dot = robot_status.leg_a_joint_space_dot.X_motor_angle
    z_angle_dot = robot_status.leg_a_joint_space_dot.Z_motor_angle
    target_torque_x = -robot_status.leg_kp * (x_angle - x_angle_desire) - robot_status.leg_kd * x_angle_dot
    target_torque_z = -robot_status.leg_kp * (z_angle - z_angle_desire) - robot_status.leg_kd * z_angle_dot
    devices.set_X_torque_A(target_torque_x)
    devices.set_Z_torque_A(target_torque_z)


def position_B_4_landing():
    r_b = robot_status.leg_b_joint_space.r
    x_f = robot_status.body_spd_x * robot_status.Ts / 2.0 + robot_status.k_xz_dot * (
            robot_status.body_spd_x - robot_status.target_body_spd_x)
    z_f = robot_status.body_spd_z * robot_status.Ts / 2.0 + robot_status.k_xz_dot * (
            robot_status.body_spd_z - robot_status.target_body_spd_z)
    y_f = -math.sqrt(r_b * r_b - x_f * x_f - z_f * z_f)

    robot_status.Point_hat_H_LegB_desire = np.array([x_f, y_f, z_f])

    # 转到{B}坐标系下 P_B = R^B_H * P_H
    robot_status.Point_hat_B_LegB_desire = copy.deepcopy(
        robot_status.rotation_matrix_W_under_B @ robot_status.Point_hat_H_LegB_desire)

    #  计算期望关节角
    [x_f_b, y_f_b, z_f_b] = robot_status.Point_hat_H_LegB_desire
    x_angle_desire = math.atan(z_f_b / y_f_b) * 180.0 / math.pi
    z_angle_desire = math.asin(x_f_b / r_b) * 180.0 / math.pi

    #  控制关节角
    x_angle = robot_status.leg_b_joint_space.X_motor_angle
    z_angle = robot_status.leg_b_joint_space.Z_motor_angle
    x_angle_dot = robot_status.leg_b_joint_space_dot.X_motor_angle
    z_angle_dot = robot_status.leg_b_joint_space_dot.Z_motor_angle
    target_torque_x = -robot_status.leg_kp * (x_angle - x_angle_desire) - robot_status.leg_kd * x_angle_dot
    target_torque_z = -robot_status.leg_kp * (z_angle - z_angle_desire) - robot_status.leg_kd * z_angle_dot
    devices.set_X_torque_B(target_torque_x)
    devices.set_Z_torque_B(target_torque_z)


def robot_control():
    # common控制部分
    # 2. 控制弹簧
    dx_a = robot_status.spring_normal_length - robot_status.leg_a_joint_space.spring_len  # 求压缩量
    dx_b = robot_status.spring_normal_length - robot_status.leg_b_joint_space.spring_len  # 求压缩量
    f_spring_a = dx_a * robot_status.k_spring
    f_spring_b = dx_b * robot_status.k_spring

    if robot_status.phase_state == THRUST_A:
        f_spring_a += robot_status.F_thrust
    devices.set_spring_force(devices.spring_motor_A, f_spring_a)

    if robot_status.phase_state == THRUST_B:
        f_spring_b += robot_status.F_thrust
    devices.set_spring_force(devices.spring_motor_B, f_spring_b)

    if robot_status.phase_state == LOADING_A:
        devices.set_X_torque_A(0.0)
        devices.set_Z_torque_A(0.0)
        # shorten B
        robot_status.offset_B = OFFSET_LEN
        robot_status.offset_A = 0.0
        # dont move hip B
        devices.fix_motor_angle_B()

    if robot_status.phase_state == COMPRESSION_A:
        # erect body
        Tx = -(
                -robot_status.pose_kp * robot_status.euler_angles.roll - robot_status.pose_kd * robot_status.euler_angles_dot.roll)
        Tz = -(
                -robot_status.pose_kp * robot_status.euler_angles.pitch - robot_status.pose_kd * robot_status.euler_angles_dot.pitch)
        devices.set_X_torque_A(Tx)
        devices.set_Z_torque_A(Tz)
        # keep b short
        robot_status.offset_B = OFFSET_LEN
        robot_status.offset_A = 0.0
        # position B for landing
        position_B_4_landing()

    if robot_status.phase_state == THRUST_A:
        # erect body
        Tx = -(
                -robot_status.pose_kp * robot_status.euler_angles.roll - robot_status.pose_kd * robot_status.euler_angles_dot.roll)
        Tz = -(
                -robot_status.pose_kp * robot_status.euler_angles.pitch - robot_status.pose_kd * robot_status.euler_angles_dot.pitch)
        devices.set_X_torque_A(Tx)
        devices.set_Z_torque_A(Tz)
        # keep b short
        robot_status.offset_B = OFFSET_LEN
        robot_status.offset_A = 0.0
        # position B for landing
        position_B_4_landing()

    if robot_status.phase_state == UNLOADING_A:
        # shorten A
        robot_status.offset_B = OFFSET_LEN
        robot_status.offset_A = OFFSET_LEN
        # zero hip torque
        devices.set_X_torque_A(0.0)
        devices.set_Z_torque_A(0.0)
        # position B for landing
        position_B_4_landing()

    if robot_status.phase_state == FLIGHT_A:
        # shorten A
        robot_status.offset_A = OFFSET_LEN
        # length B
        robot_status.offset_B = 0.0
        # don't move hip A
        devices.fix_motor_angle_A()
        # position B for landing
        position_B_4_landing()

    if robot_status.phase_state == LOADING_B:
        devices.set_X_torque_B(0.0)
        devices.set_Z_torque_B(0.0)
        # shorten A
        robot_status.offset_A = OFFSET_LEN
        robot_status.offset_B = 0.0
        # don't move hip A
        devices.fix_motor_angle_A()

    if robot_status.phase_state == COMPRESSION_B:
        # erect body
        Tx = -(
                -robot_status.pose_kp * robot_status.euler_angles.roll - robot_status.pose_kd * robot_status.euler_angles_dot.roll)
        Tz = -(
                -robot_status.pose_kp * robot_status.euler_angles.pitch - robot_status.pose_kd * robot_status.euler_angles_dot.pitch)
        devices.set_X_torque_B(Tx)
        devices.set_Z_torque_B(Tz)
        # keep b short
        robot_status.offset_A = OFFSET_LEN
        robot_status.offset_B = 0.0
        # position B for landing
        position_A_4_landing()

    if robot_status.phase_state == THRUST_B:
        # erect body
        Tx = -(
                -robot_status.pose_kp * robot_status.euler_angles.roll - robot_status.pose_kd * robot_status.euler_angles_dot.roll)
        Tz = -(
                -robot_status.pose_kp * robot_status.euler_angles.pitch - robot_status.pose_kd * robot_status.euler_angles_dot.pitch)
        devices.set_X_torque_B(Tx)
        devices.set_Z_torque_B(Tz)
        # keep b short
        robot_status.offset_A = OFFSET_LEN
        robot_status.offset_B = 0.0
        # position B for landing
        position_A_4_landing()

    if robot_status.phase_state == UNLOADING_B:
        # shorten b AND keep A short
        robot_status.offset_B = OFFSET_LEN
        robot_status.offset_A = OFFSET_LEN
        # zero hip torque
        devices.set_X_torque_B(0.0)
        devices.set_Z_torque_B(0.0)
        # position B for landing
        position_A_4_landing()

    if robot_status.phase_state == FLIGHT_B:
        # shorten B
        robot_status.offset_B = OFFSET_LEN
        # length A
        robot_status.offset_A = 0.0
        # dont move hip B
        devices.fix_motor_angle_B()
        # position A for landing
        position_A_4_landing()

    # common控制部分
    # 1. 缩短悬空腿
    devices.set_shorten_length_A(robot_status.offset_A)
    devices.set_shorten_length_B(robot_status.offset_B)

    pass


keyboard = Keyboard()
keyboard.enable(TIME_STEP)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(TIME_STEP) != -1:
    new_key = keyboard.getKey()
    if new_key == Keyboard.UP:
        robot_status.target_body_spd_x += 0.001
    if new_key == Keyboard.DOWN:
        robot_status.target_body_spd_x -= 0.001
    new_key = None
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    # devices.shorten_motor_B.setPosition(robot_status.offset_B)
    state_names = [
        "LOADING_A",
        "COMPRESSION_A",
        "THRUST_A",
        "UNLOADING_A",
        "FLIGHT_A",
        "LOADING_B",
        "COMPRESSION_B",
        "THRUST_B",
        "UNLOADING_B",
        "FLIGHT_B"
    ]
    updateRobotState()
    robot_control()
    # print(
    #     "robot state: ", state_names[robot_status.phase_state],
    #     'A offset: ', robot_status.offset_A,
    #     'B offset: ', robot_status.offset_B,
    #     'Angle of B: ', robot_status.leg_b_joint_space.X_motor_angle,
    #     robot_status.leg_b_joint_space.Z_motor_angle,
    #
    # )
    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
