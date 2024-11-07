"""my_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from interfaces import *
from scipy.spatial.transform import Rotation as R
from controller import Robot, Motor, TouchSensor, PositionSensor, InertialUnit
import copy

# create the Robot instance.
robot = Robot()
robot_status = RobotStatus()
devices = Devices(robot)

# get the time step of the current world.
TIME_STEP = int(robot.getBasicTimeStep())
# 滤波系数
ALPHA = 1.0  # new value coeff  1 - ALPHA : old value coeff


def updateRobotState():
    global robot_status
    # 时钟更新
    robot_status.system_ms += TIME_STEP

    #  足底传感器更新
    robot_status.is_foot_touching_flg = devices.is_foot_touching()

    # IMU，IMU导数，以及旋转矩阵更新
    now_IMU: EulerAnglesRPY = devices.get_IMU_Angle()
    now_IMU_dot: EulerAnglesRPY = EulerAnglesRPY()

    now_IMU_dot.roll = (now_IMU.roll - robot_status.euler_angles.roll) / (0.001 * TIME_STEP)
    now_IMU_dot.pitch = (now_IMU.pitch - robot_status.euler_angles.pitch) / (0.001 * TIME_STEP)
    now_IMU_dot.yaw = (now_IMU.yaw - robot_status.euler_angles.yaw) / (0.001 * TIME_STEP)

    robot_status.euler_angles = now_IMU
    robot_status.euler_angles_dot.roll = \
        robot_status.euler_angles_dot.roll * (1.0 - ALPHA) + now_IMU_dot.roll * ALPHA
    robot_status.euler_angles_dot.pitch = \
        robot_status.euler_angles_dot.pitch * (1.0 - ALPHA) + now_IMU_dot.pitch * ALPHA
    robot_status.euler_angles_dot.yaw = \
        robot_status.euler_angles_dot.yaw * (1.0 - ALPHA) + now_IMU_dot.yaw * ALPHA

    # 更新旋转矩阵
    r = R.from_euler('xzy', [now_IMU.roll, now_IMU.pitch, now_IMU.yaw], degrees=True)
    robot_status.rotation_matrix_B_under_H = np.array(copy.deepcopy(r.as_matrix()))
    robot_status.rotation_matrix_H_under_B = np.transpose(copy.deepcopy(robot_status.rotation_matrix_B_under_H))

    # 弹簧长度及其导数更新
    now_r: float = devices.get_spring_length()
    now_r_dot = (now_r - robot_status.joint_space_lxz.r) / (0.001 * TIME_STEP)
    robot_status.joint_space_lxz.r = now_r
    robot_status.joint_space_lxz_dot.r = robot_status.joint_space_lxz_dot.r * (
            1.0 - ALPHA) + now_r_dot * ALPHA  # 一阶低通滤波器

    # X、Z关节角度更新*/
    now_X_motor_angle: float = devices.get_X_motor_angle()
    now_X_motor_angle_dot: float = (now_X_motor_angle - robot_status.joint_space_lxz.X_motor_angle) / (
            0.001 * TIME_STEP)
    robot_status.joint_space_lxz.X_motor_angle = now_X_motor_angle
    robot_status.joint_space_lxz_dot.X_motor_angle = robot_status.joint_space_lxz_dot.X_motor_angle * (
            1.0 - ALPHA) + now_X_motor_angle_dot * ALPHA  # 一阶低通滤波器

    now_Z_motor_angle = devices.get_Z_motor_angle()
    now_Z_motor_angle_dot = (now_Z_motor_angle - robot_status.joint_space_lxz.Z_motor_angle) / (0.001 * TIME_STEP)
    robot_status.joint_space_lxz.Z_motor_angle = now_Z_motor_angle
    robot_status.joint_space_lxz_dot.Z_motor_angle = robot_status.joint_space_lxz_dot.Z_motor_angle * (
            1.0 - ALPHA) + now_Z_motor_angle_dot * ALPHA  # 一阶低通滤波器

    # 机器人在世界坐标系下水平速度估计更新
    update_xz_dot()
    # 上次支撑相时间Ts更新
    update_last_Ts()
    # 更新状态机
    updateRobotStateMachine()
    pass


def forwardKinematics(jointPoint: JointSpaceLXZ) -> np.ndarray:
    #  转换成弧度制
    Tx = jointPoint.X_motor_angle * math.pi / 180.0
    Tz = jointPoint.Z_motor_angle * math.pi / 180.0
    r = jointPoint.r
    x = r * math.sin(Tz)
    y = -r * math.cos(Tz) * math.cos(Tx)
    z = -r * math.cos(Tz) * math.sin(Tx)
    return np.array([x, y, z])


def update_xz_dot():
    global robot_status
    robot_status.Point_hat_B = forwardKinematics(robot_status.joint_space_lxz)
    #  转换到{H}坐标系下
    pre_x = robot_status.Point_hat_H[0]
    pre_z = robot_status.Point_hat_H[2]
    robot_status.Point_hat_H = copy.deepcopy(robot_status.rotation_matrix_B_under_H @ robot_status.Point_hat_B)
    now_x = robot_status.Point_hat_H[0]
    now_z = robot_status.Point_hat_H[2]
    #  求导
    now_x_dot = -(now_x - pre_x) / (0.001 * TIME_STEP)
    now_z_dot = -(now_z - pre_z) / (0.001 * TIME_STEP)
    #  滤波
    now_x_dot = robot_status.pre_x_dot * (1.0 - ALPHA) + now_x_dot * ALPHA
    now_z_dot = robot_status.pre_z_dot * (1.0 - ALPHA) + now_z_dot * ALPHA
    robot_status.pre_x_dot, robot_status.pre_z_dot = now_x_dot, now_z_dot
    if (robot_status.robot_state == COMPRESSION) or (robot_status.robot_state == THRUST):
        robot_status.x_dot = now_x_dot
        robot_status.z_dot = now_z_dot


def update_last_Ts():
    global robot_status
    if not robot_status.pre_is_foot_touching_flg and robot_status.is_foot_touching_flg:
        robot_status.stance_start_ms = robot_status.system_ms
    if robot_status.pre_is_foot_touching_flg and not robot_status.is_foot_touching_flg:
        stance_end_ms = robot_status.system_ms
        robot_status.Ts = 0.001 * float(stance_end_ms - robot_status.stance_start_ms)
    robot_status.pre_is_foot_touching_flg = robot_status.is_foot_touching_flg


def updateRobotStateMachine():
    global robot_status
    if robot_status.robot_state == LOADING:
        if robot_status.joint_space_lxz.r < robot_status.spring_normal_length * robot_status.r_threshold:
            robot_status.robot_state = COMPRESSION
        return
    elif robot_status.robot_state == COMPRESSION:
        if robot_status.joint_space_lxz_dot.r > 0.0:
            robot_status.robot_state = THRUST
            return
    elif robot_status.robot_state == THRUST:
        if robot_status.joint_space_lxz.r > robot_status.spring_normal_length * robot_status.r_threshold:
            robot_status.robot_state = UNLOADING
        return
    elif robot_status.robot_state == UNLOADING:
        if not robot_status.is_foot_touching_flg:
            robot_status.robot_state = FLIGHT
        return
    elif robot_status.robot_state == FLIGHT:
        if robot_status.is_foot_touching_flg:
            robot_status.robot_state = LOADING
        return
    else:
        return


def updateRobotStateMachine2():
    global robot_status
    if robot_status.robot_state_2 == LOADING_A:
        if robot_status.joint_space_lxz.r < robot_status.spring_normal_length * robot_status.r_threshold:
            robot_status.robot_state_2 = COMPRESSION_A
        return
    elif robot_status.robot_state_2 == COMPRESSION_A:
        if robot_status.joint_space_lxz_dot.r > 0.0:
            robot_status.robot_state_2 = THRUST_A
            return
    elif robot_status.robot_state_2 == THRUST_A:
        if robot_status.joint_space_lxz.r > robot_status.spring_normal_length * robot_status.r_threshold:
            robot_status.robot_state_2 = UNLOADING_A
        return
    elif robot_status.robot_state_2 == UNLOADING_A:
        if not robot_status.is_foot_touching_flg:
            robot_status.robot_state_2 = FLIGHT_A
        return
    elif robot_status.robot_state_2 == FLIGHT_A:
        if robot_status.is_foot_touching_flg_2:
            robot_status.robot_state_2 = LOADING_B
        return
    # B
    elif robot_status.robot_state_2 == LOADING_B:
        if robot_status.joint_space_lxz_2.r < robot_status.spring_normal_length * robot_status.r_threshold:
            robot_status.robot_state_2 = COMPRESSION_B
        return
    elif robot_status.robot_state_2 == COMPRESSION_B:
        if robot_status.joint_space_lxz_dot_2.r > 0.0:
            robot_status.robot_state_2 = THRUST_B
            return
    elif robot_status.robot_state_2 == THRUST_B:
        if robot_status.joint_space_lxz_2.r > robot_status.spring_normal_length * robot_status.r_threshold:
            robot_status.robot_state_2 = UNLOADING_B
        return
    elif robot_status.robot_state_2 == UNLOADING_B:
        if not robot_status.is_foot_touching_flg_2:
            robot_status.robot_state_2 = FLIGHT_B
        return
    else:
        return


def robot_control():
    devices.linear_motor_a.setPosition(-0.1)
    dx = robot_status.spring_normal_length - robot_status.joint_space_lxz.r  # 求压缩量
    f_spring = dx * robot_status.k_spring
    if robot_status.robot_state == THRUST:
        f_spring += robot_status.F_thrust
    devices.set_spring_force(devices.spring_motor, f_spring)

    # 控制臀部扭矩力 LOADING和UNLOADING时候，扭矩为0
    if (robot_status.robot_state == LOADING) or (robot_status.robot_state == UNLOADING):
        devices.set_X_torque(0.0)
        devices.set_Z_torque(0.0)

    # COMPRESSION和THRUST时候，臀部电机控制身体姿态
    if (robot_status.robot_state == COMPRESSION) or (robot_status.robot_state == THRUST):
        Tx = -(
                -robot_status.k_pose_p * robot_status.euler_angles.roll - robot_status.k_pose_v * robot_status.euler_angles_dot.roll)
        Tz = -(
                -robot_status.k_pose_p * robot_status.euler_angles.pitch - robot_status.k_pose_v * robot_status.euler_angles_dot.pitch)
        devices.set_X_torque(Tx)
        devices.set_Z_torque(Tz)

    #  FLIGHT的时候，控制足底移动到落足点
    if robot_status.robot_state == FLIGHT:
        r = robot_status.joint_space_lxz.r
        x_f = robot_status.x_dot * robot_status.Ts / 2.0 + robot_status.k_xz_dot * (
                robot_status.x_dot - robot_status.x_dot_desire)
        z_f = robot_status.z_dot * robot_status.Ts / 2.0 + robot_status.k_xz_dot * (
                robot_status.z_dot - robot_status.z_dot_desire)
        y_f = -math.sqrt(r * r - x_f * x_f - z_f * z_f)

        robot_status.Point_hat_H_desire = np.array([x_f, y_f, z_f])
        # 转到{B}坐标系下 P_B = R^B_H * P_H
        robot_status.Point_hat_B_desire = copy.deepcopy(
            robot_status.rotation_matrix_H_under_B @ robot_status.Point_hat_H_desire)

        #  计算期望关节角
        [x_f_B, y_f_B, z_f_B] = robot_status.Point_hat_H_desire
        x_angle_desire = math.atan(z_f_B / y_f_B) * 180.0 / math.pi
        z_angle_desire = math.asin(x_f_B / r) * 180.0 / math.pi

        #  控制关节角
        x_angle = robot_status.joint_space_lxz.X_motor_angle
        z_angle = robot_status.joint_space_lxz.Z_motor_angle
        x_angle_dot = robot_status.joint_space_lxz_dot.X_motor_angle
        z_angle_dot = robot_status.joint_space_lxz_dot.Z_motor_angle
        Tx = -robot_status.k_leg_p * (x_angle - x_angle_desire) - robot_status.k_leg_v * x_angle_dot
        Tz = -robot_status.k_leg_p * (z_angle - z_angle_desire) - robot_status.k_leg_v * z_angle_dot
        devices.set_X_torque(Tx)
        devices.set_Z_torque(Tz)

def robot_control_2():
    dx = robot_status.spring_normal_length - robot_status.joint_space_lxz.r  # 求压缩量
    dx2 = robot_status.spring_normal_length - robot_status.joint_space_lxz_2.r  # 求压缩量
    f_spring = dx * robot_status.k_spring
    f_spring_2 = dx2 * robot_status.k_spring
    if robot_status.robot_state_2 == THRUST_A:
        f_spring += robot_status.F_thrust
    devices.set_spring_force(devices.spring_motor, f_spring)

    if robot_status.robot_state_2 == FLIGHT_B:
        f_spring_2 += robot_status.F_thrust
    devices.set_spring_force(devices.spring_motor_2, f_spring_2)


    # 控制臀部扭矩力 LOADING和UNLOADING时候，扭矩为0
    if (robot_status.robot_state == LOADING) or (robot_status.robot_state == UNLOADING):
        devices.set_X_torque(0.0)
        devices.set_Z_torque(0.0)

    if robot_status.robot_state_2 == LOADING_A:
        # zero hip torque A
        devices.set_X_torque(0.0)
        devices.set_Z_torque(0.0)
        #shorted B


    # COMPRESSION和THRUST时候，臀部电机控制身体姿态
    if (robot_status.robot_state == COMPRESSION) or (robot_status.robot_state == THRUST):
        Tx = -(
                -robot_status.k_pose_p * robot_status.euler_angles.roll - robot_status.k_pose_v * robot_status.euler_angles_dot.roll)
        Tz = -(
                -robot_status.k_pose_p * robot_status.euler_angles.pitch - robot_status.k_pose_v * robot_status.euler_angles_dot.pitch)
        devices.set_X_torque(Tx)
        devices.set_Z_torque(Tz)

    #  FLIGHT的时候，控制足底移动到落足点
    if robot_status.robot_state == FLIGHT:
        r = robot_status.joint_space_lxz.r
        x_f = robot_status.x_dot * robot_status.Ts / 2.0 + robot_status.k_xz_dot * (
                robot_status.x_dot - robot_status.x_dot_desire)
        z_f = robot_status.z_dot * robot_status.Ts / 2.0 + robot_status.k_xz_dot * (
                robot_status.z_dot - robot_status.z_dot_desire)
        y_f = -math.sqrt(r * r - x_f * x_f - z_f * z_f)

        robot_status.Point_hat_H_desire = np.array([x_f, y_f, z_f])
        # 转到{B}坐标系下 P_B = R^B_H * P_H
        robot_status.Point_hat_B_desire = copy.deepcopy(
            robot_status.rotation_matrix_H_under_B @ robot_status.Point_hat_H_desire)

        #  计算期望关节角
        [x_f_B, y_f_B, z_f_B] = robot_status.Point_hat_H_desire
        x_angle_desire = math.atan(z_f_B / y_f_B) * 180.0 / math.pi
        z_angle_desire = math.asin(x_f_B / r) * 180.0 / math.pi

        #  控制关节角
        x_angle = robot_status.joint_space_lxz.X_motor_angle
        z_angle = robot_status.joint_space_lxz.Z_motor_angle
        x_angle_dot = robot_status.joint_space_lxz_dot.X_motor_angle
        z_angle_dot = robot_status.joint_space_lxz_dot.Z_motor_angle
        Tx = -robot_status.k_leg_p * (x_angle - x_angle_desire) - robot_status.k_leg_v * x_angle_dot
        Tz = -robot_status.k_leg_p * (z_angle - z_angle_desire) - robot_status.k_leg_v * z_angle_dot
        devices.set_X_torque(Tx)
        devices.set_Z_torque(Tz)
    pass

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(TIME_STEP) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    updateRobotState()
    robot_control()
    # print(robot_status.Point_hat_B_desire_a)
    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
