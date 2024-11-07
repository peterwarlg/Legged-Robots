from controller import Robot
from controller import Robot, Motor, TouchSensor, PositionSensor, InertialUnit
import numpy as np

LOADING = 0x00  # 落地
COMPRESSION = 0x01  # 压缩腿
THRUST = 0x02  # 伸长腿
UNLOADING = 0x03  # 离地
FLIGHT = 0x04  # 飞行


class EulerAnglesRPY:
    def __init__(self, r=0.0, p=0.0, y=0.0):
        self.roll = r
        self.pitch = p
        self.yaw = y


class JointSpaceLXZ:
    def __init__(self, r=0.0, p=0.0, y=0.0):
        self.r = r
        self.X_motor_angle = p
        self.Z_motor_angle = y


class OneLegRobot:
    def __init__(self):
        # 机器人属性
        self.spring_normal_length: float = 1.2  # *弹簧原长
        self.v: float = 0.2  # 机器人水平运动速度
        self.r_threshold: float = 0.95  # 状态机在脱离LOADING和进入UNLOADING状态时，腿长阈值判断
        self.k_spring: float = 2000.0  # 弹簧刚度
        self.F_thrust: float = 100.0  # THRUST推力
        self.k_leg_p: float = 6.0  # 腿部控制时的kp
        self.k_leg_v: float = 0.8  # 腿部控制时的kv
        self.k_xz_dot: float = 0.072  # 净加速度系数
        self.k_pose_p: float = 0.8  # 姿态控制时的kp
        self.k_pose_v: float = 0.025  # 姿态控制时的kv

        # 机器人状态
        # euler angle
        self.euler_angles = EulerAnglesRPY()
        self.euler_angles_dot = EulerAnglesRPY()

        self.rotation_matrix_B_under_H: np.ndarray = np.eye(3, dtype=float)  # body->world
        self.rotation_matrix_H_under_B: np.ndarray = np.eye(3, dtype=float)  # world->body

        # r x z
        self.joint_space_lxz = JointSpaceLXZ(0.8, 0.0, 0.0)
        self.joint_space_lxz_dot = JointSpaceLXZ(0.0, 0.0, 0.0)

        # x y z
        self.Point_hat_B: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.Point_hat_H: np.ndarray = np.array([0.0, 0.0, 0.0])

        self.Point_hat_B_desire: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.Point_hat_H_desire: np.ndarray = np.array([0.0, 0.0, 0.0])

        self.is_foot_touching_flg: bool = True
        self.Ts: float = 0.0
        self.x_dot: float = 0.0
        self.z_dot: float = 0.0
        self.x_dot_desire: float = 0.0
        self.z_dot_desire: float = 0.0

        self.system_ms: int = 0
        self.robot_state: int = THRUST

        self.pre_is_foot_touching_flg = False
        self.stance_start_ms: int = 0
        self.debug = []
        self.pre_x_dot = 0.0
        self.pre_z_dot = 0.0
