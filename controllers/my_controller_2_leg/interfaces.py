import math
import numpy as np
from controller import Robot, Motor, TouchSensor, PositionSensor, InertialUnit

LOADING = 0x00  # 落地
COMPRESSION = 0x01  # 压缩腿
THRUST = 0x02  # 伸长腿
UNLOADING = 0x03  # 离地
FLIGHT = 0x04  # 飞行

LOADING_A = 0x00
COMPRESSION_A = 0x01
THRUST_A = 0x02
UNLOADING_A = 0x3
FLIGHT_A = 0x04
LOADING_B = 0x05
COMPRESSION_B = 0x06
THRUST_B = 0x07
UNLOADING_B = 0x08
FLIGHT_B = 0x09


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


class RobotStatus:
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

        # 正数向上
        self.leg_a_offset = 0.0
        self.leg_b_offset = 0.0

        # 机器人状态
        # euler angle
        self.euler_angles = EulerAnglesRPY()
        self.euler_angles_dot = EulerAnglesRPY()

        # A腿计算矩阵
        self.rotation_matrix_B_under_H: np.ndarray = np.eye(3, dtype=float)  # body->world
        self.rotation_matrix_H_under_B: np.ndarray = np.eye(3, dtype=float)  # world->body
        # B腿计算矩阵
        self.rotation_matrix_B_under_H_2: np.ndarray = np.eye(3, dtype=float)  # body->world
        self.rotation_matrix_H_under_B_2: np.ndarray = np.eye(3, dtype=float)  # world->body

        # r x z
        self.joint_space_lxz = JointSpaceLXZ(0.8, 0.0, 0.0)
        self.joint_space_lxz_dot = JointSpaceLXZ(0.0, 0.0, 0.0)
        self.joint_space_lxz_2 = JointSpaceLXZ(0.8, 0.0, 0.0)
        self.joint_space_lxz_dot_2 = JointSpaceLXZ(0.0, 0.0, 0.0)

        # A: x y z
        self.Point_hat_B: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.Point_hat_H: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.Point_hat_B_desire: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.Point_hat_H_desire: np.ndarray = np.array([0.0, 0.0, 0.0])
        # B: x y z
        self.Point_hat_B_2: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.Point_hat_H_2: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.Point_hat_B_desire_2: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.Point_hat_H_desire_2: np.ndarray = np.array([0.0, 0.0, 0.0])

        self.is_foot_touching_flg: bool = True
        self.is_foot_touching_flg_2: bool = True
        self.Ts: float = 0.0
        self.x_dot: float = 0.0
        self.z_dot: float = 0.0
        self.x_dot_desire: float = 0.0
        self.z_dot_desire: float = 0.0

        self.system_ms: int = 0
        self.robot_state: int = THRUST
        self.robot_state_2: int = THRUST_A

        self.pre_is_foot_touching_flg = False
        self.pre_is_foot_touching_flg_2 = False
        self.stance_start_ms: int = 0
        self.debug = []
        self.pre_x_dot = 0.0
        self.pre_z_dot = 0.0



class Devices:
    def __init__(self, _robot: Robot) -> None:
        self.IMU: InertialUnit = _robot.getInertialUnit("inertial unit")

        self.spring_motor: Motor = _robot.getMotor("Spring linear motor A")
        self.spring_pos_sensor: PositionSensor = _robot.getPositionSensor("Spring position sensor A")
        self.touch_sensor: TouchSensor = _robot.getTouchSensor("touch sensor A")
        self.X_motor: Motor = _robot.getMotor("X rotational motor A")
        self.X_motor_position_sensor: PositionSensor = _robot.getPositionSensor("X position sensor A")
        self.Z_motor: Motor = _robot.getMotor("Z rotational motor A")
        self.Z_motor_position_sensor: PositionSensor = _robot.getPositionSensor("Z position sensor A")

        self.spring_motor_2: Motor = _robot.getMotor("Spring linear motor B")
        self.spring_pos_sensor_2: PositionSensor = _robot.getPositionSensor("Spring position sensor B")
        self.touch_sensor_2: TouchSensor = _robot.getTouchSensor("touch sensor B")
        self.X_motor_2: Motor = _robot.getMotor("X rotational motor B")
        self.X_motor_position_sensor_2: PositionSensor = _robot.getPositionSensor("X position sensor B")
        self.Z_motor_2: Motor = _robot.getMotor("Z rotational motor B")
        self.Z_motor_position_sensor_2: PositionSensor = _robot.getPositionSensor("Z position sensor B")

        # setposition 朝上为+
        self.linear_motor_a: Motor = _robot.getMotor('Shorten linear motor A')

        time_step = int(_robot.getBasicTimeStep())

        self.IMU.enable(time_step)

        self.spring_pos_sensor.enable(time_step)
        self.X_motor_position_sensor.enable(time_step)
        self.Z_motor_position_sensor.enable(time_step)
        self.touch_sensor.enable(time_step)
        self.spring_pos_sensor_2.enable(time_step)
        self.X_motor_position_sensor_2.enable(time_step)
        self.Z_motor_position_sensor_2.enable(time_step)
        self.touch_sensor_2.enable(time_step)

    def set_spring_force(self, spring_motor: Motor, force: float):
        spring_motor.setForce(-force)
        # self.spring_motor.setForce(-force)

    def get_spring_length(self) -> float:
        l: float = self.spring_pos_sensor.getValue()
        return 0.8 - l

    def get_spring_length_2(self) -> float:
        l: float = self.spring_pos_sensor_2.getValue()
        return 0.8 - l

    def set_X_torque(self, torque):
        self.X_motor.setTorque(torque)

    def set_X_torque_2(self, torque):
        self.X_motor_2.setTorque(torque)

    def set_Z_torque(self, torque):
        self.Z_motor.setTorque(torque)

    def set_Z_torque_2(self, torque):
        self.Z_motor_2.setTorque(torque)

    def get_X_motor_angle(self) -> float:
        angle: float = self.X_motor_position_sensor.getValue()
        return angle * 180.0 / math.pi

    def get_X_motor_angle_2(self) -> float:
        angle: float = self.X_motor_position_sensor_2.getValue()
        return angle * 180.0 / math.pi

    def get_Z_motor_angle(self) -> float:
        angle: float = self.Z_motor_position_sensor.getValue()
        return angle * 180.0 / math.pi

    def get_Z_motor_angle_2(self) -> float:
        angle: float = self.Z_motor_position_sensor_2.getValue()
        return angle * 180.0 / math.pi

    def is_foot_touching(self) -> bool:
        return self.touch_sensor.getValue()

    def is_foot_touching_2(self) -> bool:
        return self.touch_sensor_2.getValue()

    def get_IMU_Angle(self) -> EulerAnglesRPY:
        data = self.IMU.getRollPitchYaw()
        euler_angles_rpy = EulerAnglesRPY()
        euler_angles_rpy.roll = data[0] * 180.0 / math.pi
        euler_angles_rpy.pitch = data[1] * 180.0 / math.pi
        euler_angles_rpy.yaw = data[2] * 180.0 / math.pi
        return euler_angles_rpy

    def set_spring_position_A(self, position):
        err = position - self.get_spring_length()
