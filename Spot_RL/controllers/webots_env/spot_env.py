from controller import Robot, Motor, GPS, Supervisor
import numpy as np


class RobotConfig:
    def __init__(self):
        self.robot_name: str = 'spot'
        self.control_mode: str = 'torque'
        self.active_motors: list = [
            'front left shoulder rotation motor', 'front left elbow motor',
            'front right shoulder rotation motor', 'front right elbow motor',
            'rear left shoulder rotation motor', 'rear left elbow motor',
            'rear right shoulder rotation motor', 'rear right elbow motor'
        ]
        self.active_motors_max_torque: float = float('inf')
        self.active_motors_max_velocity: float = float('inf')
        self.active_motors_max_position: float = float('inf')
        self.max_step_one_episode: int = 50
        self.sim_time_step: int = 32


class SpotEnv:
    def __init__(self) -> None:
        spot_cfg = RobotConfig()
        self.max_step_one_episode: int = spot_cfg.max_step_one_episode
        self.sim_time_step: int = spot_cfg.sim_time_step

        self.counter_step: int = 0

        self.supervisor = Supervisor()

        self.gps = self.supervisor.getDevice('gps')
        self.imu = self.supervisor.getDevice('inertial unit')
        self.gyro = self.supervisor.getDevice('gyro')
        self.accelerometer = self.supervisor.getDevice('accelerometer')

        self.gps.enable(self.sim_time_step)
        self.imu.enable(self.sim_time_step)
        self.gyro.enable(self.sim_time_step)
        self.accelerometer.enable(self.sim_time_step)

        print(f'Sensors Init ...')
        print(f"gps info: {self.gps.getValues()}")
        print(f"imu info: {self.imu.getRollPitchYaw()}")
        print(f"gyro info: {self.gyro.getValues()}")
        print(f"accelerometer info: {self.accelerometer.getValues()}")

        num_devices = self.supervisor.getNumberOfDevices()
        for i in range(num_devices):
            device = self.supervisor.getDeviceByIndex(i)
            print(device.getName())

        self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_REAL_TIME)

        self.motor_names = spot_cfg.active_motors
        self.motors = [self.supervisor.getDevice(name) for name in self.motor_names]

        self.state_dim: int = 11
        self.action_dim: int = len(self.motors)

        for motor in self.motors:
            motor.setPosition(0.0)  # Initialize to zero position
            motor.setVelocity(0.0)  # Set velocity
            motor.getPositionSensor().enable(self.sim_time_step)


    def compute_reward(self) -> float:
        reward = 0
        return reward

    def print_sensor_info(self):
        print(f"gps info: {self.gps.getValues()}")
        print(f"imu info: {self.imu.getRollPitchYaw()}")
        print(f"gyro info: {self.gyro.getValues()}")
        print(f"accelerometer info: {self.accelerometer.getValues()}")

    def get_observations(self) -> np.ndarray:
        # gps values
        gps_values = self.gps.getValues()
        imu_values = self.imu.getRollPitchYaw()
        gyro_values = self.gyro.getValues()
        accelerometer_values = self.accelerometer.getValues()

        # motor position
        motor_positions = [motor.getPositionSensor().getValue() for motor in self.motors]
        # motor vel
        # motor torque
        # foot touched
        # body posture
        observations = np.concatenate([gps_values, motor_positions])
        # return torch.tensor(observations, dtype=torch.float32).to(device)
        return observations

    def is_done(self) -> bool:
        if self.counter_step >= self.max_step_one_episode:
            return True
        return False

    def step(self, action):
        self.counter_step += 1
        for i, motor in enumerate(self.motors):
            motor.setPosition(action[i])
            # motor.setVelocity(10)
        self.supervisor.step(self.sim_time_step * 10)

        next_state = self.get_observations()
        reward = self.compute_reward()
        done = self.is_done()
        return next_state, reward, done, None

    def reset(self):
        self.counter_step = 0
        print("Reset called")
        for motor in self.motors:
            motor.setPosition(0.0)  # Reset motor positions
            motor.setVelocity(0.0)
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()  # Reset the physics to apply changes
        self.supervisor.step(self.sim_time_step * 1)  # Step to apply the reset
        return self.get_observations()
