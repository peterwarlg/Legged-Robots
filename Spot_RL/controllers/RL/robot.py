from controller import Robot, Motor, GPS, Supervisor

TIME_STEP = 32


class robot_spot:
    def __init__(self):
        self.supervisor = Supervisor()
        self.gps = self.supervisor.getDevice('gps')
        self.gps.enable(TIME_STEP)
        self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

        self.motor_names = [
            "front left shoulder rotation motor", "front left elbow motor",
            "front right shoulder rotation motor", "front right elbow motor",
            "rear left shoulder rotation motor", "rear left elbow motor",
            "rear right shoulder rotation motor", "rear right elbow motor"
        ]

        self.motors = [self.supervisor.getDevice(name) for name in self.motor_names]
        for motor in self.motors:
            motor.setPosition(0.0)  # Initialize to zero position
            motor.setVelocity(0.0)  # Set velocity
            motor.getPositionSensor().enable(TIME_STEP)

        self.state_dim = 11
        self.action_dim = len(self.motor_names)


    def step(self):
        pass

    def reset(self):
        pass

    def