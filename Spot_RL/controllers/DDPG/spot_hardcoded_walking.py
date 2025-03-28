from controller import Robot, Motor, GPS, Supervisor
import numpy as np
from collections import deque
import torch
# Constants
TIME_STEP = 32
MAX_SPEED = 6.28
PREVIOUS_DISTANCE = None
# Initial and target positions
INITIAL_POSITION = [-2.87, 0, 0.624]
TARGET_GOAL = [1.2, -0.199, 0.624]
MEMORY = deque(maxlen=1000000)
# Motor names for Spot robot
MOTOR_NAMES = [
     "front left shoulder rotation motor", "front left elbow motor",
     "front right shoulder rotation motor", "front right elbow motor",
     "rear left shoulder rotation motor", "rear left elbow motor",
     "rear right shoulder rotation motor", "rear right elbow motor"
]

# Initialize the robot
supervisor = Supervisor()
robot = supervisor.getFromDef("SPOT")
supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
#robot = Robot()

# Initialize GPS
gps = supervisor.getDevice('gps')
gps.enable(TIME_STEP)


motors = {}
for name in MOTOR_NAMES:
    motor = supervisor.getDevice(name)
    motor.setPosition(float('inf'))  # Set motor to velocity control mode
    motor.setVelocity(0.0)
    motor.getPositionSensor().enable(TIME_STEP)
    motors[name] = motor
 
    
def get_gps_position():
    return gps.getValues()

def compute_distance_to_goal(current_position):
    return np.linalg.norm(np.array(TARGET_GOAL) - np.array(current_position))

motor_names_array = [
             "front left shoulder rotation motor", "front left elbow motor",
             "front right shoulder rotation motor", "front right elbow motor",
             "rear left shoulder rotation motor", "rear left elbow motor",
             "rear right shoulder rotation motor", "rear right elbow motor"
        ]
def get_observations():
        gps_values = gps.getValues()
        motor_positions = [motors[name].getPositionSensor().getValue() for name in motor_names_array]
        observations = np.concatenate([gps_values, motor_positions])
        return torch.tensor(observations, dtype=torch.float32).cuda()

def compute_reward():
        global PREVIOUS_DISTANCE
        gps_values = gps.getValues()
        distance_to_goal = np.linalg.norm(np.array(TARGET_GOAL) - np.array(gps_values))
        reward = -distance_to_goal  # Reward is negative distance to goal
        
        # Stability reward
        # Example condition: if the robot's height is within a certain range, it is considered stable
        stable_height = 0.6  # Adjust based on the robot's normal height
        height_tolerance = 0.1  # Allowable deviation from stable height
    
        if abs(gps_values[2] - stable_height) < height_tolerance:
            reward += 10  # Positive reward for maintaining balance
        
        # Progress reward
        if PREVIOUS_DISTANCE is not None:
            progress = PREVIOUS_DISTANCE - distance_to_goal
            reward += progress * 10  # Adjust the multiplier as needed
    
        PREVIOUS_DISTANCE = distance_to_goal
    
        if distance_to_goal < 0.1:  # If within 10 cm of the target goal
            reward += 1000  # Large reward for reaching the goal
        return reward
        
def reset():
        global PREVIOUS_DISTANCE
        print("Reset called")
        PREVIOUS_DISTANCE = None
        # for motor in motors:
            # motor.setPosition(0.0)  # Reset motor positions
            # motor.setVelocity(0.0)
        supervisor.simulationReset()
        supervisor.simulationResetPhysics()  # Reset the physics to apply changes
        supervisor.step(TIME_STEP*10)        

def move_legs(leg_motors, positions):
    for i, motor in enumerate(leg_motors):
        motors[motor].setPosition(positions[i])
        motors[motor].setVelocity(MAX_SPEED)

def step_forward():
    # Step pattern: First left legs, then right legs
    left_legs = ["front left shoulder rotation motor", "front left elbow motor",
                 "rear left shoulder rotation motor", "rear left elbow motor"]
    right_legs = ["front right shoulder rotation motor", "front right elbow motor",
                  "rear right shoulder rotation motor", "rear right elbow motor"]
    done = False
    
    # Move left legs forward
    state1 = get_observations()
    move_legs(left_legs, [0.3, -0.08, 0.3, -0.08]) 
    supervisor.step(TIME_STEP*10)  # Wait for step to complete
    action1 = torch.tensor([0.3, -0.08, 0.0, 0.0, 0.3, -0.08, 0.0, 0.0])
    reward1 = compute_reward()
    #print("Reward1 : ", reward1)

    # Move left legs back to neutral
    state2 = get_observations()
    move_legs(left_legs, [0.0, 0.0, 0.0, 0.0])
    supervisor.step(TIME_STEP*10)
    action2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    reward2 = compute_reward()
    #print("Reward2 : ", reward2)

    # Move right legs forward
    state3 = get_observations()
    move_legs(right_legs, [0.3, -0.08, 0.3, -0.08])
    supervisor.step(TIME_STEP*10)
    action3 = torch.tensor([0.0, 0.0, 0.3, -0.08, 0.0, 0.0, 0.3, -0.08])
    reward3 = compute_reward()
    #print("Reward3 : ", reward3)

    # Move right legs back to neutral
    state4 = get_observations()
    move_legs(right_legs, [0.0, 0.0, 0.0, 0.0])
    supervisor.step(TIME_STEP*10)
    action4 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    reward4 = compute_reward()
    state5 = get_observations()
    #print("Reward4 : ", reward4)
    
    MEMORY.append((state1, action1, reward1, state2, done))
    MEMORY.append((state2, action2, reward2, state3, done))
    MEMORY.append((state3, action3, reward3, state4, done))
    MEMORY.append((state4, action4, reward4, state5, done))
    
    
# Main loop
for i in range(100):
    reset()
    count = 0
    reward = 0
    print("Iteration :",i)
    while count < 30:
        step_forward()
        count+=1
        print("Count : ",count)
        #print(f"Current Position: {current_position}, Distance to Goal: {distance_to_goal}")
        
        
checkpoint = {
          'replay_buffer': MEMORY
        }
torch.save(checkpoint, "checkpoint_replay.pth")
print("GPS : ",get_gps_position())
print("Simulation ended.")
